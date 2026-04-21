"""Fine-tune Gemma 4 Instruct 26B on TPU with LoRA, then run inference.

The model is gemma4_instruct_26b_a4b — a Mixture of Experts architecture with
26B total parameters and 4B active per forward pass. All 26B weights load into
memory (~52 GB in bfloat16), so a v5litepod-8 (8 chips × 16 GB = 128 GB HBM)
is the minimum supported configuration.

Set the placeholder values in __main__ before running.
See docs/guides/gemma4_finetuning.md for a full walkthrough.

Dependencies are declared in examples/requirements.txt, which Kinetic
auto-discovers and installs in the remote container.
"""

import os

import kinetic


def _make_layout_map(keras):
  """Build the ModelParallel layout map for Gemma4 26B-A4B."""
  import numpy as np

  devices = keras.distribution.list_devices()
  mesh = keras.distribution.DeviceMesh(
    shape=(1, len(devices)),
    axis_names=["batch", "model"],
    devices=np.array(devices).reshape(1, len(devices)),
  )
  layout_map = keras.distribution.LayoutMap(mesh)
  layout_map[".*moe_expert_bank/gate_proj"] = (None, None, "model")
  layout_map[".*moe_expert_bank/up_proj"] = (None, None, "model")
  layout_map[".*moe_expert_bank/down_proj"] = (None, None, "model")
  layout_map[".*query/kernel"] = ("model", None, None)
  layout_map[".*key/kernel"] = (None, "model", None)
  layout_map[".*value/kernel"] = (None, "model", None)
  layout_map[".*attention_output/kernel"] = ("model", None, None)
  layout_map[".*ffw_gating/kernel"] = (None, "model")
  layout_map[".*ffw_gating_2/kernel"] = (None, "model")
  layout_map[".*ffw_linear/kernel"] = ("model", None)
  layout_map[".*per_layer_input_gate/kernel"] = (None, "model")
  layout_map[".*per_layer_up_proj/kernel"] = (None, "model")
  layout_map[".*token_embedding/embeddings"] = ("model", None)
  keras.distribution.set_distribution(
    keras.distribution.ModelParallel(
      layout_map=layout_map, batch_dim_name="batch"
    )
  )


def _load_sharded_weights(backbone, manifest_path):
  """Load backbone weights directly from sharded H5 files.

  Bypasses Keras's ShardedH5IOStore to avoid a shard-switching bug
  (current_shard_path not updated after shard switch) that causes "received 0
  variables" failures when layer weights span multiple shard files.
  """
  import json
  import pathlib

  import h5py
  import jax
  import numpy as np
  from keras.src.saving import saving_lib
  from keras.src.saving.keras_saveable import KerasSaveable
  from keras.src.utils import naming

  try:
    import ml_dtypes
  except ImportError:
    ml_dtypes = None

  manifest_path = pathlib.Path(manifest_path)
  with open(manifest_path) as f:
    config = json.load(f)
  weight_map = config["weight_map"]

  _shards = {}

  def _shard(filename):
    if filename not in _shards:
      _shards[filename] = h5py.File(manifest_path.parent / filename, "r")
    return _shards[filename]

  def _read_var(h5_path, var_index):
    map_key = f"/{h5_path}/vars"
    filenames = weight_map.get(map_key)
    if filenames is None:
      return None
    if not isinstance(filenames, list):
      filenames = [filenames]
    str_idx = str(var_index)
    for filename in filenames:
      f = _shard(filename)
      try:
        vars_grp = f[h5_path]["vars"]
      except KeyError:
        continue
      if str_idx not in vars_grp:
        continue
      ds = vars_grp[str_idx]
      if (
        hasattr(ds, "attrs")
        and "dtype" in ds.attrs
        and ds.attrs["dtype"] == "bfloat16"
        and ml_dtypes is not None
      ):
        return np.array(ds, dtype=ml_dtypes.bfloat16)
      return np.array(ds)
    return None

  visited = set()
  counts = {"loaded": 0, "skipped": 0}

  def _load_layer(layer, h5_path):
    if id(layer) in visited:
      return
    visited.add(id(layer))

    all_vars = list(getattr(layer, "_trainable_variables", [])) + list(
      getattr(layer, "_non_trainable_variables", [])
    )
    for i, var in enumerate(all_vars):
      value = _read_var(h5_path, i)
      if value is not None:
        sharded = jax.device_put(value, var.value.sharding)
        var.assign(sharded)
        counts["loaded"] += 1
      else:
        counts["skipped"] += 1

    for child_attr, child_obj in saving_lib._walk_saveable(layer):
      child_path = f"{h5_path}/{child_attr}" if h5_path else child_attr
      if isinstance(child_obj, KerasSaveable):
        _load_layer(child_obj, child_path)
      elif isinstance(child_obj, (list, dict, tuple, set)):
        _load_container(child_obj, child_path)

  def _load_container(container, h5_path):
    used_names = {}
    items = (
      list(container.values())
      if isinstance(container, dict)
      else list(container)
    )
    for item in items:
      if not isinstance(item, KerasSaveable):
        continue
      cls_name = naming.to_snake_case(item.__class__.__name__)
      if cls_name in used_names:
        used_names[cls_name] += 1
        name = f"{cls_name}_{used_names[cls_name]}"
      else:
        used_names[cls_name] = 0
        name = cls_name
      item_path = f"{h5_path}/{name}" if h5_path else name
      _load_layer(item, item_path)

  _load_layer(backbone, "")

  for f in _shards.values():
    f.close()

  print(
    f"Sharded weight load complete: {counts['loaded']} variables assigned, "
    f"{counts['skipped']} paths not in weight map."
  )


@kinetic.run(
  accelerator="tpu-v5litepod-8",
  capture_env_vars=["KAGGLE_*", "GOOGLE_CLOUD_*"],
)
def fine_tune_gemma4():
  import io

  import h5py
  import jax
  import kagglehub
  import keras
  import keras_hub
  import numpy as np

  prompts = [
    "<start_of_turn>user\nExplain what a transformer is in one paragraph.<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nWrite a Python function that reverses a string.<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nWhat is gradient descent?<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nWhat is the difference between a Python list and a tuple?<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nExplain backpropagation in simple terms.<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nWrite SQL to find duplicate rows in a table.<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nWhat is overfitting and how do you prevent it?<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nExplain the attention mechanism.<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nWrite a Python regex to validate an email address.<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nWhat is a learning rate and how does it affect training?<end_of_turn>\n<start_of_turn>model\n",
  ]
  responses = [
    "A transformer is a neural network architecture introduced in the paper 'Attention Is All You Need'. It relies entirely on self-attention mechanisms to compute representations of its input and output, dispensing with recurrence and convolutions. Transformers process all tokens in parallel, making them highly efficient on modern hardware.",
    "def reverse_string(s: str) -> str:\n    return s[::-1]",
    "Gradient descent is an optimization algorithm that iteratively adjusts a model's parameters in the direction that minimizes a loss function. At each step it computes the gradient of the loss with respect to each parameter and moves a small step (the learning rate) in the opposite direction.",
    "Lists are mutable — you can add, remove, or change elements after creation. Tuples are immutable — once created their contents cannot be modified. Tuples are generally faster and use less memory, and are used for fixed collections of values.",
    "Backpropagation is the algorithm used to compute gradients of the loss with respect to each weight in a neural network. It works by applying the chain rule layer by layer from the output back to the input, so each layer's gradient is computed from the gradient of the layer above it.",
    "SELECT col1, col2, COUNT(*) AS cnt\nFROM my_table\nGROUP BY col1, col2\nHAVING COUNT(*) > 1;",
    "Overfitting happens when a model learns the training data too well, including its noise, so it performs poorly on unseen data. Common remedies include: adding dropout layers, using L2 regularization, early stopping, data augmentation, and collecting more training data.",
    "The attention mechanism allows a model to dynamically focus on different parts of the input when producing each output token. For every token, it computes a weighted sum of all other tokens' representations, where the weights reflect how relevant each token is to the current one.",
    "import re\n\ndef is_valid_email(email: str) -> bool:\n    pattern = r'^[\\w.+-]+@[\\w-]+\\.[\\w.-]+$'\n    return bool(re.fullmatch(pattern, email))",
    "The learning rate controls how large a step gradient descent takes at each update. A rate that is too high causes the loss to oscillate or diverge; too low means slow convergence. Common strategies include starting with a moderate rate (e.g. 5e-5) and decaying it over training.",
  ]

  keras.mixed_precision.set_global_policy("bfloat16")
  _make_layout_map(keras)

  print(
    "Loading Gemma 4 Instruct 26B weights (~52 GB, this may take several minutes)..."
  )
  model = keras_hub.models.Gemma4CausalLM.from_preset(
    "gemma4_instruct_26b_a4b",
    load_weights=False,
  )
  model_path = kagglehub.model_download(
    "keras/gemma4/keras/gemma4_instruct_26b_a4b"
  )
  _load_sharded_weights(
    model.backbone, os.path.join(model_path, "model.weights.json")
  )

  model.backbone.enable_lora(rank=4)
  print(f"Trainable parameters: {model.count_params():,}")

  model.preprocessor.sequence_length = 128
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5))
  model.fit(
    x={"prompts": prompts, "responses": responses}, epochs=1, batch_size=1
  )

  output_dir = os.environ.get("KINETIC_OUTPUT_DIR", "/tmp/gemma4_lora")
  weights_path = f"{output_dir}/gemma4_lora.weights.h5"

  buffer = io.BytesIO()
  with h5py.File(buffer, "w") as f:
    for var in model.trainable_variables:
      val = np.asarray(jax.device_get(var.value), dtype=np.float32)
      f.create_dataset(var.path, data=val)

  if weights_path.startswith("gs://"):
    from google.cloud import storage as gcs_storage

    without_scheme = weights_path[5:]
    bucket_name, _, blob_name = without_scheme.partition("/")
    blob = gcs_storage.Client().bucket(bucket_name).blob(blob_name)
    buffer.seek(0)
    blob.upload_from_file(buffer, content_type="application/x-hdf5")
  else:
    os.makedirs(output_dir, exist_ok=True)
    with open(weights_path, "wb") as out_f:
      out_f.write(buffer.getvalue())

  print(f"LoRA weights saved to: {weights_path}")
  return weights_path


@kinetic.run(
  accelerator="tpu-v5litepod-8",
  capture_env_vars=["KAGGLE_*", "GOOGLE_CLOUD_*"],
)
def run_inference(weights_path: str):
  import io

  import h5py
  import kagglehub
  import keras
  import keras_hub
  import numpy as np

  keras.mixed_precision.set_global_policy("bfloat16")
  _make_layout_map(keras)

  print("Loading Gemma 4 Instruct 26B weights (~52 GB)...")
  model = keras_hub.models.Gemma4CausalLM.from_preset(
    "gemma4_instruct_26b_a4b",
    load_weights=False,
  )
  model_path = kagglehub.model_download(
    "keras/gemma4/keras/gemma4_instruct_26b_a4b"
  )
  _load_sharded_weights(
    model.backbone, os.path.join(model_path, "model.weights.json")
  )

  model.backbone.enable_lora(rank=4)
  print(f"Loading LoRA weights from: {weights_path}")

  if weights_path.startswith("gs://"):
    from google.cloud import storage as gcs_storage

    without_scheme = weights_path[5:]
    bucket_name, _, blob_name = without_scheme.partition("/")
    buffer = io.BytesIO()
    gcs_storage.Client().bucket(bucket_name).blob(blob_name).download_to_file(
      buffer
    )
    buffer.seek(0)
    h5_source = buffer
  else:
    h5_source = weights_path

  path_to_var = {var.path: var for var in model.trainable_variables}
  with h5py.File(h5_source, "r") as f:
    for path, var in path_to_var.items():
      if path in f:
        var.assign(np.array(f[path]))

  prompt = (
    "<start_of_turn>user\n"
    "Explain what a transformer is in one paragraph."
    "<end_of_turn>\n<start_of_turn>model\n"
  )
  output = model.generate([prompt], max_length=256)
  return output[0]


if __name__ == "__main__":
  os.environ["KERAS_BACKEND"] = "jax"
  os.environ["GOOGLE_CLOUD_PROJECT"] = "your-project-id"
  os.environ["KINETIC_ZONE"] = "us-central1-a"
  os.environ["GOOGLE_CLOUD_ZONE"] = "us-central1-a"

  weights_path = fine_tune_gemma4()
  print(f"Training complete. Weights at: {weights_path}")

  response = run_inference(weights_path)
  print(f"\nModel response:\n{response}")
