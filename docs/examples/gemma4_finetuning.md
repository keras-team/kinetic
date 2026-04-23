# Fine-tuning Gemma 4 on TPU

This guide walks through fine-tuning [Gemma 4 Instruct 26B](https://www.kaggle.com/models/keras/gemma4) on a TPU slice using Kinetic. You will use Low-Rank Adaptation (LoRA) to reduce memory requirements, save the adapted weights to GCS, and run inference with the fine-tuned model, all from your local machine.

The model used here is `gemma4_instruct_26b_a4b`, a Mixture of Experts (MoE) architecture with 26B total parameters and 4B active parameters per forward pass. All 26B weights load into memory (~52 GB in bfloat16), so a v5litepod-8 (8 chips × 16 GB = 128 GB HBM) is the minimum required configuration. A self-contained script combining both steps is available at [`examples/gemma4_finetuning.py`](../../examples/gemma4_finetuning/gemma4_finetuning.py).

## Prerequisites

Before starting, you need:

- A GCP project with billing enabled.
- A Kinetic cluster provisioned (`kinetic up`). See the [Getting Started](../getting_started.md) guide.
- A **v5litepod-8 TPU node pool** in your cluster. Run `kinetic status` to check what pools you have. If you need to add one:
  ```bash
  kinetic pool add --accelerator tpu-v5litepod-8 --project your-project-id
  ```
- A Kaggle account with [Gemma 4 access accepted](https://www.kaggle.com/models/keras/gemma4).
- `KAGGLE_USERNAME` and `KAGGLE_KEY` set in your local environment.

## GCloud Setup

Authenticate and configure your project:

```bash
gcloud auth login
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_ZONE="us-central1-a"
```

`us-central1-a` reliably has on-demand `v5litepod-8` availability. Before running, verify your zone has the hardware:

```bash
gcloud compute tpus accelerator-types list --zone=your-zone --project=your-project-id
```

Confirm `v5litepod-8` appears in the output before submitting a job.

## Forwarding Credentials

Kaggle credentials must be present in the remote pod to download the model weights. Use `capture_env_vars` to forward them automatically:

```python
import kinetic

@kinetic.run(
    accelerator="tpu-v5litepod-8",
    capture_env_vars=["KAGGLE_*", "GOOGLE_CLOUD_*"],
)
def fine_tune_gemma4():
    ...
```

This pattern is covered in depth in the [Environment Variables](../guides/env_vars.md) guide.

`keras-hub` and its tokenizer backends are not installed in the Kinetic base container by default. Add a `requirements.txt` to your project so Kinetic picks them up automatically:

```
keras==3.13.2
keras-hub==0.27.1
tokenizers==0.22.2
sentencepiece==0.2.1
```

Kinetic detects changes to this file and rebuilds the container only when needed. See the [Managing Dependencies](../guides/dependencies.md) guide for details.

## Fine-tuning with LoRA

The full training function loads the model, enables LoRA, and fits on a small instruction-following dataset. Imports live inside the function so they run on the remote worker.

Four things are worth understanding before reading the code:

**Precision policy.** The 26B model stores ~52 GB of weights. Using `mixed_bfloat16` would keep float32 master copies (~13 GB/chip on 8 chips), which — combined with MoE activation tensors — exceeds the 15.75 GB/chip HBM limit. The `bfloat16` policy stores variables directly as bfloat16 (~6.5 GB/chip), which fits.

**Sequence length.** MoE activation tensors scale with the compiled sequence length. The preset default (~1024 tokens) produces ~10 GB/chip of HLO temporaries. Setting `model.preprocessor.sequence_length = 128` before `compile()` keeps it under ~2 GB/chip.

**Weight sharding.** The 26B model does not fit on a single 16 GB chip. `ModelParallel` with an explicit `LayoutMap` splits weights across all 8 chips at variable creation time. The `LayoutMap` must be set before calling `from_preset()` so that variables are created with the correct sharding specs from the start.

**Custom weight loading.** The Kaggle preset stores weights across 6 sharded H5 files described by a `model.weights.json` manifest. Keras's built-in `load_weights()` on the full `CausalLM` prepends a `backbone/` prefix that mismatches every path in the manifest. Loading via `model.backbone.load_weights()` avoids that prefix, but Keras ≤ 3.14 has a bug in `ShardedH5IOStore`: after switching to a different shard file, the internal `current_shard_path` pointer is not updated. When a subsequent `keys()` call restores to the stale path, layers whose weights span multiple shards — every MoE expert bank and the token embedding — fail to load, producing a "received 0 variables" error. The solution is to bypass `ShardedH5IOStore` entirely and read the H5 files directly with h5py, pre-sharding each tensor with `jax.device_put` before assigning it to avoid a memory spike on device 0. The complete loader is implemented as `_load_sharded_weights()` in [`examples/gemma4_finetuning.py`](../../examples/gemma4_finetuning/gemma4_finetuning.py).

> **TODO:** remove `_load_sharded_weights` once Keras exposes a public loading path that handles the `backbone/` prefix correctly and fixes the `ShardedH5IOStore` shard-switching bug.

The code below assumes `_load_sharded_weights` and `_make_layout_map` are defined as in [`examples/gemma4_finetuning.py`](../../examples/gemma4_finetuning/gemma4_finetuning.py).

```python
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
        keras.distribution.ModelParallel(layout_map=layout_map, batch_dim_name="batch")
    )


@kinetic.run(
    accelerator="tpu-v5litepod-8",
    capture_env_vars=["KAGGLE_*", "GOOGLE_CLOUD_*"],
)
def fine_tune_gemma4():
    import h5py
    import io

    import jax
    import keras
    import keras_hub
    import kagglehub
    import numpy as np

    prompts = [
        "<start_of_turn>user\nExplain what a transformer is in one paragraph.<end_of_turn>\n<start_of_turn>model\n",
        "<start_of_turn>user\nWrite a Python function that reverses a string.<end_of_turn>\n<start_of_turn>model\n",
        # ... more examples
    ]
    responses = [
        "A transformer is a neural network architecture...",
        "def reverse_string(s: str) -> str:\n    return s[::-1]",
        # ...
    ]

    keras.mixed_precision.set_global_policy("bfloat16")
    _make_layout_map(keras)

    print("Loading Gemma 4 Instruct 26B weights (~52 GB, this may take several minutes)...")
    model = keras_hub.models.Gemma4CausalLM.from_preset(
        "gemma4_instruct_26b_a4b",
        load_weights=False,
    )
    model_path = kagglehub.model_download("keras/gemma4/keras/gemma4_instruct_26b_a4b")
    _load_sharded_weights(model.backbone, os.path.join(model_path, "model.weights.json"))

    model.backbone.enable_lora(rank=4)
    print(f"Trainable parameters: {model.count_params():,}")

    model.preprocessor.sequence_length = 128
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5))
    model.fit(x={"prompts": prompts, "responses": responses}, epochs=1, batch_size=1)

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


if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "jax"
    os.environ["GOOGLE_CLOUD_PROJECT"] = "your-project-id"
    os.environ["GOOGLE_CLOUD_ZONE"] = "us-central1-a"

    weights_path = fine_tune_gemma4()
    print(f"Training complete. Weights at: {weights_path}")
```

Only the LoRA adapter variables (a few MB) are saved — not the full 26B backbone. `KINETIC_OUTPUT_DIR` is automatically set to a unique GCS path (e.g. `gs://your-bucket/job-abc123/output/`) for every job. The full path is printed to your terminal so you can pass it to the inference job below.

## Monitoring the Job

While the fine-tuning job is running you can inspect it from a separate terminal using the `kinetic jobs` CLI. All commands require `--project` (or the `KINETIC_PROJECT` env var) to locate your cluster.

List all live jobs:

```bash
kinetic jobs list --project your-project-id
```

Check the status of a specific job — the job ID is printed to your terminal when the job is submitted (e.g. `job-534ffeb6`):

```bash
kinetic jobs status JOB_ID --project your-project-id
```

Stream live logs until the job finishes:

```bash
kinetic jobs logs --follow JOB_ID --project your-project-id
```

If you need to stop the job early:

```bash
kinetic jobs cancel JOB_ID --project your-project-id
```

If the job stays in `PENDING` for more than a few minutes, inspect the pod to diagnose scheduling failures:

```bash
kubectl describe pod -l leaderworkerset.sigs.k8s.io/name=keras-pathways-JOB_ID -n default
```

Check the **Events** section at the bottom — common causes are insufficient TPU quota, no matching node pool for the requested accelerator, or image pull errors.

## Inference with Fine-tuned Weights

After the training job completes, copy the printed weights path and pass it to a separate inference job:

```python
import os
import kinetic


@kinetic.run(
    accelerator="tpu-v5litepod-8",
    capture_env_vars=["KAGGLE_*", "GOOGLE_CLOUD_*"],
)
def run_inference(weights_path: str):
    import h5py
    import io

    import keras
    import keras_hub
    import kagglehub
    import numpy as np

    keras.mixed_precision.set_global_policy("bfloat16")
    _make_layout_map(keras)

    print("Loading Gemma 4 Instruct 26B weights (~52 GB)...")
    model = keras_hub.models.Gemma4CausalLM.from_preset(
        "gemma4_instruct_26b_a4b",
        load_weights=False,
    )
    model_path = kagglehub.model_download("keras/gemma4/keras/gemma4_instruct_26b_a4b")
    _load_sharded_weights(model.backbone, os.path.join(model_path, "model.weights.json"))

    model.backbone.enable_lora(rank=4)
    print(f"Loading LoRA weights from: {weights_path}")

    if weights_path.startswith("gs://"):
        from google.cloud import storage as gcs_storage
        without_scheme = weights_path[5:]
        bucket_name, _, blob_name = without_scheme.partition("/")
        buffer = io.BytesIO()
        gcs_storage.Client().bucket(bucket_name).blob(blob_name).download_to_file(buffer)
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
    os.environ["GOOGLE_CLOUD_ZONE"] = "us-central1-a"

    # Replace with the path printed at the end of the fine-tuning job.
    weights_path = "gs://your-bucket/job-abc123/output/gemma4_lora.weights.h5"
    response = run_inference(weights_path)
    print(response)
```

## Cleaning Up

TPU node pools accrue cost while they exist, even when no job is running. Remove resources when you are done to avoid unnecessary charges.

Remove the v5litepod-8 pool while keeping the cluster intact for other workloads:

```bash
# Find the exact pool name
kinetic pool list --project your-project-id

# Remove it (use the name printed above, e.g. tpu-v5litepod-a1b2)
kinetic pool remove POOL_NAME --project your-project-id
```

To tear down the entire cluster, including all pools, the GKE cluster, and associated infrastructure:

```bash
kinetic down --project your-project-id
```

## Next Steps

- **Checkpointing during training:** use Orbax to save intermediate checkpoints so a long run can resume if interrupted. See the [Checkpointing](../guides/checkpointing.md) guide.
- **Distributed training:** scale to larger TPU slices or multiple hosts. See the [Distributed Training](../guides/distributed_training.md) guide.
