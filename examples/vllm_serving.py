"""Export a KerasHub causal LM to the Hugging Face Transformers format
and serve it with vLLM on a Kinetic GPU, in a single job.

See the accompanying guide for prerequisites and configuration.
"""

import os

import kinetic

# Any preset with a Transformers exporter works (Gemma 3 / Gemma /
# Qwen / GPT-2).
MODEL_PRESET = "gemma3_4b"

# Load and export in the model's native precision; "float32" for GPT-2
# or pre-Ampere GPUs.
DTYPE = "bfloat16"

EXPORT_DIR = "/tmp/hf_export"

# vLLM routes checkpoints via the `architectures` key in config.json.
# The Gemma 3 exporter writes it natively; the others don't yet, so we
# patch it after export. TODO: drop once all exporters in
# `keras_hub/src/utils/transformers/export/` write this key.
HF_ARCHITECTURES = {
  "gpt2": "GPT2LMHeadModel",
  "qwen2": "Qwen2ForCausalLM",
  "gemma": "GemmaForCausalLM",
  "gemma3_text": "Gemma3ForCausalLM",
}


def _ensure_architectures(export_dir):
  """Add the `architectures` key to config.json if missing."""
  import json

  config_path = os.path.join(export_dir, "config.json")
  with open(config_path) as f:
    config = json.load(f)
  if "architectures" not in config:
    arch = HF_ARCHITECTURES.get(config.get("model_type"))
    if arch is None:
      raise ValueError(
        f"Unknown model_type {config.get('model_type')!r}; "
        "add it to HF_ARCHITECTURES."
      )
    config["architectures"] = [arch]
    with open(config_path, "w") as f:
      json.dump(config, f, indent=2)


def _setup_gpu_runtime():
  """Expose the GKE-mounted NVIDIA driver to torch/vLLM and select
  the torch-native sampler (pip-only images have no nvcc)."""
  import ctypes
  import glob

  print("GPU devices:", glob.glob("/dev/nvidia*"))

  # For vLLM's spawned engine processes, which read this at startup.
  nvidia_dirs = [
    d
    for d in ("/usr/local/nvidia/lib64", "/usr/local/nvidia/lib")
    if os.path.isdir(d)
  ]
  if nvidia_dirs:
    prev = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(
      nvidia_dirs + ([prev] if prev else [])
    )

  # For this process, whose linker ignores late LD_LIBRARY_PATH edits.
  for lib in ("libcuda.so.1", "libnvidia-ml.so.1"):
    for root in ("/usr/local/nvidia/lib64", "/usr/local/nvidia/lib"):
      path = os.path.join(root, lib)
      if os.path.exists(path):
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        print("preloaded:", path)
        break
    else:
      print("NOT FOUND:", lib)

  os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"


def _persist_export(export_dir):
  """Upload the exported checkpoint directory to the job's durable
  output dir."""
  import shutil

  output_dir = os.environ.get("KINETIC_OUTPUT_DIR")
  if not output_dir:
    return None
  dest = f"{output_dir.rstrip('/')}/hf_export"
  if dest.startswith("gs://"):
    # Direct directory upload: no tar copy on ephemeral disk, and
    # parallel workers make it fast.
    from google.cloud import storage
    from google.cloud.storage import transfer_manager

    bucket_name, _, blob_prefix = dest[5:].partition("/")
    bucket = storage.Client().bucket(bucket_name)
    files = []
    for root, _, filenames in os.walk(export_dir):
      for filename in filenames:
        rel_path = os.path.relpath(os.path.join(root, filename), export_dir)
        files.append(rel_path)
    print("Uploading exported checkpoint to GCS...")
    transfer_manager.upload_many_from_filenames(
      bucket,
      files,
      source_directory=export_dir,
      blob_name_prefix=blob_prefix + "/",
      worker_type=transfer_manager.THREAD,
      raise_exception=True,
    )
  else:
    shutil.copytree(export_dir, dest, dirs_exist_ok=True)
  return dest


@kinetic.run(
  accelerator="gpu-l4",
  capture_env_vars=["KAGGLE_*", "GOOGLE_CLOUD_*"],
)
def export_and_serve(prompts):
  _setup_gpu_runtime()

  # torch is the only backend that frees VRAM for vLLM in the same
  # process; keras reads this at first (transitive) import.
  os.environ["KERAS_BACKEND"] = "torch"

  import gc

  import keras_hub
  import torch

  # Forwarded via capture_env_vars; ~/.kaggle/kaggle.json doesn't
  # travel to the pod.
  print(
    "Kaggle creds in pod:",
    bool(os.environ.get("KAGGLE_USERNAME"))
    and bool(os.environ.get("KAGGLE_KEY")),
  )

  print(f"Loading {MODEL_PRESET} from KerasHub ({DTYPE})...")
  lm = keras_hub.models.CausalLM.from_preset(MODEL_PRESET, dtype=DTYPE)

  print("Exporting to Hugging Face Transformers format...")
  lm.export_to_transformers(EXPORT_DIR)
  _ensure_architectures(EXPORT_DIR)

  # Persist before starting vLLM: engine init is the riskiest step,
  # and the checkpoint survives even if it crashes.
  artifact = _persist_export(EXPORT_DIR)
  if artifact:
    print(f"Exported model uploaded to: {artifact}")

  # Release VRAM before handing the GPU to vLLM.
  del lm
  gc.collect()
  torch.cuda.empty_cache()

  from vllm import LLM, SamplingParams

  print("Starting vLLM engine from the exported checkpoint...")
  llm = LLM(
    model=EXPORT_DIR,
    dtype=DTYPE,
    gpu_memory_utilization=0.85,
    max_model_len=1024,
    enforce_eager=False,
  )
  sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=128)

  # All prompts are processed in one continuous-batching pass.
  outputs = llm.generate(prompts, sampling_params)
  results = [
    {"prompt": o.prompt, "completion": o.outputs[0].text.strip()}
    for o in outputs
  ]

  return results


if __name__ == "__main__":
  prompts = [
    "The future of artificial intelligence will involve",
    "A short recipe for a perfect weekend:",
    "In one sentence, the theory of relativity says",
    "The most underrated skill in software engineering is",
  ]
  for result in export_and_serve(prompts):
    print("=" * 60)
    print(f"Prompt: {result['prompt']}")
    print(f"Completion: {result['completion']}")
