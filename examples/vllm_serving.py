"""Export a KerasHub model and serve it with vLLM on a Cloud TPU or GPU, in a
single Kinetic job. See docs/guides/vllm_serving.md."""

import importlib.util
import os
import subprocess
import sys

import kinetic

DEVICE = "tpu"  # "tpu" | "gpu"
BACKEND = (
  "jax" if DEVICE == "tpu" else "torch"
)  # "jax" | "torch" | "tensorflow"
MODEL_PRESET = "gemma3_4b"
DTYPE = "bfloat16"
EXPORT_DIR = "/tmp/hf_export"

# Any slice/GPU that fits your model works, e.g. "tpu-v5litepod-4", "gpu-a100".
ACCELERATOR = "tpu-v5litepod-1" if DEVICE == "tpu" else "gpu-l4"


def _setup_gpu_runtime():
  # Expose the GKE NVIDIA driver so the export child (which inherits
  # LD_LIBRARY_PATH) and vLLM can use the GPU.
  import ctypes

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
  for lib in ("libcuda.so.1", "libnvidia-ml.so.1"):
    for root in nvidia_dirs:
      if os.path.exists(os.path.join(root, lib)):
        ctypes.CDLL(os.path.join(root, lib), mode=ctypes.RTLD_GLOBAL)
        break
  os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"


@kinetic.run(
  accelerator=ACCELERATOR,
  capture_env_vars=["KAGGLE_*", "GOOGLE_CLOUD_*"],
)
def export_and_serve(prompts):
  if DEVICE == "gpu":
    _setup_gpu_runtime()

  # Export in a child process; on exit the OS frees its device memory. find_spec
  # locates the worker on the pod without importing it (which would defeat the
  # isolation).
  worker = importlib.util.find_spec("export_worker").origin
  child_env = {
    **os.environ,
    "KERAS_BACKEND": BACKEND,
    "MODEL_PRESET": MODEL_PRESET,
    "EXPORT_PATH": EXPORT_DIR,
    "DTYPE": DTYPE,
  }
  subprocess.run([sys.executable, worker], env=child_env, check=True)

  if DEVICE == "tpu":
    os.environ["VLLM_TARGET_DEVICE"] = "tpu"
    os.environ["JAX_PLATFORMS"] = "tpu,cpu"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = (
      "0"  # in-process engine, no fork
    )
    serve_kwargs = dict(max_model_len=1024, tensor_parallel_size=1)
  else:
    serve_kwargs = dict(max_model_len=1024, gpu_memory_utilization=0.85)

  from vllm import LLM, SamplingParams

  llm = LLM(
    model=EXPORT_DIR, load_format="safetensors", dtype=DTYPE, **serve_kwargs
  )
  sampling = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=128)
  outputs = llm.generate(prompts, sampling)
  return [
    {"prompt": o.prompt, "completion": o.outputs[0].text.strip()}
    for o in outputs
  ]


if __name__ == "__main__":
  prompts = [
    "The future of artificial intelligence will involve",
    "A short recipe for a perfect weekend:",
    "In one sentence, the theory of relativity says",
  ]
  for r in export_and_serve(prompts):
    print("=" * 60)
    print("Prompt:", r["prompt"])
    print("Completion:", r["completion"])
