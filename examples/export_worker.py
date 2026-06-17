"""KerasHub -> Hugging Face export. Run as a subprocess by vllm_serving.py so its
device memory is released on exit. Configured via environment variables."""

import os

import keras_hub  # KERAS_BACKEND is set by the parent before this import

preset = os.environ["MODEL_PRESET"]
export_path = os.environ["EXPORT_PATH"]
dtype = os.environ.get("DTYPE", "bfloat16")

print(
  f"[export] backend={os.environ.get('KERAS_BACKEND')} preset={preset}",
  flush=True,
)
lm = keras_hub.models.CausalLM.from_preset(preset, dtype=dtype)
lm.export_to_transformers(export_path)
print(f"[export] done -> {export_path}", flush=True)
