# Installation
# before you begin please install VLLM in your remote environment
# pip install vllm-tpu

import os

import kinetic


# We use a TPU accelerator. Adjust as needed (e.g., 'tpu-v5e-1', 'tpu-v5litepod-4')
@kinetic.run(
  accelerator="tpu-v5litepod",
  # Capture Hugging Face token if using a gated model like Gemma
  capture_env_vars=["HF_TOKEN"],
)
def run_vllm_inference():
  # Imports must happen inside the decorated function because they need to run
  # in the remote container where vllm is installed.
  os.environ["VLLM_TARGET_DEVICE"] = "tpu"
  os.environ["JAX_PLATFORMS"] = "tpu,cpu"
  os.environ["VLLM_USE_V1"] = "0"

  from vllm import LLM, SamplingParams

  model_id = "meta-llama/Llama-3.1-8B"

  print(f"Initializing vLLM with model: {model_id}")
  # We use arguments matching the quickstart
  llm = LLM(model=model_id, tensor_parallel_size=4, max_model_len=2048)

  sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

  prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The president of the United States is",
  ]

  print("Generating completions...")
  outputs = llm.generate(prompts, sampling_params)

  # Print the results
  for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"\nPrompt: {prompt!r}")
    print(f"Generated text: {generated_text!r}")


if __name__ == "__main__":
  run_vllm_inference()
