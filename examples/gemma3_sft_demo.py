import os

# JAX must be set as the backend before importing Keras
os.environ["KERAS_BACKEND"] = "jax"

import keras_hub

import kinetic


@kinetic.run(
  accelerator="tpu-v5litepod-1",
  capture_env_vars=["KAGGLE_USERNAME", "KAGGLE_KEY"],
)
def train_gemma():
  # Data for SFT
  print("Starting Gemma 3 SFT training...")
  features = {
    "prompts": ["Capital of India?", "Capital of South Africa?"],
    "responses": ["New Delhi", "Pretoria"],
  }
  print("Data prepared.")
  gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_1b")
  print("Model initialized.")
  # Fine-tune
  gemma_lm.fit(x=features, batch_size=1)

  print("Gemma 3 SFT training done")


if __name__ == "__main__":
  train_gemma()
