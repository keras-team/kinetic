import os

import keras_hub

from kinetic import core as kinetic


@kinetic.run(
  accelerator="v5litepod-1", capture_env_vars=["KAGGLE_*", "GOOGLE_CLOUD_*"]
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
  # Set environment variables for TPU
  os.environ["KERAS_BACKEND"] = "jax"
  # set environment variables for gcp
  os.environ["GOOGLE_CLOUD_PROJECT"] = "tpu-prod-123456"
  os.environ["GOOGLE_CLOUD_ZONE"] = "us-central1-a"
  # set environment variables for kaggle
  os.environ["KAGGLE_USERNAME"] = "your_kaggle_username"
  os.environ["KAGGLE_KEY"] = "your_kaggle_key"

  train_gemma()
