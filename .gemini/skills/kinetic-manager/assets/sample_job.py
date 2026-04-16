import os

from absl import logging

import kinetic
from kinetic import Data


# Optional: Set your Keras backend
os.environ["KERAS_BACKEND"] = "jax"


@kinetic.run(accelerator="tpu-v5e-1")
def train_model(data_dir):
  import keras
  import numpy as np

  # data_dir is resolved to a local path on the remote pod
  logging.info(f"Loading data from {data_dir}")

  # Build a simple model
  model = keras.Sequential(
    [
      keras.layers.Dense(64, activation="relu", input_shape=(10,)),
      keras.layers.Dense(1),
    ]
  )
  model.compile(optimizer="adam", loss="mse")

  # Generate dummy data
  rng = np.random.default_rng()
  x_train = rng.standard_normal((100, 10))
  y_train = rng.standard_normal((100, 1))

  # Train
  history = model.fit(x_train, y_train, epochs=5)

  return history.history["loss"][-1]


if __name__ == "__main__":
  # Path to your local data
  local_data_path = "./my_dataset"

  # Ensure local data directory exists for the demo
  if not os.path.exists(local_data_path):
    os.makedirs(local_data_path, exist_ok=True)
    with open(os.path.join(local_data_path, "info.txt"), "w") as f:
      f.write("Sample dataset info")

  # Execute remotely
  final_loss = train_model(Data(local_data_path))
  logging.info(f"Remote training complete. Final loss: {final_loss}")
