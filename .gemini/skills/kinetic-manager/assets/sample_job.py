import os

import absl.logging

import kinetic

# Optional: Set your Keras backend
os.environ["KERAS_BACKEND"] = "jax"


@kinetic.submit(accelerator="tpu-v5e-1")
def train_model(data_dir):
  import keras
  import numpy as np

  # data_dir is resolved to a local path on the remote pod
  absl.logging.info(f"Loading data from {data_dir}")

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

  # Execute remotely via submit.
  handle = train_model(kinetic.Data(local_data_path))
  absl.logging.info(f"Submitted job with ID: {handle.job_id}")

  # Showcase async methods.
  # 1. Check status.
  status = handle.status()
  absl.logging.info(f"Initial job status: {status}")

  # 2. Get logs (non-blocking).
  logs = handle.logs(follow=False)
  absl.logging.info(f"Fetched logs (length: {len(logs) if logs else 0}).")

  # 3. Wait for result without automatic cleanup to showcase explicit cleanup.
  try:
    final_loss = handle.result(timeout=600, cleanup=False)
    absl.logging.info(f"Remote training complete. Final loss: {final_loss}")
  except TimeoutError:
    absl.logging.info("Job timed out. Cancelling...")
    handle.cancel()
    raise

  # 4. Explicit cleanup.
  absl.logging.info("Cleaning up job resources...")
  handle.cleanup()
