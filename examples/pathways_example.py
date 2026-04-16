import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np
from keras import layers

import kinetic


# A simple model that will be executed remotely on pathways.
# Multi-host TPU slices (here: v6e-16 = 4x4 across 4 nodes) auto-select the
# Pathways backend, so an explicit `backend="pathways"` is not needed.
@kinetic.run(accelerator="tpu-v6e-16")
def train_simple_model():
  import jax
  from jax import lax

  print("Running Pathways job on JAX Backend!")

  # Verify distributed JAX setup (Pathways auto-initialization)
  process_count = jax.process_count()
  process_index = jax.process_index()
  device_count = jax.device_count()
  local_device_count = jax.local_device_count()

  print("JAX Distributed Environment:")
  print(f"  Process Count: {process_count}")
  print(f"  Process Index: {process_index}")
  print(f"  Total Devices: {device_count}")
  print(f"  Local Devices: {local_device_count}")

  # Fail if not actually running on multiple hosts
  if process_count <= 1:
    raise RuntimeError(
      f"Pathways verification failed: Expected > 1 processes, but found {process_count}. "
      "This indicates the job is NOT running in a multi-host Pathways environment."
    )

  # Verify collective communication (cross-host psum)
  try:
    # Use jax.pmap to sum values across all devices in the cluster
    x = np.ones(local_device_count)
    distributed_sum = jax.pmap(lambda val: lax.psum(val, "i"), axis_name="i")(x)
    total_sum = distributed_sum[0]

    if total_sum != device_count:
      raise RuntimeError(
        f"Collective verification failed: Expected psum {device_count}, got {total_sum}"
      )
    print(
      f"Successfully verified collective communication across all {total_sum} devices!"
    )
  except Exception as e:
    print(f"Warning: Collective verification failed: {e}")
    if isinstance(e, RuntimeError) and "Collective verification failed" in str(
      e
    ):
      raise

  # Create a simple dataset
  x = np.random.rand(1000, 10)
  y = np.random.randint(0, 2, size=(1000, 1))

  # A simple sequential model
  model = keras.Sequential(
    [
      keras.Input(shape=(10,)),
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(1, activation="sigmoid"),
    ]
  )

  model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
  )

  print("Model Architecture:")
  model.summary()

  # Train the model
  print("\nStarting Training...")
  history = model.fit(x, y, epochs=5, batch_size=32, validation_split=0.2)

  print("\nTraining completed successfully on Pathways!")
  return history.history


if __name__ == "__main__":
  print("Submitting Pathways training job...")
  result_history = train_simple_model()
  print("Final validation accuracy:", result_history["val_accuracy"][-1])
