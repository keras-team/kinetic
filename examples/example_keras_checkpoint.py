import os

# Set backend to JAX before any keras imports
os.environ["KERAS_BACKEND"] = "jax"

import kinetic


@kinetic.run(accelerator="cpu")
def train_keras_with_checkpoints():
  """Demo function showing Orbax checkpointing with a Keras model and Auto-Resume."""
  import keras
  import numpy as np
  import orbax.checkpoint as ocp

  output_dir = os.environ.get("KINETIC_OUTPUT_DIR")
  print(f"\n--- Kinetic Output Dir: {output_dir} ---")

  if not output_dir:
    # Fallback for local testing if run without kinetic context
    output_dir = "/tmp/local_keras_checkpoints"
    print(f"No KINETIC_OUTPUT_DIR found, using local: {output_dir}")

  # Define a simple Keras model
  model = keras.Sequential(
    [
      keras.layers.Input(shape=(10,)),
      keras.layers.Dense(32, activation="relu"),
      keras.layers.Dense(1),
    ]
  )
  model.compile(optimizer="adam", loss="mse")

  # Initialize Orbax CheckpointManager
  options = ocp.CheckpointManagerOptions(max_to_keep=2)
  mngr = ocp.CheckpointManager(
    output_dir, ocp.StandardCheckpointer(), options=options
  )

  # Orbax handles discovery + restore natively. model.get_weights() returns a
  # list of numpy arrays, which Orbax treats as a PyTree.
  latest = mngr.latest_step()
  if latest is not None:
    print(f"Found latest checkpoint for epoch {latest}. Restoring...")
    state = mngr.restore(latest)
    model.set_weights(state["weights"])
    start_epoch = latest + 1
  else:
    print("No checkpoint found. Starting from scratch (epoch 0).")
    start_epoch = 0

  print(f"--- Starting from epoch: {start_epoch} ---\n")

  # Dummy data
  x_train = np.random.randn(256, 10).astype("float32")
  y_train = np.random.randn(256, 1).astype("float32")

  # Simulated training loop (run 3 epochs)
  end_epoch = start_epoch + 3
  print(f"Will run epochs from {start_epoch} to {end_epoch - 1}")

  for epoch in range(start_epoch, end_epoch):
    print(f"\n>> Training epoch {epoch}...")
    history = model.fit(x_train, y_train, epochs=1, verbose=0)
    loss = history.history["loss"][-1]
    print(f"epoch {epoch}: loss={loss:.4f}")

    state = {
      "epoch": epoch,
      "weights": model.get_weights(),
    }

    print(f"Saving checkpoint at epoch {epoch}...")
    mngr.save(epoch, state)
    mngr.wait_until_finished()
    print(f"Checkpoint for epoch {epoch} saved successfully.")

  # Verify by restoring the latest step
  latest_step = mngr.latest_step()
  print(f"\nVerifying by restoring latest epoch ({latest_step})...")
  if latest_step is not None:
    restored_state = mngr.restore(latest_step)
    model.set_weights(restored_state["weights"])
    assert restored_state["epoch"] == latest_step
    print(f"Verified: Restored model weights match epoch {latest_step}!")

  return True


if __name__ == "__main__":
  print("Starting Keras + Orbax checkpointing demo...")
  success = train_keras_with_checkpoints()
  print(f"Demo run success: {success}")
