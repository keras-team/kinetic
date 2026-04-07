import os

# Set backend to JAX before any keras imports
os.environ["KERAS_BACKEND"] = "jax"

import kinetic


@kinetic.run(accelerator="cpu")
def train_with_checkpoints():
  """Demo function showing Orbax checkpointing with Kinetic and Auto-Resume."""
  import jax.numpy as jnp
  import orbax.checkpoint as ocp

  output_dir = os.environ.get("KINETIC_OUTPUT_DIR")
  print(f"\n--- Kinetic Output Dir: {output_dir} ---")

  if not output_dir:
    # Fallback for local testing if run without kinetic context
    output_dir = "/tmp/local_checkpoints"
    print(f"No KINETIC_OUTPUT_DIR found, using local: {output_dir}")

  # Initialize Orbax CheckpointManager
  options = ocp.CheckpointManagerOptions(max_to_keep=2)
  mngr = ocp.CheckpointManager(
    output_dir, ocp.StandardCheckpointer(), options=options
  )

  # Orbax handles discovery + restore natively
  latest = mngr.latest_step()
  if latest is not None:
    print(f"Found latest checkpoint for step {latest}. Restoring...")
    state = mngr.restore(latest)
    start_step = latest + 1
  else:
    print("No checkpoint found. Starting from scratch (step 0).")
    state = {
      "step": 0,
      "weights": jnp.ones((10, 10)),
      "bias": jnp.zeros((10,)),
    }
    start_step = 0

  print(f"--- Starting from step: {start_step} ---\n")

  # Simulated training loop (run 3 steps)
  end_step = start_step + 3
  print(f"Will run steps from {start_step} to {end_step - 1}")

  for step in range(start_step, end_step):
    print(f"\n>> Simulating Step {step}...")
    state["step"] = step
    # Change weights so we can see they resume correctly
    state["weights"] = jnp.ones((10, 10)) * (step + 1)

    print(f"Saving checkpoint at step {step}...")
    mngr.save(step, state)
    mngr.wait_until_finished()
    print(f"Checkpoint for step {step} saved successfully.")

  # Verify by restoring the latest step
  latest_step = mngr.latest_step()
  print(f"\nVerifying by restoring latest step ({latest_step})...")
  if latest_step is not None:
    restored_state = mngr.restore(latest_step)
    assert restored_state["step"] == latest_step
    print(f"Verified: Restored state step matches latest step {latest_step}!")

  return True


if __name__ == "__main__":
  print("Starting Orbax checkpointing demo...")
  success = train_with_checkpoints()
  print(f"Demo run success: {success}")
