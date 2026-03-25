"""
Example: Async Jobs with Kinetic

This demonstrates the submit/attach/list workflow for detached execution.
Instead of blocking until the remote function finishes (@kinetic.run),
@kinetic.submit returns a JobHandle immediately so you can monitor,
reattach from another session, or manage multiple jobs concurrently.

Prerequisites:
1. A GKE cluster with CPU node pool (default setup works)
2. kubectl configured to access the cluster
3. KINETIC_PROJECT environment variable set

Workflow overview:
    1. submit()   → fire-and-forget, get a JobHandle back instantly
    2. status()   → poll the job without blocking
    3. logs()     → fetch or stream logs
    4. result()   → block until completion and collect the return value
    5. attach()   → reattach to a job from a different session using its ID
    6. list_jobs()→ discover all live jobs on the cluster
"""

import os
import time

os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np

import kinetic

# ---------------------------------------------------------------------------
# 1. Define functions using @kinetic.submit (same params as @kinetic.run)
# ---------------------------------------------------------------------------


@kinetic.submit(accelerator="cpu")
def train_model_a():
  """Train a small dense model — returns final loss."""
  model = keras.Sequential(
    [
      keras.layers.Dense(64, activation="relu", input_shape=(10,)),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(1),
    ]
  )
  model.compile(optimizer="adam", loss="mse")

  x = np.random.randn(1000, 10)
  y = np.random.randn(1000, 1)

  print("Training model A...")
  history = model.fit(x, y, epochs=5, batch_size=32, verbose=1)
  final_loss = history.history["loss"][-1]
  print(f"Model A done — loss: {final_loss}")
  return final_loss


@kinetic.submit(accelerator="cpu")
def train_model_b():
  """Train a slightly larger model — returns final loss."""
  model = keras.Sequential(
    [
      keras.layers.Dense(128, activation="relu", input_shape=(20,)),
      keras.layers.Dense(128, activation="relu"),
      keras.layers.Dense(1),
    ]
  )
  model.compile(optimizer="adam", loss="mse")

  x = np.random.randn(2000, 20)
  y = np.random.randn(2000, 1)

  print("Training model B...")
  history = model.fit(x, y, epochs=10, batch_size=64, verbose=1)
  final_loss = history.history["loss"][-1]
  print(f"Model B done — loss: {final_loss}")
  return final_loss


# ---------------------------------------------------------------------------
# 2. Submit both jobs (non-blocking)
# ---------------------------------------------------------------------------


def demo_submit_and_monitor():
  """Submit two jobs and monitor them until completion."""
  print("=" * 60)
  print("Submitting two training jobs...")
  print("=" * 60)

  job_a = train_model_a()
  job_b = train_model_b()

  print(f"\nJob A: id={job_a.job_id}")
  print(f"Job B: id={job_b.job_id}")

  # -----------------------------------------------------------------
  # 3. Poll status until both are running
  # -----------------------------------------------------------------
  print("\n--- Polling status ---")
  for _ in range(30):
    sa = job_a.status()
    sb = job_b.status()
    print(f"  A: {sa.value}  |  B: {sb.value}")
    if sa.value in ("RUNNING", "SUCCEEDED") and sb.value in (
      "RUNNING",
      "SUCCEEDED",
    ):
      break
    time.sleep(5)

  # -----------------------------------------------------------------
  # 4. Tail recent logs
  # -----------------------------------------------------------------
  print("\n--- Last 20 log lines from Job A ---")
  print(job_a.tail(n=20))

  # -----------------------------------------------------------------
  # 5. Collect results (blocks until done)
  # -----------------------------------------------------------------
  print("\n--- Collecting results ---")
  loss_a = job_a.result(cleanup=False)
  loss_b = job_b.result(cleanup=False)
  print(f"Job A loss: {loss_a}")
  print(f"Job B loss: {loss_b}")

  return job_a, job_b


# ---------------------------------------------------------------------------
# 6. Reattach from a "different session"
# ---------------------------------------------------------------------------


def demo_reattach(job_id: str):
  """Simulate reattaching to a completed job from another session."""
  print("\n" + "=" * 60)
  print(f"Reattaching to job {job_id} ...")
  print("=" * 60)

  job = kinetic.attach(job_id)
  print(f"Status: {job.status().value}")
  print(f"Function: {job.func_name}")
  print(f"Backend:  {job.backend}")


# ---------------------------------------------------------------------------
# 7. List all live jobs
# ---------------------------------------------------------------------------


def demo_list_jobs():
  """List every kinetic job currently visible on the cluster."""
  print("\n" + "=" * 60)
  print("Listing live jobs...")
  print("=" * 60)

  jobs = kinetic.list_jobs()
  if not jobs:
    print("  (no live jobs)")
  for j in jobs:
    print(f"  {j.job_id}  {j.func_name:30s}  {j.accelerator}")


# ---------------------------------------------------------------------------
# 8. Cleanup
# ---------------------------------------------------------------------------


def demo_cleanup(job_a, job_b):
  """Clean up k8s resources and GCS artifacts for both jobs."""
  print("\n" + "=" * 60)
  print("Cleaning up resources...")
  print("=" * 60)

  job_a.cleanup(k8s=True, gcs=True)
  print(f"  Cleaned up Job A ({job_a.job_id})")
  job_b.cleanup(k8s=True, gcs=True)
  print(f"  Cleaned up Job B ({job_b.job_id})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
  job_a, job_b = demo_submit_and_monitor()
  demo_reattach(job_a.job_id)
  demo_list_jobs()
  demo_cleanup(job_a, job_b)
  print("\nDone!")
