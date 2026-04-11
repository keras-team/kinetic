"""
Example: Async Collections with Kinetic

This demonstrates the kinetic.map() workflow for running the same function
over many inputs — hyperparameter sweeps, data shards, evaluation runs, etc.

Instead of submitting jobs one by one and managing handles manually,
kinetic.map() fans out across accelerators and gives you a single
BatchHandle to monitor, collect results, and clean up the whole batch.

Prerequisites:
1. A GKE cluster with a CPU node pool (default setup works)
2. kubectl configured to access the cluster
3. KINETIC_PROJECT environment variable set

Workflow overview:
    1. Define a @kinetic.submit function
    2. kinetic.map()       → fan out over inputs, get a BatchHandle
    3. statuses/wait       → monitor batch progress
    4. results()           → collect all results (ordered or streaming)
    5. attach_batch()      → reattach from a different session
    6. cleanup()           → release resources
"""

import os
import time

os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np

import kinetic


@kinetic.submit(accelerator="cpu")
def train(lr, epochs):
  """Train a small model and return the final loss."""
  model = keras.Sequential(
    [
      keras.layers.Dense(64, activation="relu", input_shape=(10,)),
      keras.layers.Dense(1),
    ]
  )
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")

  x = np.random.randn(500, 10)
  y = np.random.randn(500, 1)

  print(f"Training with lr={lr}, epochs={epochs}")
  history = model.fit(x, y, epochs=epochs, batch_size=32, verbose=0)
  final_loss = history.history["loss"][-1]
  print(f"Done — loss: {final_loss:.4f}")
  return {"lr": lr, "epochs": epochs, "loss": final_loss}


def demo_basic_sweep():
  """Launch a hyperparameter sweep and collect ordered results."""
  print("=" * 60)
  print("Hyperparameter sweep")
  print("=" * 60)

  configs = [
    {"lr": 0.001, "epochs": 5},
    {"lr": 0.005, "epochs": 10},
    {"lr": 0.01, "epochs": 10},
    {"lr": 0.05, "epochs": 5},
  ]

  # kinetic.map() dispatches each dict as **kwargs to train().
  # max_concurrent=2 limits how many jobs run at once.
  batch = kinetic.map(train, configs, max_concurrent=2)

  print(f"\nBatch ID: {batch.group_id}")
  print(f"Submitted {len(configs)} jobs (max 2 concurrent)\n")

  # Ordered results — losses[i] corresponds to configs[i].
  results = batch.results(timeout=600, cleanup=False)
  for r in results:
    print(f"  lr={r['lr']:<6}  epochs={r['epochs']:<3}  loss={r['loss']:.4f}")

  best = min(results, key=lambda r: r["loss"])
  print(f"\nBest config: lr={best['lr']}, loss={best['loss']:.4f}")

  return batch


def demo_monitoring():
  """Show status polling and streaming results via as_completed()."""
  print("\n" + "=" * 60)
  print("Monitoring and streaming")
  print("=" * 60)

  configs = [
    {"lr": 0.01, "epochs": 3},
    {"lr": 0.02, "epochs": 5},
    {"lr": 0.03, "epochs": 8},
  ]

  batch = kinetic.map(train, configs)
  print(f"\nBatch ID: {batch.group_id}")

  # Poll status a few times.
  for _ in range(6):
    counts = batch.status_counts()
    print(f"  Status: {counts}")
    if counts.get("SUCCEEDED", 0) == len(configs):
      break
    time.sleep(10)

  # Stream results as jobs finish (completion order, not input order).
  print("\nResults (completion order):")
  for job in batch.as_completed(timeout=600):
    result = job.result(cleanup=False)
    print(f"  {job.job_id}: lr={result['lr']}, loss={result['loss']:.4f}")

  batch.cleanup()


@kinetic.submit(accelerator="cpu")
def fragile_train(lr):
  """Training that fails on extreme learning rates."""
  if lr > 1.0:
    raise ValueError(f"Learning rate {lr} is too high!")
  return {"lr": lr, "loss": 1.0 - lr}


def demo_error_handling():
  """Demonstrate return_exceptions for fault-tolerant collection."""
  print("\n" + "=" * 60)
  print("Error handling")
  print("=" * 60)

  # The third config will fail.
  batch = kinetic.map(fragile_train, [0.01, 0.1, 5.0, 0.5])

  results = batch.results(timeout=600, return_exceptions=True, cleanup=False)

  for i, r in enumerate(results):
    if isinstance(r, Exception):
      print(f"  Job {i}: FAILED — {r}")
    else:
      print(f"  Job {i}: loss={r['loss']:.4f}")

  failed = batch.failures()
  print(f"\n{len(failed)} job(s) failed")

  batch.cleanup()


def demo_reattach(group_id):
  """Simulate recovering a batch from a different session."""
  print("\n" + "=" * 60)
  print(f"Reattaching to batch {group_id}")
  print("=" * 60)

  batch = kinetic.attach_batch(group_id)
  print(f"  Name:  {batch.name}")
  print(f"  Jobs:  {len(batch.jobs)}")
  print(f"  Counts: {batch.status_counts()}")

  # Collect results from the reattached handle.
  results = batch.results(timeout=600, cleanup=False)
  for r in results:
    print(f"  lr={r['lr']:<6}  loss={r['loss']:.4f}")

  return batch


def demo_cleanup(batch):
  """Full teardown — children and manifest."""
  print("\n" + "=" * 60)
  print("Cleaning up batch resources")
  print("=" * 60)

  batch.cleanup(k8s=True, gcs=True)
  print(f"  Cleaned up batch {batch.group_id}")


if __name__ == "__main__":
  # Run the sweep and keep resources for reattachment demo.
  batch = demo_basic_sweep()

  # # Reattach using just the group_id.
  reattached = demo_reattach(batch.group_id)

  # # Cleanup the reattached handle.
  demo_cleanup(reattached)

  # # Monitoring with as_completed().
  demo_monitoring()

  # Fault-tolerant error handling.
  demo_error_handling()

  print("\nDone!")
