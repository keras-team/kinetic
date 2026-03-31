"""E2E tests for async collections — map, BatchHandle, attach_batch.

These tests require a real GCP project with:
- A GKE cluster with a CPU node pool
- Cloud Storage, Cloud Build, and Artifact Registry APIs enabled
- Proper IAM permissions

Set E2E_TESTS=1 to enable.
"""

import time

from absl.testing import absltest

import kinetic
from kinetic.jobs import JobStatus
from tests.e2e.e2e_utils import skip_unless_e2e

# Timeout for individual jobs (seconds).  CPU jobs typically finish in
# under 2 minutes once the container is cached; 600 s gives headroom
# for cold-start builds.
_JOB_TIMEOUT = 600


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


@kinetic.submit(accelerator="cpu")
def _double(x):
  return x * 2


@kinetic.submit(accelerator="cpu")
def _add(a, b):
  return a + b


@kinetic.submit(accelerator="cpu")
def _train(lr, epochs):
  """Simulate training — returns a dict summary."""
  loss = 1.0
  for _ in range(epochs):
    loss *= 1 - lr
  return {"lr": lr, "epochs": epochs, "loss": round(loss, 6)}


@kinetic.submit(accelerator="cpu")
def _fail_on_negative(x):
  if x < 0:
    raise ValueError(f"negative input: {x}")
  return x


@kinetic.submit(accelerator="cpu")
def _slow(seconds):
  import time as _time

  _time.sleep(seconds)
  return seconds


# ------------------------------------------------------------------
# Basic map + results
# ------------------------------------------------------------------


@skip_unless_e2e()
class TestMapBasic(absltest.TestCase):
  """Submit a batch via map() and collect ordered results."""

  def test_map_scalar_inputs(self):
    """map() over scalars with default auto input mode."""
    batch = kinetic.map(_double, [1, 2, 3])
    results = batch.results(timeout=_JOB_TIMEOUT)

    self.assertEqual(results, [2, 4, 6])
    self.assertTrue(batch.group_id.startswith("grp-"))

  def test_map_dict_inputs_auto_kwargs(self):
    """Dicts with valid identifier keys are unpacked as **kwargs."""
    configs = [
      {"lr": 0.1, "epochs": 5},
      {"lr": 0.01, "epochs": 10},
    ]
    batch = kinetic.map(_train, configs)
    results = batch.results(timeout=_JOB_TIMEOUT)

    self.assertEqual(len(results), 2)
    self.assertAlmostEqual(results[0]["lr"], 0.1)
    self.assertAlmostEqual(results[1]["lr"], 0.01)
    # Verify the training logic ran.
    self.assertLess(results[0]["loss"], 1.0)
    self.assertLess(results[1]["loss"], 1.0)

  def test_map_tuple_inputs_auto_args(self):
    """Tuples are unpacked as *args in auto mode."""
    batch = kinetic.map(_add, [(1, 2), (10, 20)])
    results = batch.results(timeout=_JOB_TIMEOUT)

    self.assertEqual(results, [3, 30])

  def test_map_name_and_tags(self):
    """name and tags are stored on the handle."""
    batch = kinetic.map(
      _double,
      [1],
      name="e2e-test-batch",
      tags={"env": "test"},
    )
    batch.results(timeout=_JOB_TIMEOUT)

    self.assertEqual(batch.name, "e2e-test-batch")
    self.assertEqual(batch.tags, {"env": "test"})


# ------------------------------------------------------------------
# Monitoring
# ------------------------------------------------------------------


@skip_unless_e2e()
class TestMapMonitoring(absltest.TestCase):
  """Batch observation during and after execution."""

  def test_status_counts_and_wait(self):
    """status_counts() reflects progress; wait() blocks to completion."""
    batch = kinetic.map(_double, [1, 2, 3])

    # Wait for all jobs to finish.
    batch.wait(timeout=_JOB_TIMEOUT)

    counts = batch.status_counts()
    self.assertEqual(counts.get("SUCCEEDED", 0), 3)

    # Every job should have a handle.
    self.assertTrue(all(j is not None for j in batch.jobs))

  def test_as_completed_yields_all_jobs(self):
    """as_completed() should yield one handle per input."""
    batch = kinetic.map(_double, [10, 20, 30])

    seen = []
    for job in batch.as_completed(timeout=_JOB_TIMEOUT):
      result = job.result(cleanup=False)
      seen.append(result)

    self.assertEqual(sorted(seen), [20, 40, 60])
    batch.cleanup()


# ------------------------------------------------------------------
# Unordered results
# ------------------------------------------------------------------


@skip_unless_e2e()
class TestMapUnordered(absltest.TestCase):
  """Completion-order result collection."""

  def test_results_unordered(self):
    """ordered=False returns results as jobs finish."""
    batch = kinetic.map(_double, [5, 10])
    results = batch.results(timeout=_JOB_TIMEOUT, ordered=False)

    self.assertEqual(sorted(results), [10, 20])


# ------------------------------------------------------------------
# Concurrency control
# ------------------------------------------------------------------


@skip_unless_e2e()
class TestMapConcurrency(absltest.TestCase):
  """Bounded concurrency via max_concurrent."""

  def test_max_concurrent(self):
    """All jobs should complete even with max_concurrent=1."""
    batch = kinetic.map(_double, [1, 2, 3], max_concurrent=1)
    results = batch.results(timeout=_JOB_TIMEOUT)

    self.assertEqual(results, [2, 4, 6])


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------


@skip_unless_e2e()
class TestMapErrorHandling(absltest.TestCase):
  """Failure propagation through BatchError and return_exceptions."""

  def test_batch_error_raised_on_failure(self):
    """A failing job should cause results() to raise BatchError."""
    batch = kinetic.map(_fail_on_negative, [1, -1, 2])

    with self.assertRaises(kinetic.BatchError) as ctx:
      batch.results(timeout=_JOB_TIMEOUT)

    err = ctx.exception
    self.assertEqual(len(err.failures), 1)
    # Successful results should still be present.
    self.assertEqual(err.partial_results[0], 1)
    self.assertEqual(err.partial_results[2], 2)

  def test_return_exceptions(self):
    """return_exceptions=True should place exceptions in the list."""
    batch = kinetic.map(_fail_on_negative, [1, -1, 2])
    results = batch.results(timeout=_JOB_TIMEOUT, return_exceptions=True)

    self.assertEqual(results[0], 1)
    self.assertIsInstance(results[1], ValueError)
    self.assertIn("negative input", str(results[1]))
    self.assertEqual(results[2], 2)

  def test_failures_method(self):
    """failures() should return handles for failed jobs only."""
    batch = kinetic.map(_fail_on_negative, [1, -1])
    # Collect with return_exceptions to avoid raising.
    batch.results(timeout=_JOB_TIMEOUT, return_exceptions=True, cleanup=False)

    failed = batch.failures()
    self.assertEqual(len(failed), 1)
    self.assertEqual(failed[0].status(), JobStatus.FAILED)

    # Clean up manually since we used cleanup=False.
    batch.cleanup()


# ------------------------------------------------------------------
# Reattachment
# ------------------------------------------------------------------


@skip_unless_e2e()
class TestAttachBatch(absltest.TestCase):
  """Cross-session recovery via attach_batch()."""

  def test_attach_batch_and_collect(self):
    """attach_batch() should reconstruct a usable handle."""
    batch = kinetic.map(_double, [7, 8])
    batch.results(timeout=_JOB_TIMEOUT, cleanup=False)

    # Simulate reattachment from a different session.
    group_id = batch.group_id
    reattached = kinetic.attach_batch(group_id)

    self.assertEqual(reattached.group_id, group_id)
    self.assertEqual(len(reattached.jobs), 2)
    self.assertTrue(all(j is not None for j in reattached.jobs))

    # Results should match.
    results = reattached.results(cleanup=False)
    self.assertEqual(results, [14, 16])

    reattached.cleanup()


# ------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------


@skip_unless_e2e()
class TestMapCleanup(absltest.TestCase):
  """Resource cleanup for batch collections."""

  def test_results_cleanup_removes_child_resources(self):
    """results(cleanup=True) should clean up each child."""
    batch = kinetic.map(_double, [1])
    batch.results(timeout=_JOB_TIMEOUT, cleanup=True)

    # The child's K8s resource should be gone.
    self.assertEqual(batch.jobs[0].status(), JobStatus.NOT_FOUND)

  def test_batch_cleanup_removes_everything(self):
    """BatchHandle.cleanup() should remove children and manifest."""
    batch = kinetic.map(_double, [1])
    batch.results(timeout=_JOB_TIMEOUT, cleanup=False)

    batch.cleanup(k8s=True, gcs=True)
    self.assertEqual(batch.jobs[0].status(), JobStatus.NOT_FOUND)


# ------------------------------------------------------------------
# Cancel
# ------------------------------------------------------------------


@skip_unless_e2e()
class TestMapCancel(absltest.TestCase):
  """Batch-level cancellation."""

  def test_cancel_stops_running_jobs(self):
    """cancel() should delete K8s resources for non-terminal jobs."""
    batch = kinetic.map(_slow, [120, 120])

    # Give jobs a moment to be submitted and start.
    time.sleep(15)

    batch.cancel()

    # Both jobs should now be NOT_FOUND.
    for job in batch.jobs:
      if job is not None:
        self.assertEqual(job.status(), JobStatus.NOT_FOUND)

    batch.cleanup(k8s=False, gcs=True)


if __name__ == "__main__":
  absltest.main()
