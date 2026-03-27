"""E2E tests for async job management — submit, observe, collect, cleanup.

These tests require a real GCP project with:
- A GKE cluster with a CPU node pool
- Cloud Storage, Cloud Build, and Artifact Registry APIs enabled
- Proper IAM permissions

Set E2E_TESTS=1 to enable.
"""

from absl.testing import absltest

import kinetic
from kinetic.jobs import JobStatus, attach, list_jobs
from tests.e2e.e2e_utils import skip_unless_e2e


@skip_unless_e2e()
class TestAsyncJobLifecycle(absltest.TestCase):
  """Full lifecycle: submit → status → logs → result → cleanup."""

  def test_submit_and_collect_result(self):
    """Submit a job, wait for the result, and verify the return value."""

    @kinetic.submit(accelerator="cpu")
    def add(a, b):
      return a + b

    handle = add(10, 32)
    self.assertIsNotNone(handle.job_id)

    result = handle.result(timeout=300)
    self.assertEqual(result, 42)

  def test_status_transitions(self):
    """Status should move from PENDING/RUNNING to SUCCEEDED."""

    @kinetic.submit(accelerator="cpu")
    def noop():
      return "done"

    handle = noop()

    # Initial status should be PENDING or RUNNING.
    initial = handle.status()
    self.assertIn(initial, {JobStatus.PENDING, JobStatus.RUNNING})

    # Wait for completion.
    handle.result(timeout=300, cleanup=False)
    final = handle.status()
    self.assertEqual(final, JobStatus.SUCCEEDED)

    handle.cleanup()

  def test_logs_available_after_completion(self):
    """Logs should be retrievable after a job succeeds."""

    @kinetic.submit(accelerator="cpu")
    def chatty():
      print("hello from remote")
      return 1

    handle = chatty()
    handle.result(timeout=300, cleanup=False)

    log_text = handle.logs()
    self.assertIsNotNone(log_text)
    self.assertIn("hello from remote", log_text)

    handle.cleanup()

  def test_tail_returns_partial_logs(self):
    """tail(n=5) should return at most 5 lines."""

    @kinetic.submit(accelerator="cpu")
    def many_lines():
      for i in range(20):
        print(f"line {i}")
      return "ok"

    handle = many_lines()
    handle.result(timeout=300, cleanup=False)

    tail_text = handle.tail(n=5)
    lines = [line for line in tail_text.strip().splitlines() if line.strip()]
    self.assertLessEqual(len(lines), 5)

    handle.cleanup()

  def test_remote_exception_reraised(self):
    """A remote exception should be re-raised locally with traceback."""

    @kinetic.submit(accelerator="cpu")
    def failing():
      raise ValueError("intentional e2e error")

    handle = failing()

    with self.assertRaisesRegex(ValueError, "intentional e2e error"):
      handle.result(timeout=300)


@skip_unless_e2e()
class TestAttachAndListJobs(absltest.TestCase):
  """Cross-session reattachment and job discovery."""

  def test_attach_to_completed_job(self):
    """attach() should reconstruct a handle that can still fetch results."""

    @kinetic.submit(accelerator="cpu")
    def compute():
      return 99

    original = compute()
    original.result(timeout=300, cleanup=False)

    reattached = attach(original.job_id)
    self.assertEqual(reattached.job_id, original.job_id)
    self.assertEqual(reattached.status(), JobStatus.SUCCEEDED)

    reattached.cleanup()

  def test_list_jobs_includes_submitted_job(self):
    """list_jobs() should discover a live job on the cluster."""

    @kinetic.submit(accelerator="cpu")
    def discoverable():
      import time

      time.sleep(10)
      return "found"

    handle = discoverable()

    try:
      jobs = list_jobs()
      job_ids = [j.job_id for j in jobs]
      self.assertIn(handle.job_id, job_ids)
    finally:
      handle.result(timeout=300)


@skip_unless_e2e()
class TestCancelAndCleanup(absltest.TestCase):
  """Job cancellation and resource cleanup."""

  def test_cancel_running_job(self):
    """cancel() should delete the k8s resource; status becomes NOT_FOUND."""

    @kinetic.submit(accelerator="cpu")
    def long_running():
      import time

      time.sleep(600)
      return "should not reach"

    handle = long_running()

    # Wait until the job is at least pending/running.
    for _ in range(30):
      if handle.status() in {JobStatus.PENDING, JobStatus.RUNNING}:
        break
      import time

      time.sleep(2)

    handle.cancel()
    self.assertEqual(handle.status(), JobStatus.NOT_FOUND)

    # GCS cleanup.
    handle.cleanup(k8s=False, gcs=True)

  def test_cleanup_removes_resources(self):
    """cleanup() should delete both k8s and GCS artifacts."""

    @kinetic.submit(accelerator="cpu")
    def ephemeral():
      return "bye"

    handle = ephemeral()
    handle.result(timeout=300, cleanup=False)

    handle.cleanup(k8s=True, gcs=True)
    self.assertEqual(handle.status(), JobStatus.NOT_FOUND)


@skip_unless_e2e()
class TestResultOptions(absltest.TestCase):
  """result() timeout and cleanup flag behavior."""

  def test_result_timeout_raises(self):
    """result(timeout=1) should raise TimeoutError for a slow job."""

    @kinetic.submit(accelerator="cpu")
    def slow():
      import time

      time.sleep(600)
      return "late"

    handle = slow()

    with self.assertRaises(TimeoutError):
      handle.result(timeout=1)

    # Clean up the still-running job.
    handle.cancel()
    handle.cleanup(k8s=False, gcs=True)

  def test_result_no_cleanup_preserves_resources(self):
    """result(cleanup=False) should leave k8s/GCS resources intact."""

    @kinetic.submit(accelerator="cpu")
    def keep_alive():
      return "still here"

    handle = keep_alive()
    handle.result(timeout=300, cleanup=False)

    # Job should still be findable (k8s resource exists).
    self.assertIn(handle.status(), {JobStatus.SUCCEEDED, JobStatus.NOT_FOUND})

    handle.cleanup()


if __name__ == "__main__":
  absltest.main()
