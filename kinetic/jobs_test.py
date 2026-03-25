"""Tests for kinetic.jobs — async job handles and observation API."""

from unittest import mock
from unittest.mock import MagicMock

from absl.testing import absltest
from google.api_core import exceptions as google_exceptions

from kinetic.backend.execution import JobContext
from kinetic.jobs import JobHandle, JobStatus, attach, list_jobs


class TestJobHandleSerialization(absltest.TestCase):
  def _make_ctx(self):
    def train():
      return 1

    ctx = JobContext(
      func=train,
      args=(),
      kwargs={},
      env_vars={},
      accelerator="v6e-8",
      container_image=None,
      zone="us-central1-a",
      project="proj",
      cluster_name="cluster",
    )
    ctx.image_uri = "us-docker.pkg.dev/proj/repo/image:tag"
    return ctx

  def test_round_trip_from_job_context(self):
    handle = JobHandle.from_job_context(
      self._make_ctx(),
      backend_name="gke",
      namespace="team-a",
      k8s_name="kinetic-job-1234",
    )

    rebuilt = JobHandle.from_dict(handle.to_dict())

    self.assertEqual(rebuilt.job_id, handle.job_id)
    self.assertEqual(rebuilt.backend, "gke")
    self.assertEqual(rebuilt.namespace, "team-a")
    self.assertEqual(rebuilt.k8s_name, "kinetic-job-1234")
    self.assertEqual(rebuilt.image_uri, handle.image_uri)

  def test_from_dict_ignores_extra_keys(self):
    payload = {
      "job_id": "job-1",
      "backend": "gke",
      "project": "proj",
      "cluster_name": "cluster",
      "zone": "us-central1-a",
      "namespace": "default",
      "bucket_name": "proj-kn-cluster-jobs",
      "k8s_name": "kinetic-job-1",
      "image_uri": "image:tag",
      "accelerator": "cpu",
      "func_name": "train",
      "display_name": "kinetic-train-job-1",
      "created_at": "2026-03-25T10:00:00Z",
      "unknown_future_field": "some_value",
    }
    handle = JobHandle.from_dict(payload)
    self.assertEqual(handle.job_id, "job-1")


class TestAttachAndListJobs(absltest.TestCase):
  def _make_payload(self, job_id, created_at):
    return {
      "job_id": job_id,
      "backend": "gke",
      "project": "proj",
      "cluster_name": "cluster",
      "zone": "us-central1-a",
      "namespace": "default",
      "bucket_name": "proj-kn-cluster-jobs",
      "k8s_name": f"kinetic-{job_id}",
      "image_uri": "image:tag",
      "accelerator": "cpu",
      "func_name": "train",
      "display_name": f"kinetic-train-{job_id}",
      "created_at": created_at,
    }

  def test_attach_downloads_persisted_handle(self):
    payload = self._make_payload("job-a1", "2026-03-25T10:00:00Z")

    with mock.patch(
      "kinetic.jobs.storage.download_handle",
      return_value=payload,
    ) as mock_download:
      handle = attach("job-a1", project="proj", cluster="cluster")

    self.assertEqual(handle.job_id, "job-a1")
    mock_download.assert_called_once_with(
      "proj-kn-cluster-jobs",
      "job-a1",
      project="proj",
    )

  def test_list_jobs_hydrates_handles_and_skips_missing_entries(self):
    payloads = {
      "job-new": self._make_payload("job-new", "2026-03-25T11:00:00Z"),
      "job-old": self._make_payload("job-old", "2026-03-25T10:00:00Z"),
    }

    def download_side_effect(bucket_name, job_id, project=None):
      del bucket_name, project
      if job_id == "job-missing":
        raise google_exceptions.NotFound("missing handle")
      return payloads[job_id]

    with (
      mock.patch("kinetic.jobs.ensure_credentials"),
      mock.patch(
        "kinetic.jobs.gke_client.list_jobs",
        return_value=[{"job_id": "job-old", "k8s_name": "kinetic-job-old"}],
      ),
      mock.patch(
        "kinetic.jobs.pathways_client.list_jobs",
        return_value=[
          {"job_id": "job-new", "k8s_name": "keras-pathways-job-new"},
          {"job_id": "job-missing", "k8s_name": "kinetic-job-missing"},
        ],
      ),
      mock.patch(
        "kinetic.jobs.storage.download_handle",
        side_effect=download_side_effect,
      ),
    ):
      handles = list_jobs(
        project="proj",
        zone="us-central1-a",
        cluster="cluster",
        namespace="default",
      )

    self.assertEqual(
      [handle.job_id for handle in handles], ["job-new", "job-old"]
    )

  def test_list_jobs_tolerates_backend_failures(self):
    payload = self._make_payload("job-1", "2026-03-25T10:00:00Z")

    with (
      mock.patch("kinetic.jobs.ensure_credentials"),
      mock.patch(
        "kinetic.jobs.gke_client.list_jobs",
        side_effect=RuntimeError("gke down"),
      ),
      mock.patch(
        "kinetic.jobs.pathways_client.list_jobs",
        return_value=[{"job_id": "job-1", "k8s_name": "keras-pathways-job-1"}],
      ),
      mock.patch(
        "kinetic.jobs.storage.download_handle",
        return_value=payload,
      ),
    ):
      handles = list_jobs(
        project="proj",
        zone="us-central1-a",
        cluster="cluster",
      )

    self.assertEqual(len(handles), 1)
    self.assertEqual(handles[0].job_id, "job-1")


class TestJobHandleMethods(absltest.TestCase):
  def _make_handle(self, backend="gke"):
    return JobHandle(
      job_id="job-a1b2",
      backend=backend,
      project="proj",
      cluster_name="cluster",
      zone="us-central1-a",
      namespace="default",
      bucket_name="proj-kn-cluster-jobs",
      k8s_name=(
        "kinetic-job-a1b2" if backend == "gke" else "keras-pathways-job-a1b2"
      ),
      image_uri="image:tag",
      accelerator="cpu",
      func_name="train",
      display_name="kinetic-train-job-a1b2",
      created_at="2026-03-25T10:00:00Z",
    )

  def test_credentials_checked_every_call(self):
    handle = self._make_handle()

    with (
      mock.patch("kinetic.jobs.ensure_credentials") as mock_creds,
      mock.patch(
        "kinetic.jobs.gke_client.get_job_status",
        return_value=JobStatus.RUNNING,
      ),
    ):
      handle.status()
      handle.status()

    self.assertEqual(mock_creds.call_count, 2)

  def test_tail_returns_text(self):
    handle = self._make_handle()

    with (
      mock.patch("kinetic.jobs.ensure_credentials"),
      mock.patch(
        "kinetic.jobs.gke_client.get_job_logs",
        return_value="last line",
      ) as mock_logs,
    ):
      text = handle.tail(n=50)

    self.assertEqual(text, "last line")
    mock_logs.assert_called_once_with(
      handle.k8s_name, namespace="default", tail_lines=50
    )

  def test_follow_logs_streams_via_thread_join(self):
    handle = self._make_handle()
    mock_streamer = MagicMock()
    mock_streamer.__enter__.return_value = mock_streamer
    mock_thread = MagicMock()
    mock_streamer._thread = mock_thread

    with (
      mock.patch("kinetic.jobs.ensure_credentials"),
      mock.patch.object(handle, "_get_pod_name", return_value="pod-1"),
      mock.patch("kinetic.jobs.client.CoreV1Api"),
      mock.patch(
        "kinetic.jobs.LogStreamer",
        return_value=mock_streamer,
      ),
    ):
      result = handle.logs(follow=True)

    self.assertIsNone(result)
    mock_streamer.start.assert_called_once_with("pod-1")
    mock_thread.join.assert_called_once()

  def test_result_returns_value_and_cleans_up(self):
    handle = self._make_handle()

    with (
      mock.patch.object(
        handle,
        "status",
        side_effect=[JobStatus.RUNNING, JobStatus.SUCCEEDED],
      ),
      mock.patch.object(
        handle,
        "_download_result_payload_with_backoff",
        return_value={"success": True, "result": 42},
      ),
      mock.patch.object(handle, "cleanup") as mock_cleanup,
      mock.patch("kinetic.jobs.time.sleep"),
    ):
      result = handle.result()

    self.assertEqual(result, 42)
    mock_cleanup.assert_called_once_with(
      k8s=True, gcs=True, cleanup_timeout=180, cleanup_poll_interval=2
    )

  def test_result_checks_gcs_after_not_found(self):
    handle = self._make_handle()

    with (
      mock.patch.object(handle, "status", return_value=JobStatus.NOT_FOUND),
      mock.patch.object(
        handle,
        "_download_result_payload_with_backoff",
        return_value={"success": True, "result": "done"},
      ),
      mock.patch.object(handle, "cleanup") as mock_cleanup,
    ):
      result = handle.result()

    self.assertEqual(result, "done")
    mock_cleanup.assert_called_once_with(
      k8s=True, gcs=True, cleanup_timeout=180, cleanup_poll_interval=2
    )

  def test_result_raises_clear_error_when_result_missing(self):
    handle = self._make_handle()

    with (
      mock.patch.object(handle, "status", return_value=JobStatus.NOT_FOUND),
      mock.patch.object(
        handle,
        "_download_result_payload_with_backoff",
        side_effect=google_exceptions.NotFound("missing"),
      ),
      mock.patch.object(handle, "cleanup") as mock_cleanup,
      self.assertRaisesRegex(RuntimeError, "Job resource was not found"),
    ):
      handle.result()

    mock_cleanup.assert_called_once_with(
      k8s=True, gcs=False, cleanup_timeout=180, cleanup_poll_interval=2
    )

  def test_result_failed_raises_clear_error_when_result_missing(self):
    handle = self._make_handle()

    with (
      mock.patch.object(handle, "status", return_value=JobStatus.FAILED),
      mock.patch.object(
        handle,
        "_download_result_payload_with_backoff",
        side_effect=google_exceptions.NotFound("missing"),
      ),
      mock.patch.object(handle, "cleanup") as mock_cleanup,
      self.assertRaisesRegex(RuntimeError, "Job failed but no result"),
    ):
      handle.result()

    mock_cleanup.assert_called_once_with(
      k8s=True, gcs=False, cleanup_timeout=180, cleanup_poll_interval=2
    )

  def test_result_reraises_user_exception_with_remote_traceback(self):
    handle = self._make_handle()
    user_error = ValueError("boom")

    with (
      mock.patch.object(handle, "status", return_value=JobStatus.FAILED),
      mock.patch.object(
        handle,
        "_download_result_payload_with_backoff",
        return_value={
          "success": False,
          "exception": user_error,
          "traceback": "Traceback: remote boom",
        },
      ),
      self.assertRaisesRegex(ValueError, "boom") as raised,
    ):
      handle.result(cleanup=False)

    self.assertIn(
      "Remote traceback:\nTraceback: remote boom",
      raised.exception.__notes__,
    )

  def test_result_timeout(self):
    handle = self._make_handle()

    with (
      mock.patch.object(
        handle,
        "status",
        return_value=JobStatus.RUNNING,
      ),
      mock.patch("kinetic.jobs.time.sleep"),
      mock.patch(
        "kinetic.jobs.time.monotonic",
        side_effect=[0, 0, 100],
      ),
      self.assertRaisesRegex(TimeoutError, "Timed out"),
    ):
      handle.result(timeout=10)

  def test_result_no_cleanup(self):
    handle = self._make_handle()

    with (
      mock.patch.object(handle, "status", return_value=JobStatus.SUCCEEDED),
      mock.patch.object(
        handle,
        "_download_result_payload_with_backoff",
        return_value={"success": True, "result": 42},
      ),
      mock.patch.object(handle, "cleanup") as mock_cleanup,
    ):
      result = handle.result(cleanup=False)

    self.assertEqual(result, 42)
    mock_cleanup.assert_not_called()

  def test_cancel_deletes_only_k8s_resources(self):
    handle = self._make_handle()

    with mock.patch.object(handle, "cleanup") as mock_cleanup:
      handle.cancel()

    mock_cleanup.assert_called_once_with(
      k8s=True, gcs=False, cleanup_timeout=180, cleanup_poll_interval=2
    )

  def test_cleanup_deletes_k8s_and_gcs(self):
    handle = self._make_handle()

    with (
      mock.patch("kinetic.jobs.ensure_credentials"),
      mock.patch("kinetic.jobs.gke_client.cleanup_job") as mock_cleanup_job,
      mock.patch(
        "kinetic.jobs.storage.cleanup_artifacts",
      ) as mock_cleanup_artifacts,
    ):
      handle.cleanup()

    mock_cleanup_job.assert_called_once_with(
      handle.k8s_name,
      namespace=handle.namespace,
      timeout=180,
      poll_interval=2,
    )
    mock_cleanup_artifacts.assert_called_once_with(
      handle.bucket_name,
      handle.job_id,
      project=handle.project,
    )

  def test_cleanup_k8s_only(self):
    handle = self._make_handle()

    with (
      mock.patch("kinetic.jobs.ensure_credentials"),
      mock.patch("kinetic.jobs.gke_client.cleanup_job") as mock_k8s,
      mock.patch("kinetic.jobs.storage.cleanup_artifacts") as mock_gcs,
    ):
      handle.cleanup(k8s=True, gcs=False)

    mock_k8s.assert_called_once()
    mock_gcs.assert_not_called()

  def test_cleanup_gcs_only(self):
    handle = self._make_handle()

    with (
      mock.patch("kinetic.jobs.ensure_credentials"),
      mock.patch("kinetic.jobs.gke_client.cleanup_job") as mock_k8s,
      mock.patch("kinetic.jobs.storage.cleanup_artifacts") as mock_gcs,
    ):
      handle.cleanup(k8s=False, gcs=True)

    mock_k8s.assert_not_called()
    mock_gcs.assert_called_once()


if __name__ == "__main__":
  absltest.main()
