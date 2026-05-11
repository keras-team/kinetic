"""Tests for kinetic.cli.commands.jobs — CLI smoke tests."""

from unittest import mock
from unittest.mock import MagicMock

from absl.testing import absltest
from click.testing import CliRunner

from kinetic.cli.commands.jobs import jobs
from kinetic.jobs import JobHandle, JobStatus

_JOBS_MODULE = "kinetic.cli.commands.jobs"


def _make_handle(**overrides):
  defaults = {
    "job_id": "job-abc",
    "backend": "gke",
    "project": "proj",
    "cluster_name": "cluster",
    "zone": "us-central1-a",
    "namespace": "default",
    "bucket_name": "proj-kn-cluster-jobs",
    "k8s_name": "kinetic-job-abc",
    "image_uri": "img",
    "accelerator": "l4",
    "func_name": "train",
    "display_name": "kinetic-train-job-abc",
    "created_at": "2026-03-07T12:00:00Z",
  }
  defaults.update(overrides)
  return JobHandle(**defaults)


class TestJobsList(absltest.TestCase):
  @mock.patch(f"{_JOBS_MODULE}.list_jobs", return_value=[])
  def test_list_no_jobs(self, mock_lj):
    runner = CliRunner()
    result = runner.invoke(
      jobs,
      ["list", "--project", "proj", "--cluster", "cluster"],
      catch_exceptions=False,
    )
    self.assertEqual(result.exit_code, 0)
    self.assertIn("No live jobs found", result.output)

  @mock.patch(f"{_JOBS_MODULE}.list_jobs")
  def test_list_with_jobs(self, mock_lj):
    mock_lj.return_value = [
      _make_handle(job_id="job-1"),
      _make_handle(job_id="job-2", backend="pathways"),
    ]
    runner = CliRunner()
    result = runner.invoke(
      jobs,
      ["list", "--project", "proj", "--cluster", "cluster"],
      catch_exceptions=False,
    )
    self.assertEqual(result.exit_code, 0)
    self.assertIn("job-1", result.output)
    self.assertIn("job-2", result.output)
    # Should show accelerator, not call status()
    self.assertIn("l4", result.output)


class TestJobsStatus(absltest.TestCase):
  @mock.patch(f"{_JOBS_MODULE}._attach")
  def test_status(self, mock_attach):
    handle = _make_handle()
    handle.status = MagicMock(return_value=JobStatus.RUNNING)
    mock_attach.return_value = handle

    runner = CliRunner()
    result = runner.invoke(
      jobs,
      ["status", "job-abc", "--project", "proj", "--cluster", "cluster"],
      catch_exceptions=False,
    )
    self.assertEqual(result.exit_code, 0)
    self.assertIn("RUNNING", result.output)


class TestJobsCancel(absltest.TestCase):
  @mock.patch(f"{_JOBS_MODULE}._attach")
  def test_cancel(self, mock_attach):
    handle = _make_handle()
    handle.cancel = MagicMock()
    mock_attach.return_value = handle

    runner = CliRunner()
    result = runner.invoke(
      jobs,
      ["cancel", "job-abc", "--project", "proj", "--cluster", "cluster"],
      catch_exceptions=False,
    )
    self.assertEqual(result.exit_code, 0)
    self.assertIn("Cancelled", result.output)
    handle.cancel.assert_called_once()


class TestJobsResult(absltest.TestCase):
  @mock.patch(f"{_JOBS_MODULE}._attach")
  def test_result(self, mock_attach):
    handle = _make_handle()
    handle.result = MagicMock(return_value={"accuracy": 0.95})
    mock_attach.return_value = handle

    runner = CliRunner()
    result = runner.invoke(
      jobs,
      ["result", "job-abc", "--project", "proj", "--cluster", "cluster"],
      catch_exceptions=False,
    )
    self.assertEqual(result.exit_code, 0)
    self.assertIn("accuracy", result.output)
    handle.result.assert_called_once_with(
      timeout=None,
      cleanup=True,
      cleanup_timeout=180.0,
      cleanup_poll_interval=2.0,
    )

  @mock.patch(f"{_JOBS_MODULE}._attach")
  def test_result_no_cleanup(self, mock_attach):
    handle = _make_handle()
    handle.result = MagicMock(return_value=42)
    mock_attach.return_value = handle

    runner = CliRunner()
    result = runner.invoke(
      jobs,
      [
        "result",
        "job-abc",
        "--no-cleanup",
        "--project",
        "proj",
        "--cluster",
        "cluster",
      ],
      catch_exceptions=False,
    )
    self.assertEqual(result.exit_code, 0)
    handle.result.assert_called_once_with(
      timeout=None,
      cleanup=False,
      cleanup_timeout=180.0,
      cleanup_poll_interval=2.0,
    )

  @mock.patch(f"{_JOBS_MODULE}._attach")
  def test_result_with_timeout(self, mock_attach):
    handle = _make_handle()
    handle.result = MagicMock(return_value=42)
    mock_attach.return_value = handle

    runner = CliRunner()
    result = runner.invoke(
      jobs,
      [
        "result",
        "job-abc",
        "--timeout",
        "300",
        "--project",
        "proj",
        "--cluster",
        "cluster",
      ],
      catch_exceptions=False,
    )
    self.assertEqual(result.exit_code, 0)
    handle.result.assert_called_once_with(
      timeout=300.0,
      cleanup=True,
      cleanup_timeout=180.0,
      cleanup_poll_interval=2.0,
    )


class TestJobsCleanup(absltest.TestCase):
  @mock.patch(f"{_JOBS_MODULE}._attach")
  def test_cleanup_both(self, mock_attach):
    handle = _make_handle()
    handle.cleanup = MagicMock()
    mock_attach.return_value = handle

    runner = CliRunner()
    result = runner.invoke(
      jobs,
      ["cleanup", "job-abc", "--project", "proj", "--cluster", "cluster"],
      catch_exceptions=False,
    )
    self.assertEqual(result.exit_code, 0)
    handle.cleanup.assert_called_once_with(
      k8s=True,
      gcs=True,
      cleanup_timeout=180.0,
      cleanup_poll_interval=2.0,
    )

  @mock.patch(f"{_JOBS_MODULE}._attach")
  def test_cleanup_gcs_only(self, mock_attach):
    handle = _make_handle()
    handle.cleanup = MagicMock()
    mock_attach.return_value = handle

    runner = CliRunner()
    result = runner.invoke(
      jobs,
      [
        "cleanup",
        "job-abc",
        "--no-k8s",
        "--project",
        "proj",
        "--cluster",
        "cluster",
      ],
      catch_exceptions=False,
    )
    self.assertEqual(result.exit_code, 0)
    handle.cleanup.assert_called_once_with(
      k8s=False,
      gcs=True,
      cleanup_timeout=180.0,
      cleanup_poll_interval=2.0,
    )

  @mock.patch(f"{_JOBS_MODULE}._attach")
  def test_cleanup_k8s_only(self, mock_attach):
    handle = _make_handle()
    handle.cleanup = MagicMock()
    mock_attach.return_value = handle

    runner = CliRunner()
    result = runner.invoke(
      jobs,
      [
        "cleanup",
        "job-abc",
        "--no-gcs",
        "--project",
        "proj",
        "--cluster",
        "cluster",
      ],
      catch_exceptions=False,
    )
    self.assertEqual(result.exit_code, 0)
    handle.cleanup.assert_called_once_with(
      k8s=True,
      gcs=False,
      cleanup_timeout=180.0,
      cleanup_poll_interval=2.0,
    )


class TestJobsLogs(absltest.TestCase):
  @mock.patch(f"{_JOBS_MODULE}._attach")
  def test_logs_default(self, mock_attach):
    handle = _make_handle()
    handle.logs = MagicMock(return_value="log output")
    mock_attach.return_value = handle

    runner = CliRunner()
    result = runner.invoke(
      jobs,
      ["logs", "job-abc", "--project", "proj", "--cluster", "cluster"],
      catch_exceptions=False,
    )
    self.assertEqual(result.exit_code, 0)
    self.assertIn("log output", result.output)

  @mock.patch(f"{_JOBS_MODULE}._attach")
  def test_logs_follow(self, mock_attach):
    handle = _make_handle()
    handle.logs = MagicMock(return_value=None)
    mock_attach.return_value = handle

    runner = CliRunner()
    result = runner.invoke(
      jobs,
      [
        "logs",
        "job-abc",
        "--follow",
        "--project",
        "proj",
        "--cluster",
        "cluster",
      ],
      catch_exceptions=False,
    )
    self.assertEqual(result.exit_code, 0)
    handle.logs.assert_called_once_with(follow=True)

  @mock.patch(f"{_JOBS_MODULE}._attach")
  def test_logs_tail(self, mock_attach):
    handle = _make_handle()
    handle.tail = MagicMock(return_value="last 50 lines")
    mock_attach.return_value = handle

    runner = CliRunner()
    result = runner.invoke(
      jobs,
      [
        "logs",
        "job-abc",
        "--tail",
        "50",
        "--project",
        "proj",
        "--cluster",
        "cluster",
      ],
      catch_exceptions=False,
    )
    self.assertEqual(result.exit_code, 0)
    self.assertIn("last 50 lines", result.output)
    handle.tail.assert_called_once_with(n=50)

  @mock.patch(f"{_JOBS_MODULE}._attach")
  def test_follow_and_tail_rejects(self, mock_attach):
    handle = _make_handle()
    mock_attach.return_value = handle

    runner = CliRunner()
    result = runner.invoke(
      jobs,
      [
        "logs",
        "job-abc",
        "--follow",
        "--tail",
        "50",
        "--project",
        "proj",
        "--cluster",
        "cluster",
      ],
    )
    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("--follow or --tail", result.output)


class TestMissingProject(absltest.TestCase):
  def test_status_requires_project(self):
    runner = CliRunner()
    with mock.patch.dict("os.environ", {}, clear=True):
      result = runner.invoke(jobs, ["status", "job-abc"])
    self.assertNotEqual(result.exit_code, 0)

  def test_list_requires_project(self):
    runner = CliRunner()
    with mock.patch.dict("os.environ", {}, clear=True):
      result = runner.invoke(jobs, ["list"])
    self.assertNotEqual(result.exit_code, 0)


if __name__ == "__main__":
  absltest.main()
