"""Tests for keras_remote.backend.execution — JobContext and execute_remote."""

import os
import pathlib
import tempfile
from unittest import mock
from unittest.mock import MagicMock

from absl.testing import absltest
from google.api_core import exceptions as google_exceptions

from keras_remote.backend.execution import (
  JobContext,
  _find_requirements,
  execute_remote,
)


def _make_temp_path(test_case):
  """Create a temp directory that is cleaned up after the test."""
  td = tempfile.TemporaryDirectory()
  test_case.addCleanup(td.cleanup)
  return pathlib.Path(td.name)


class TestJobContext(absltest.TestCase):
  def _make_func(self):
    def my_train():
      return 42

    return my_train

  def test_post_init_derived_fields(self):
    ctx = JobContext(
      func=self._make_func(),
      args=(),
      kwargs={},
      env_vars={},
      accelerator="cpu",
      container_image=None,
      zone="europe-west4-b",
      project="my-proj",
      cluster_name="my-cluster",
    )
    self.assertEqual(ctx.bucket_name, "my-proj-kr-my-cluster-jobs")
    self.assertEqual(ctx.region, "europe-west4")
    self.assertTrue(ctx.display_name.startswith("keras-remote-my_train-"))
    self.assertRegex(ctx.job_id, r"^job-[0-9a-f]{8}$")

  def test_from_params_explicit(self):
    ctx = JobContext.from_params(
      func=self._make_func(),
      args=(1, 2),
      kwargs={"k": "v"},
      accelerator="l4",
      container_image="my-image:latest",
      zone="us-west1-a",
      project="explicit-proj",
      env_vars={"X": "Y"},
    )
    self.assertEqual(ctx.zone, "us-west1-a")
    self.assertEqual(ctx.project, "explicit-proj")
    self.assertEqual(ctx.accelerator, "l4")
    self.assertEqual(ctx.container_image, "my-image:latest")
    self.assertEqual(ctx.args, (1, 2))
    self.assertEqual(ctx.kwargs, {"k": "v"})
    self.assertEqual(ctx.env_vars, {"X": "Y"})

  def test_from_params_resolves_zone_from_env(self):
    with mock.patch.dict(
      os.environ,
      {"KERAS_REMOTE_ZONE": "asia-east1-c", "KERAS_REMOTE_PROJECT": "env-proj"},
    ):
      ctx = JobContext.from_params(
        func=self._make_func(),
        args=(),
        kwargs={},
        accelerator="cpu",
        container_image=None,
        zone=None,
        project=None,
        env_vars={},
      )
    self.assertEqual(ctx.zone, "asia-east1-c")
    self.assertEqual(ctx.project, "env-proj")

  def test_from_params_falls_back_to_google_cloud_project(self):
    env = {
      k: v
      for k, v in os.environ.items()
      if k not in ("KERAS_REMOTE_PROJECT", "GOOGLE_CLOUD_PROJECT")
    }
    env["GOOGLE_CLOUD_PROJECT"] = "gc-proj"
    with mock.patch.dict(os.environ, env, clear=True):
      ctx = JobContext.from_params(
        func=self._make_func(),
        args=(),
        kwargs={},
        accelerator="cpu",
        container_image=None,
        zone="us-central1-a",
        project=None,
        env_vars={},
      )
    self.assertEqual(ctx.project, "gc-proj")

  def test_from_params_no_project_raises(self):
    env = {
      k: v
      for k, v in os.environ.items()
      if k not in ("KERAS_REMOTE_PROJECT", "GOOGLE_CLOUD_PROJECT")
    }
    with (
      mock.patch.dict(os.environ, env, clear=True),
      self.assertRaisesRegex(ValueError, "project must be specified"),
    ):
      JobContext.from_params(
        func=self._make_func(),
        args=(),
        kwargs={},
        accelerator="cpu",
        container_image=None,
        zone="us-central1-a",
        project=None,
        env_vars={},
      )


class TestFindRequirements(absltest.TestCase):
  def test_finds_in_start_dir(self):
    """Returns the path when requirements.txt exists in the start directory."""
    tmp_path = _make_temp_path(self)
    (tmp_path / "requirements.txt").write_text("numpy\n")
    self.assertEqual(
      _find_requirements(str(tmp_path)),
      str(tmp_path / "requirements.txt"),
    )

  def test_finds_in_parent_dir(self):
    """Walks up the directory tree to find requirements.txt in a parent."""
    tmp_path = _make_temp_path(self)
    (tmp_path / "requirements.txt").write_text("numpy\n")
    child = tmp_path / "subdir"
    child.mkdir()
    self.assertEqual(
      _find_requirements(str(child)),
      str(tmp_path / "requirements.txt"),
    )

  def test_returns_none_when_not_found(self):
    """Returns None when no requirements.txt exists in any ancestor."""
    tmp_path = _make_temp_path(self)
    empty = tmp_path / "empty"
    empty.mkdir()
    self.assertIsNone(_find_requirements(str(empty)))


class TestExecuteRemote(absltest.TestCase):
  def _make_func(self):
    def my_train():
      return 42

    return my_train

  def _make_ctx(self, container_image=None):
    return JobContext(
      func=self._make_func(),
      args=(),
      kwargs={},
      env_vars={},
      accelerator="cpu",
      container_image=container_image,
      zone="us-central1-a",
      project="proj",
      cluster_name="keras-remote-cluster",
    )

  def test_success_flow(self):
    with (
      mock.patch("keras_remote.backend.execution.ensure_credentials"),
      mock.patch("keras_remote.backend.execution._build_container"),
      mock.patch("keras_remote.backend.execution._upload_artifacts"),
      mock.patch(
        "keras_remote.backend.execution._download_result",
        return_value={"success": True, "result": 42},
      ),
      mock.patch(
        "keras_remote.backend.execution._cleanup_and_return",
        return_value=42,
      ),
    ):
      ctx = self._make_ctx()
      backend = MagicMock()

      result = execute_remote(ctx, backend)

      backend.submit_job.assert_called_once_with(ctx)
      backend.wait_for_job.assert_called_once()
      backend.cleanup_job.assert_called_once()
      self.assertEqual(result, 42)

  def test_cleanup_on_wait_failure(self):
    with (
      mock.patch("keras_remote.backend.execution.ensure_credentials"),
      mock.patch("keras_remote.backend.execution._build_container"),
      mock.patch("keras_remote.backend.execution._upload_artifacts"),
      mock.patch(
        "keras_remote.backend.execution._download_result",
        side_effect=google_exceptions.NotFound("no result uploaded"),
      ),
    ):
      ctx = self._make_ctx()
      backend = MagicMock()
      backend.wait_for_job.side_effect = RuntimeError("job failed")

      with self.assertRaisesRegex(RuntimeError, "job failed"):
        execute_remote(ctx, backend)

      # cleanup_job is called in finally block even when wait fails
      backend.cleanup_job.assert_called_once()


if __name__ == "__main__":
  absltest.main()
