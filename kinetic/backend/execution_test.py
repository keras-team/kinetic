"""Tests for kinetic.backend.execution — JobContext and submit_remote."""

import os
import pathlib
import tempfile
import zipfile
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

import cloudpickle
from absl.testing import absltest

from kinetic.backend.execution import (
  _FUSE_DATA_MOUNT_PREFIX,
  JobContext,
  _find_requirements,
  _prepare_artifacts,
  _process_volumes,
  _requirements_uri,
  _upload_artifacts,
  submit_remote,
)
from kinetic.data import Data


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
    self.assertEqual(ctx.bucket_name, "my-proj-kn-my-cluster-jobs")
    self.assertEqual(ctx.region, "europe-west4")
    self.assertTrue(ctx.display_name.startswith("kinetic-my_train-"))
    self.assertRegex(ctx.job_id, r"^job-[0-9a-f]{8}$")

  def test_from_params_resolves_zone_from_env(self):
    with mock.patch.dict(
      os.environ,
      {"KINETIC_ZONE": "asia-east1-c", "KINETIC_PROJECT": "env-proj"},
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
      if k not in ("KINETIC_PROJECT", "GOOGLE_CLOUD_PROJECT")
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
      if k not in ("KINETIC_PROJECT", "GOOGLE_CLOUD_PROJECT")
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

  def test_post_init_resolves_working_dir_from_function_module(self):
    with mock.patch(
      "kinetic.backend.execution.inspect.getmodule",
      return_value=SimpleNamespace(__file__="/tmp/project/train.py"),
    ):
      ctx = JobContext(
        func=self._make_func(),
        args=(),
        kwargs={},
        env_vars={},
        accelerator="cpu",
        container_image=None,
        zone="us-central1-a",
        project="proj",
        cluster_name="cluster",
      )

    self.assertEqual(ctx.working_dir, "/tmp/project")

  def test_post_init_falls_back_to_cwd_when_function_module_unknown(self):
    with (
      mock.patch(
        "kinetic.backend.execution.inspect.getmodule",
        return_value=None,
      ),
      mock.patch("kinetic.backend.execution.os.getcwd", return_value="/cwd"),
    ):
      ctx = JobContext(
        func=self._make_func(),
        args=(),
        kwargs={},
        env_vars={},
        accelerator="cpu",
        container_image=None,
        zone="us-central1-a",
        project="proj",
        cluster_name="cluster",
      )

    self.assertEqual(ctx.working_dir, "/cwd")


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
    """Returns None when no requirements.txt or pyproject.toml exists."""
    tmp_path = _make_temp_path(self)
    empty = tmp_path / "empty"
    empty.mkdir()
    self.assertIsNone(_find_requirements(str(empty)))

  def test_finds_pyproject_toml(self):
    """Returns pyproject.toml path when no requirements.txt exists."""
    tmp_path = _make_temp_path(self)
    (tmp_path / "pyproject.toml").write_text(
      '[project]\ndependencies = ["numpy"]\n'
    )
    self.assertEqual(
      _find_requirements(str(tmp_path)),
      str(tmp_path / "pyproject.toml"),
    )

  def test_requirements_txt_preferred_over_pyproject_toml(self):
    """requirements.txt in the same directory wins over pyproject.toml."""
    tmp_path = _make_temp_path(self)
    (tmp_path / "requirements.txt").write_text("numpy\n")
    (tmp_path / "pyproject.toml").write_text(
      '[project]\ndependencies = ["scipy"]\n'
    )
    self.assertEqual(
      _find_requirements(str(tmp_path)),
      str(tmp_path / "requirements.txt"),
    )

  def test_parent_pyproject_toml_found_from_child(self):
    """Walks up to find pyproject.toml in parent when child has nothing."""
    tmp_path = _make_temp_path(self)
    (tmp_path / "pyproject.toml").write_text(
      '[project]\ndependencies = ["numpy"]\n'
    )
    child = tmp_path / "subdir"
    child.mkdir()
    self.assertEqual(
      _find_requirements(str(child)),
      str(tmp_path / "pyproject.toml"),
    )

  def test_child_requirements_txt_beats_parent_pyproject_toml(self):
    """requirements.txt in child dir is found before pyproject.toml in parent."""
    tmp_path = _make_temp_path(self)
    (tmp_path / "pyproject.toml").write_text(
      '[project]\ndependencies = ["scipy"]\n'
    )
    child = tmp_path / "subdir"
    child.mkdir()
    (child / "requirements.txt").write_text("numpy\n")
    self.assertEqual(
      _find_requirements(str(child)),
      str(child / "requirements.txt"),
    )


class TestPrepareArtifactsFuse(absltest.TestCase):
  """Tests for FUSE volume handling in _prepare_artifacts."""

  def _make_func(self):
    def my_train():
      return 42

    return my_train

  def _make_ctx(self, volumes=None, args=(), kwargs=None):
    return JobContext(
      func=self._make_func(),
      args=args,
      kwargs=kwargs or {},
      env_vars={},
      accelerator="cpu",
      container_image=None,
      zone="us-central1-a",
      project="proj",
      cluster_name="kinetic-cluster",
      volumes=volumes,
    )

  @mock.patch("kinetic.backend.execution.storage.upload_data")
  @mock.patch("kinetic.backend.execution.packager.zip_working_dir")
  def test_fuse_volume_creates_fuse_spec(self, _zip, mock_upload):
    mock_upload.return_value = "gs://bucket/hash/"
    tmp = _make_temp_path(self)
    data_dir = tmp / "dataset"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text("data")

    ctx = self._make_ctx(volumes={"/data": Data(str(data_dir), fuse=True)})
    _prepare_artifacts(ctx, str(tmp))

    self.assertIsNotNone(ctx.fuse_volume_specs)
    self.assertLen(ctx.fuse_volume_specs, 1)
    spec = ctx.fuse_volume_specs[0]
    self.assertEqual(spec["gcs_uri"], "gs://bucket/hash/")
    self.assertEqual(spec["mount_path"], "/data")
    self.assertTrue(spec["is_dir"])
    self.assertTrue(spec["read_only"])

  @mock.patch("kinetic.backend.execution.storage.upload_data")
  @mock.patch("kinetic.backend.execution.packager.zip_working_dir")
  def test_non_fuse_volume_no_fuse_specs(self, _zip, mock_upload):
    mock_upload.return_value = "gs://bucket/hash/"
    tmp = _make_temp_path(self)
    data_dir = tmp / "dataset"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text("data")

    ctx = self._make_ctx(volumes={"/data": Data(str(data_dir))})
    _prepare_artifacts(ctx, str(tmp))

    self.assertIsNone(ctx.fuse_volume_specs)

  @mock.patch("kinetic.backend.execution.storage.upload_data")
  @mock.patch("kinetic.backend.execution.packager.zip_working_dir")
  def test_fuse_data_arg_creates_auto_mount(self, _zip, mock_upload):
    mock_upload.return_value = "gs://bucket/hash/"
    tmp = _make_temp_path(self)

    fuse_data = Data("gs://bucket/dataset/", fuse=True)
    ctx = self._make_ctx(args=(fuse_data,))
    _prepare_artifacts(ctx, str(tmp))

    self.assertIsNotNone(ctx.fuse_volume_specs)
    self.assertLen(ctx.fuse_volume_specs, 1)
    spec = ctx.fuse_volume_specs[0]
    self.assertEqual(spec["mount_path"], "/_kinetic/fuse-data/0")
    self.assertTrue(spec["is_dir"])
    self.assertTrue(spec["read_only"])

  @mock.patch("kinetic.backend.execution.storage.upload_data")
  @mock.patch("kinetic.backend.execution.packager.zip_working_dir")
  def test_mixed_fuse_and_non_fuse_volumes(self, _zip, mock_upload):
    mock_upload.return_value = "gs://bucket/hash/"
    tmp = _make_temp_path(self)
    data_dir = tmp / "dataset"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text("data")
    config_dir = tmp / "config"
    config_dir.mkdir()
    (config_dir / "cfg.json").write_text("{}")

    ctx = self._make_ctx(
      volumes={
        "/data": Data(str(data_dir), fuse=True),
        "/config": Data(str(config_dir)),
      }
    )
    _prepare_artifacts(ctx, str(tmp))

    # Only the fuse volume should be in fuse_volume_specs
    self.assertIsNotNone(ctx.fuse_volume_specs)
    self.assertLen(ctx.fuse_volume_specs, 1)
    self.assertEqual(ctx.fuse_volume_specs[0]["mount_path"], "/data")


class TestPrepareArtifacts(absltest.TestCase):
  def _make_working_dir(self):
    """Create a temp working directory with some source files."""
    wd = _make_temp_path(self)
    (wd / "train.py").write_text("print('hello')\n")
    (wd / "utils.py").write_text("x = 1\n")
    return wd

  def _make_ctx(self, working_dir, args=(), kwargs=None, volumes=None):
    def train():
      return 42

    return JobContext(
      func=train,
      args=args,
      kwargs=kwargs or {},
      env_vars={},
      accelerator="cpu",
      container_image=None,
      zone="us-central1-a",
      project="proj",
      cluster_name="kinetic-cluster",
      working_dir=str(working_dir),
      volumes=volumes,
    )

  def _zip_names(self, ctx):
    with zipfile.ZipFile(ctx.context_path) as zf:
      return zf.namelist()

  def _load_payload(self, ctx):
    with open(ctx.payload_path, "rb") as f:
      return cloudpickle.load(f)

  def test_local_data_arg_excluded_from_zip(self):
    working_dir = self._make_working_dir()
    data_dir = working_dir / "dataset"
    data_dir.mkdir()
    (data_dir / "data.csv").write_text("a,b\n1,2\n")
    build_dir = _make_temp_path(self)
    ctx = self._make_ctx(working_dir, args=(Data(str(data_dir)),))

    with mock.patch(
      "kinetic.backend.execution.storage.upload_data",
      return_value="gs://bucket/data",
    ):
      _prepare_artifacts(ctx, str(build_dir))

    names = self._zip_names(ctx)
    self.assertIn("train.py", names)
    self.assertNotIn("dataset/data.csv", names)

  def test_local_data_volume_excluded_from_zip(self):
    working_dir = self._make_working_dir()
    vol_dir = working_dir / "weights"
    vol_dir.mkdir()
    (vol_dir / "model.bin").write_text("weights")
    build_dir = _make_temp_path(self)
    ctx = self._make_ctx(
      working_dir, volumes={"/mnt/weights": Data(str(vol_dir))}
    )

    with mock.patch(
      "kinetic.backend.execution.storage.upload_data",
      return_value="gs://bucket/weights",
    ):
      _prepare_artifacts(ctx, str(build_dir))

    names = self._zip_names(ctx)
    self.assertIn("train.py", names)
    self.assertNotIn("weights/model.bin", names)

  def test_data_arg_replaced_with_ref_in_payload(self):
    working_dir = self._make_working_dir()
    data_file = working_dir / "input.txt"
    data_file.write_text("input")
    build_dir = _make_temp_path(self)
    ctx = self._make_ctx(
      working_dir, args=(Data(str(data_file)), "regular_arg")
    )

    with mock.patch(
      "kinetic.backend.execution.storage.upload_data",
      return_value="gs://bucket/input",
    ):
      _prepare_artifacts(ctx, str(build_dir))

    payload = self._load_payload(ctx)
    self.assertTrue(payload["args"][0].get("__data_ref__"))
    self.assertEqual(payload["args"][0]["gcs_uri"], "gs://bucket/input")
    self.assertEqual(payload["args"][1], "regular_arg")

  def test_volume_ref_in_payload(self):
    working_dir = self._make_working_dir()
    vol_dir = working_dir / "data"
    vol_dir.mkdir()
    (vol_dir / "f.txt").write_text("x")
    build_dir = _make_temp_path(self)
    ctx = self._make_ctx(working_dir, volumes={"/mnt/data": Data(str(vol_dir))})

    with mock.patch(
      "kinetic.backend.execution.storage.upload_data",
      return_value="gs://bucket/data",
    ):
      _prepare_artifacts(ctx, str(build_dir))

    payload = self._load_payload(ctx)
    self.assertLen(payload["volumes"], 1)
    vol_ref = payload["volumes"][0]
    self.assertTrue(vol_ref["__data_ref__"])
    self.assertEqual(vol_ref["gcs_uri"], "gs://bucket/data")
    self.assertEqual(vol_ref["mount_path"], "/mnt/data")

  def test_gcs_data_not_excluded_from_zip(self):
    working_dir = self._make_working_dir()
    build_dir = _make_temp_path(self)
    ctx = self._make_ctx(
      working_dir, args=(Data("gs://bucket/remote-dataset/"),)
    )

    with mock.patch(
      "kinetic.backend.execution.storage.upload_data",
      return_value="gs://bucket/remote-dataset/",
    ):
      _prepare_artifacts(ctx, str(build_dir))

    names = self._zip_names(ctx)
    self.assertIn("train.py", names)
    self.assertIn("utils.py", names)

  def test_sets_artifact_paths_on_ctx(self):
    working_dir = self._make_working_dir()
    (working_dir / "requirements.txt").write_text("numpy\n")
    build_dir = _make_temp_path(self)
    ctx = self._make_ctx(working_dir)

    _prepare_artifacts(ctx, str(build_dir))

    self.assertEqual(
      ctx.payload_path, os.path.join(str(build_dir), "payload.pkl")
    )
    self.assertEqual(
      ctx.context_path, os.path.join(str(build_dir), "context.zip")
    )
    self.assertEqual(
      ctx.requirements_path, str(working_dir / "requirements.txt")
    )


class TestUploadArtifactsRequirementsFlag(absltest.TestCase):
  """Tests that _upload_artifacts returns the correct has_requirements flag."""

  def _make_ctx(self, requirements_path=None, container_image=None):
    def train():
      return 1

    return JobContext(
      func=train,
      args=(),
      kwargs={},
      env_vars={},
      accelerator="v6e-8",
      container_image=container_image,
      zone="us-central1-a",
      project="proj",
      cluster_name="cluster",
      payload_path="/tmp/payload.pkl",
      context_path="/tmp/context.zip",
      requirements_path=requirements_path,
    )

  @mock.patch("kinetic.backend.execution.storage.upload_artifacts")
  @mock.patch(
    "kinetic.backend.execution.container_builder.prepare_requirements_content",
    return_value=None,
  )
  def test_returns_false_when_content_is_none(self, mock_prepare, mock_upload):
    """has_requirements is False when prepare_requirements_content returns None."""
    ctx = self._make_ctx(
      requirements_path="/tmp/requirements.txt", container_image="prebuilt"
    )
    has_requirements = _upload_artifacts(ctx)
    self.assertFalse(has_requirements)

  @mock.patch("kinetic.backend.execution.storage.upload_artifacts")
  @mock.patch(
    "kinetic.backend.execution.container_builder.prepare_requirements_content",
    return_value=None,
  )
  def test_requirements_uri_returns_none_when_path_cleared(
    self, mock_prepare, mock_upload
  ):
    """_requirements_uri returns None after caller clears requirements_path."""
    ctx = self._make_ctx(
      requirements_path="/tmp/requirements.txt", container_image="prebuilt"
    )
    has_requirements = _upload_artifacts(ctx)
    if not has_requirements:
      ctx.requirements_path = None
    self.assertIsNone(_requirements_uri(ctx))

  @mock.patch("kinetic.backend.execution.storage.upload_artifacts")
  @mock.patch(
    "kinetic.backend.execution.container_builder.prepare_requirements_content",
    return_value="numpy==1.26\n",
  )
  def test_returns_true_when_content_exists(self, mock_prepare, mock_upload):
    """has_requirements is True when prepare_requirements_content returns content."""
    ctx = self._make_ctx(
      requirements_path="/tmp/requirements.txt", container_image="prebuilt"
    )
    has_requirements = _upload_artifacts(ctx)
    self.assertTrue(has_requirements)

  @mock.patch("kinetic.backend.execution.storage.upload_artifacts")
  @mock.patch(
    "kinetic.backend.execution.container_builder.prepare_requirements_content",
    return_value="numpy==1.26\n",
  )
  def test_requirements_uri_returned_when_content_exists(
    self, mock_prepare, mock_upload
  ):
    """_requirements_uri returns a GCS URI when requirements content exists."""
    ctx = self._make_ctx(
      requirements_path="/tmp/requirements.txt", container_image="prebuilt"
    )
    _upload_artifacts(ctx)
    self.assertEqual(
      _requirements_uri(ctx),
      f"gs://{ctx.bucket_name}/{ctx.job_id}/requirements.txt",
    )

  @mock.patch("kinetic.backend.execution.storage.upload_artifacts")
  def test_non_prebuilt_skips_filtering(self, mock_upload):
    """Non-prebuilt mode does not call prepare_requirements_content."""
    ctx = self._make_ctx(
      requirements_path="/tmp/requirements.txt",
      container_image="gcr.io/my-proj/custom:latest",
    )
    with mock.patch(
      "kinetic.backend.execution.container_builder.prepare_requirements_content"
    ) as mock_prepare:
      has_requirements = _upload_artifacts(ctx)
      mock_prepare.assert_not_called()
    self.assertTrue(has_requirements)


class TestSubmitRemote(absltest.TestCase):
  def _make_ctx(self):
    def train():
      return 1

    return JobContext(
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

  def _make_backend(self):
    backend = MagicMock()
    backend.namespace = "default"
    backend.get_k8s_name.return_value = "kinetic-job-1234"
    return backend

  def test_handle_uploaded_before_k8s_submit(self):
    ctx = self._make_ctx()
    backend = self._make_backend()
    call_order = []
    backend.submit_job.side_effect = lambda *a, **kw: call_order.append(
      "submit"
    )

    with (
      mock.patch(
        "kinetic.backend.execution.prepare_execution",
        side_effect=lambda _ctx, _b: setattr(_ctx, "image_uri", "img:tag"),
      ),
      mock.patch(
        "kinetic.backend.execution.storage.upload_handle",
        side_effect=lambda *a, **kw: call_order.append("handle"),
      ),
    ):
      submit_remote(ctx, backend)

    self.assertEqual(call_order, ["handle", "submit"])

  def test_conclusive_submit_failure_cleans_up(self):
    ctx = self._make_ctx()
    backend = self._make_backend()
    backend.job_exists.return_value = False
    backend.submit_job.side_effect = RuntimeError("submit failed")

    with (
      mock.patch(
        "kinetic.backend.execution.prepare_execution",
        side_effect=lambda _ctx, _b: setattr(_ctx, "image_uri", "img:tag"),
      ),
      mock.patch("kinetic.backend.execution.storage.upload_handle"),
      mock.patch(
        "kinetic.backend.execution.storage.cleanup_artifacts"
      ) as mock_cleanup,
      self.assertRaisesRegex(RuntimeError, "submit failed"),
    ):
      submit_remote(ctx, backend)

    mock_cleanup.assert_called_once_with(
      ctx.bucket_name, ctx.job_id, project=ctx.project
    )

  def test_ambiguous_submit_failure_returns_handle_when_job_exists(self):
    ctx = self._make_ctx()
    backend = self._make_backend()
    backend.job_exists.return_value = True
    backend.submit_job.side_effect = RuntimeError("transport reset")

    with (
      mock.patch(
        "kinetic.backend.execution.prepare_execution",
        side_effect=lambda _ctx, _b: setattr(_ctx, "image_uri", "img:tag"),
      ),
      mock.patch("kinetic.backend.execution.storage.upload_handle"),
      mock.patch(
        "kinetic.backend.execution.storage.cleanup_artifacts"
      ) as mock_cleanup,
    ):
      handle = submit_remote(ctx, backend)

    self.assertEqual(handle.job_id, ctx.job_id)
    mock_cleanup.assert_not_called()

  def test_reconciliation_failure_cleans_up(self):
    ctx = self._make_ctx()
    backend = self._make_backend()
    backend.job_exists.side_effect = RuntimeError("k8s unreachable")
    backend.submit_job.side_effect = RuntimeError("submit failed")

    with (
      mock.patch(
        "kinetic.backend.execution.prepare_execution",
        side_effect=lambda _ctx, _b: setattr(_ctx, "image_uri", "img:tag"),
      ),
      mock.patch("kinetic.backend.execution.storage.upload_handle"),
      mock.patch(
        "kinetic.backend.execution.storage.cleanup_artifacts"
      ) as mock_cleanup,
      self.assertRaisesRegex(RuntimeError, "submit failed"),
    ):
      submit_remote(ctx, backend)

    mock_cleanup.assert_called_once_with(
      ctx.bucket_name, ctx.job_id, project=ctx.project
    )


class TestProcessVolumesReservedPath(absltest.TestCase):
  """Tests that _process_volumes rejects mount paths under the reserved prefix."""

  def _make_ctx(self, volumes):
    ctx = MagicMock()
    ctx.volumes = volumes
    ctx.bucket_name = "test-bucket"
    ctx.project = "test-project"
    return ctx

  def _make_data_stub(self, *, is_gcs=True, is_dir=False, fuse=False):
    obj = MagicMock()
    obj.is_gcs = is_gcs
    obj.is_dir = is_dir
    obj.fuse = fuse
    obj.path = "gs://b/p"
    return obj

  def test_rejects_direct_child_of_reserved_prefix(self):
    mount_path = f"{_FUSE_DATA_MOUNT_PREFIX}/0"
    ctx = self._make_ctx({mount_path: self._make_data_stub()})

    with self.assertRaises(ValueError) as cm:
      _process_volumes(ctx, "/tmp/caller", set())
    self.assertIn(mount_path, str(cm.exception))

  def test_rejects_nested_path_under_reserved_prefix(self):
    mount_path = f"{_FUSE_DATA_MOUNT_PREFIX}/42/sub"
    ctx = self._make_ctx({mount_path: self._make_data_stub()})

    with self.assertRaises(ValueError) as cm:
      _process_volumes(ctx, "/tmp/caller", set())
    self.assertIn(mount_path, str(cm.exception))

  @mock.patch("kinetic.backend.execution.storage.upload_data")
  def test_allows_non_reserved_path(self, mock_upload):
    mock_upload.return_value = "gs://test-bucket/data/hash"
    ctx = self._make_ctx({"/mnt/my-data": self._make_data_stub()})

    volume_refs, _ = _process_volumes(ctx, "/tmp/caller", set())
    self.assertLen(volume_refs, 1)

  @mock.patch("kinetic.backend.execution.storage.upload_data")
  def test_allows_similar_but_distinct_prefix(self, mock_upload):
    mock_upload.return_value = "gs://test-bucket/data/hash"
    ctx = self._make_ctx(
      {f"{_FUSE_DATA_MOUNT_PREFIX}-extra": self._make_data_stub()}
    )

    volume_refs, _ = _process_volumes(ctx, "/tmp/caller", set())
    self.assertLen(volume_refs, 1)


if __name__ == "__main__":
  absltest.main()
