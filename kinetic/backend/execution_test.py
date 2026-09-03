"""Tests for kinetic.backend.execution — JobContext and submit_remote."""

import os
import pathlib
import sys
import tempfile
import zipfile
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

from absl.testing import absltest

from kinetic.backend import execution
from kinetic.backend.execution import (
  _FUSE_DATA_MOUNT_PREFIX,
  JobContext,
  _find_requirements,
  _prepare_artifacts,
  _process_volumes,
  _requirements_uri,
  _resolve_working_dir,
  _upload_artifacts,
  compute_packaging_plan,
  submit_remote,
)
from kinetic.data import Data
from kinetic.utils import packager


def _make_temp_path(test_case):
  """Create a temp directory that is cleaned up after the test."""
  td = tempfile.TemporaryDirectory()
  test_case.addCleanup(td.cleanup)
  return pathlib.Path(td.name)


def _no_package_root_override(test_case):
  """Ensure KINETIC_PACKAGE_ROOT does not leak into a test."""
  env = {k: v for k, v in os.environ.items() if k != "KINETIC_PACKAGE_ROOT"}
  patcher = mock.patch.dict(os.environ, env, clear=True)
  patcher.start()
  test_case.addCleanup(patcher.stop)


def _mock_packager(test_case):
  """Patch the packager/hash boundary; return the (save, zip) mocks."""
  patchers = [
    mock.patch("kinetic.backend.execution.packager.save_payload"),
    mock.patch("kinetic.backend.execution.packager.zip_working_dir"),
    mock.patch(
      "kinetic.backend.execution._file_sha256", return_value="dummy_hash"
    ),
  ]
  mocks = [p.start() for p in patchers]
  for patcher in patchers:
    test_case.addCleanup(patcher.stop)
  return mocks[0], mocks[1]


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
      self.assertRaisesRegex(ValueError, "No GCP project configured"),
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


class TestResolveWorkingDir(absltest.TestCase):
  """Tests for the interpreter-context guard in _resolve_working_dir."""

  def _make_func(self):
    def train():
      return 1

    return train

  def test_module_without_file_attribute_falls_back_to_cwd(self):
    """A __main__ without __file__ (notebook, python -c) must not crash."""
    module = SimpleNamespace()
    self.assertFalse(hasattr(module, "__file__"))
    with (
      mock.patch(
        "kinetic.backend.execution.inspect.getmodule", return_value=module
      ),
      mock.patch("kinetic.backend.execution.os.getcwd", return_value="/cwd"),
    ):
      self.assertEqual(_resolve_working_dir(self._make_func()), "/cwd")

  def test_module_with_none_file_falls_back_to_cwd(self):
    with (
      mock.patch(
        "kinetic.backend.execution.inspect.getmodule",
        return_value=SimpleNamespace(__file__=None),
      ),
      mock.patch("kinetic.backend.execution.os.getcwd", return_value="/cwd"),
    ):
      self.assertEqual(_resolve_working_dir(self._make_func()), "/cwd")

  def test_normal_module_unchanged(self):
    with mock.patch(
      "kinetic.backend.execution.inspect.getmodule",
      return_value=SimpleNamespace(__file__="/tmp/project/train.py"),
    ):
      self.assertEqual(_resolve_working_dir(self._make_func()), "/tmp/project")


class TestFindRequirements(absltest.TestCase):
  def test_finds_in_start_dir(self):
    """Returns the path when requirements.txt exists in the start directory."""
    tmp_path = _make_temp_path(self)
    (tmp_path / "requirements.txt").write_text("numpy\n")
    self.assertEqual(
      _find_requirements(str(tmp_path)),
      str(tmp_path / "requirements.txt"),
    )

  def test_directory_named_like_dependency_file_is_skipped(self):
    """A directory named requirements.txt/pyproject.toml must not be selected."""
    tmp_path = _make_temp_path(self)
    (tmp_path / "requirements.txt").write_text("numpy\n")
    child = tmp_path / "subdir"
    child.mkdir()
    (child / "requirements.txt").mkdir()
    (child / "pyproject.toml").mkdir()

    self.assertEqual(
      _find_requirements(str(child)),
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


class TestFindRequirementsBoundary(absltest.TestCase):
  """The upward walk must stop at the repository / home boundary."""

  def _repo_tree(self, git_as_file=False):
    """Build base/requirements.txt + base/repo/{.git,proj}."""
    tmp_path = _make_temp_path(self)
    base = tmp_path / "base"
    base.mkdir()
    (base / "requirements.txt").write_text("flask==1.0\n")
    repo = base / "repo"
    repo.mkdir()
    if git_as_file:
      (repo / ".git").write_text("gitdir: /elsewhere/.git/worktrees/repo\n")
    else:
      (repo / ".git").mkdir()
    proj = repo / "proj"
    proj.mkdir()
    return base, repo, proj

  def test_stops_at_git_directory(self):
    """A stray requirements.txt above the repo root is not adopted."""
    _, _, proj = self._repo_tree()
    self.assertIsNone(_find_requirements(str(proj)))

  def test_stops_at_git_file_worktree(self):
    """`.git` as a file (worktrees, submodules) is also a boundary."""
    _, _, proj = self._repo_tree(git_as_file=True)
    self.assertIsNone(_find_requirements(str(proj)))

  def test_repo_root_itself_is_examined(self):
    """The directory holding `.git` is searched before the walk stops."""
    _, repo, proj = self._repo_tree()
    (repo / "requirements.txt").write_text("numpy\n")
    self.assertEqual(
      _find_requirements(str(proj)), str(repo / "requirements.txt")
    )

  def test_stops_at_home_directory(self):
    """The walk never ascends above $HOME."""
    tmp_path = _make_temp_path(self)
    (tmp_path / "requirements.txt").write_text("flask==1.0\n")
    home = tmp_path / "home"
    home.mkdir()
    proj = home / "code"
    proj.mkdir()
    with mock.patch.dict(os.environ, {"HOME": str(home)}):
      self.assertIsNone(_find_requirements(str(proj)))

  def test_home_directory_itself_is_examined(self):
    tmp_path = _make_temp_path(self)
    home = tmp_path / "home"
    home.mkdir()
    (home / "requirements.txt").write_text("numpy\n")
    proj = home / "code"
    proj.mkdir()
    with mock.patch.dict(os.environ, {"HOME": str(home)}):
      self.assertEqual(
        _find_requirements(str(proj)), str(home / "requirements.txt")
      )


class TestFindRequirementsLogging(absltest.TestCase):
  """Discovery must tell the user which file it picked and why."""

  def test_selection_is_logged(self):
    tmp_path = _make_temp_path(self)
    (tmp_path / "requirements.txt").write_text("numpy\n")
    with mock.patch("kinetic.backend.execution.logging") as mock_logging:
      _find_requirements(str(tmp_path))
    logged = [c.args[0] for c in mock_logging.info.call_args_list]
    self.assertIn("Using dependency file: %s", logged)

  def test_both_files_at_same_level_logged(self):
    tmp_path = _make_temp_path(self)
    (tmp_path / "requirements.txt").write_text("numpy\n")
    (tmp_path / "pyproject.toml").write_text(
      '[project]\ndependencies = ["scipy"]\n'
    )
    with mock.patch("kinetic.backend.execution.logging") as mock_logging:
      _find_requirements(str(tmp_path))
    messages = " ".join(str(c.args) for c in mock_logging.info.call_args_list)
    self.assertIn("Both requirements.txt and pyproject.toml", messages)

  def test_warns_when_file_is_outside_the_project(self):
    """Degraded mode: no repo marker and the file came from a parent."""
    tmp_path = _make_temp_path(self)
    (tmp_path / "requirements.txt").write_text("flask==1.0\n")
    proj = tmp_path / "proj"
    proj.mkdir()
    with mock.patch("kinetic.backend.execution.logging") as mock_logging:
      _find_requirements(str(proj))
    mock_logging.warning.assert_called_once()
    self.assertIn(
      "outside your code directory", mock_logging.warning.call_args.args[0]
    )

  def test_no_outside_warning_within_repo(self):
    tmp_path = _make_temp_path(self)
    (tmp_path / ".git").mkdir()
    (tmp_path / "requirements.txt").write_text("numpy\n")
    proj = tmp_path / "proj"
    proj.mkdir()
    with mock.patch("kinetic.backend.execution.logging") as mock_logging:
      _find_requirements(str(proj))
    mock_logging.warning.assert_not_called()

  def test_no_outside_warning_for_local_file(self):
    tmp_path = _make_temp_path(self)
    (tmp_path / "requirements.txt").write_text("numpy\n")
    with mock.patch("kinetic.backend.execution.logging") as mock_logging:
      _find_requirements(str(tmp_path))
    mock_logging.warning.assert_not_called()

  def test_warns_when_pyproject_declares_no_dependencies(self):
    tmp_path = _make_temp_path(self)
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\nline-length = 80\n")
    with mock.patch("kinetic.backend.execution.logging") as mock_logging:
      selected = _find_requirements(str(tmp_path))
    self.assertEqual(selected, str(tmp_path / "pyproject.toml"))
    warnings = " ".join(
      str(c.args) for c in mock_logging.warning.call_args_list
    )
    self.assertIn("no [project].dependencies", warnings)

  def test_no_warning_when_pyproject_declares_dependencies(self):
    tmp_path = _make_temp_path(self)
    (tmp_path / "pyproject.toml").write_text(
      '[project]\ndependencies = ["numpy"]\n'
    )
    with mock.patch("kinetic.backend.execution.logging") as mock_logging:
      _find_requirements(str(tmp_path))
    mock_logging.warning.assert_not_called()

  def test_unparseable_pyproject_does_not_raise_during_discovery(self):
    tmp_path = _make_temp_path(self)
    (tmp_path / "pyproject.toml").write_text("[project\n")
    self.assertEqual(
      _find_requirements(str(tmp_path)), str(tmp_path / "pyproject.toml")
    )


class TestPrepareArtifactsFuse(absltest.TestCase):
  """Tests for FUSE volume handling in _prepare_artifacts."""

  def setUp(self):
    super().setUp()
    _no_package_root_override(self)
    _mock_packager(self)

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
  def test_fuse_volume_creates_fuse_spec(self, mock_upload):
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
    self.assertEqual(ctx.payload_sha256, "dummy_hash")
    self.assertEqual(ctx.context_sha256, "dummy_hash")

  @mock.patch("kinetic.backend.execution.storage.upload_data")
  def test_non_fuse_volume_no_fuse_specs(self, mock_upload):
    mock_upload.return_value = "gs://bucket/hash/"
    tmp = _make_temp_path(self)
    data_dir = tmp / "dataset"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text("data")

    ctx = self._make_ctx(volumes={"/data": Data(str(data_dir))})
    _prepare_artifacts(ctx, str(tmp))

    self.assertIsNone(ctx.fuse_volume_specs)

  @mock.patch("kinetic.backend.execution.storage.upload_data")
  def test_fuse_data_arg_creates_auto_mount(self, mock_upload):
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
  def test_fuse_gcs_object_keeps_its_own_uri(self, mock_upload):
    """A GCS-native object is already file-level; nothing is appended.

    ``build_gcs_fuse_volumes`` mounts the object's parent, and the pod
    picks the object back out of it by name — both need the URI to keep
    naming the object.
    """
    mock_upload.side_effect = lambda bucket, data, project: data.path
    tmp = _make_temp_path(self)

    fuse_data = Data("gs://bucket/datasets/weights.h5", fuse=True)
    ctx = self._make_ctx(args=(fuse_data,))
    _prepare_artifacts(ctx, str(tmp))

    spec = ctx.fuse_volume_specs[0]
    self.assertEqual(spec["gcs_uri"], "gs://bucket/datasets/weights.h5")
    self.assertFalse(spec["is_dir"])

  @mock.patch("kinetic.backend.execution.storage.upload_data")
  def test_fuse_uploaded_file_uri_gains_the_filename(self, mock_upload):
    """An uploaded file's URI is the hash dir, so the name is appended."""
    mock_upload.return_value = "gs://bucket/ns/data-cache/abc123"
    tmp = _make_temp_path(self)
    config = tmp / "config.json"
    config.write_text("{}")

    ctx = self._make_ctx(args=(Data(str(config), fuse=True),))
    _prepare_artifacts(ctx, str(tmp))

    spec = ctx.fuse_volume_specs[0]
    self.assertEqual(
      spec["gcs_uri"], "gs://bucket/ns/data-cache/abc123/config.json"
    )
    self.assertFalse(spec["is_dir"])

  @mock.patch("kinetic.backend.execution.storage.upload_data")
  def test_mixed_fuse_and_non_fuse_volumes(self, mock_upload):
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
  """_prepare_artifacts wiring, asserted at the packager boundary."""

  def setUp(self):
    super().setUp()
    _no_package_root_override(self)
    self.real_zip = packager.zip_working_dir
    self.mock_save, self.mock_zip = _mock_packager(self)

  def _make_working_dir(self):
    """Create a temp project root with some source files."""
    wd = _make_temp_path(self)
    (wd / ".git").mkdir()
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

  def _run(self, ctx, upload_uri="gs://bucket/data"):
    build_dir = _make_temp_path(self)
    with mock.patch(
      "kinetic.backend.execution.storage.upload_data",
      return_value=upload_uri,
    ):
      _prepare_artifacts(ctx, str(build_dir))
    return build_dir

  def _zip_call(self):
    return self.mock_zip.call_args

  def _excludes(self):
    return self._zip_call().kwargs["exclude_paths"]

  def test_local_data_arg_excluded_from_zip(self):
    working_dir = self._make_working_dir()
    data_dir = working_dir / "dataset"
    data_dir.mkdir()
    (data_dir / "data.csv").write_text("a,b\n1,2\n")
    ctx = self._make_ctx(working_dir, args=(Data(str(data_dir)),))

    build_dir = self._run(ctx)

    self.assertIn(str(data_dir), self._excludes())
    # Re-run the real archiver with the captured root/excludes to confirm
    # the exclusion actually takes effect for the packaged tree.
    zip_path = str(build_dir / "real-context.zip")
    self.real_zip(
      self._zip_call().args[0], zip_path, exclude_paths=self._excludes()
    )
    with zipfile.ZipFile(zip_path) as zf:
      names = zf.namelist()
    self.assertIn("train.py", names)
    self.assertNotIn("dataset/data.csv", names)

  def test_local_data_volume_excluded_from_zip(self):
    working_dir = self._make_working_dir()
    vol_dir = working_dir / "weights"
    vol_dir.mkdir()
    (vol_dir / "model.bin").write_text("weights")
    ctx = self._make_ctx(
      working_dir, volumes={"/mnt/weights": Data(str(vol_dir))}
    )

    self._run(ctx)

    self.assertIn(str(vol_dir), self._excludes())

  def test_data_outside_working_dir_but_inside_package_root_excluded(self):
    """Data dirs elsewhere in the packaged tree must still be excluded."""
    root = self._make_working_dir()
    pkg = root / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "train.py").write_text("x = 1\n")
    data_dir = root / "dataset"
    data_dir.mkdir()
    (data_dir / "data.csv").write_text("a,b\n1,2\n")
    ctx = self._make_ctx(pkg, args=(Data(str(data_dir)),))

    build_dir = self._run(ctx)

    self.assertEqual(self._zip_call().args[0], str(root))
    self.assertIn(str(data_dir), self._excludes())
    zip_path = str(build_dir / "real-context.zip")
    self.real_zip(str(root), zip_path, exclude_paths=self._excludes())
    with zipfile.ZipFile(zip_path) as zf:
      names = zf.namelist()
    self.assertIn(os.path.join("pkg", "train.py"), names)
    self.assertNotIn(os.path.join("dataset", "data.csv"), names)

  def test_data_arg_replaced_with_ref_in_payload(self):
    working_dir = self._make_working_dir()
    data_file = working_dir / "input.txt"
    data_file.write_text("input")
    ctx = self._make_ctx(
      working_dir, args=(Data(str(data_file)), "regular_arg")
    )

    self._run(ctx, upload_uri="gs://bucket/input")

    saved_args = self.mock_save.call_args.args[1]
    self.assertTrue(saved_args[0].get("__data_ref__"))
    self.assertEqual(saved_args[0]["uri"], "gs://bucket/input")
    self.assertEqual(saved_args[1], "regular_arg")

  def test_volume_ref_in_payload(self):
    working_dir = self._make_working_dir()
    vol_dir = working_dir / "data"
    vol_dir.mkdir()
    (vol_dir / "f.txt").write_text("x")
    ctx = self._make_ctx(working_dir, volumes={"/mnt/data": Data(str(vol_dir))})

    self._run(ctx)

    volumes = self.mock_save.call_args.kwargs["volumes"]
    self.assertLen(volumes, 1)
    self.assertTrue(volumes[0]["__data_ref__"])
    self.assertEqual(volumes[0]["uri"], "gs://bucket/data")
    self.assertEqual(volumes[0]["mount_path"], "/mnt/data")

  def test_gcs_data_not_excluded_from_zip(self):
    working_dir = self._make_working_dir()
    ctx = self._make_ctx(
      working_dir, args=(Data("gs://bucket/remote-dataset/"),)
    )

    self._run(ctx, upload_uri="gs://bucket/remote-dataset/")

    self.assertEmpty(self._excludes())

  def test_hf_data_path_not_added_to_excludes(self):
    working_dir = self._make_working_dir()
    ctx = self._make_ctx(working_dir, args=(Data("hf://imdb?split=train"),))

    self._run(ctx, upload_uri="hf://imdb?split=train")

    self.assertEmpty(self._excludes())

  def test_sets_artifact_paths_on_ctx(self):
    working_dir = self._make_working_dir()
    (working_dir / "requirements.txt").write_text("numpy\n")
    ctx = self._make_ctx(working_dir)

    build_dir = self._run(ctx)

    self.assertEqual(
      ctx.payload_path, os.path.join(str(build_dir), "payload.pkl")
    )
    self.assertEqual(
      ctx.context_path, os.path.join(str(build_dir), "context.zip")
    )
    self.assertEqual(
      ctx.requirements_path, str(working_dir / "requirements.txt")
    )

  def test_zips_from_package_root_not_working_dir(self):
    root = self._make_working_dir()
    pkg = root / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    ctx = self._make_ctx(pkg)

    self._run(ctx)

    self.assertEqual(self._zip_call().args[0], str(root))
    self.assertEqual(ctx.working_dir, str(pkg))

  def test_plan_fields_passed_to_packager(self):
    root = self._make_working_dir()
    pkg = root / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    ctx = self._make_ctx(pkg)

    self._run(ctx)

    extra = self.mock_save.call_args.kwargs["payload_extra"]
    self.assertEqual(extra["package_root"], str(root))
    self.assertEqual(extra["entry_rel"], "pkg")
    self.assertEqual(extra["sys_path_rel"][0], "")
    self.assertIn("client_cwd_rel", extra)
    self.assertEqual(self.mock_save.call_args.kwargs["package_root"], str(root))
    self.assertEqual(self.mock_save.call_args.kwargs["working_dir"], str(pkg))
    self.assertEqual(self._zip_call().kwargs["plan_json"], extra)

  def test_requirements_search_starts_at_working_dir(self):
    root = self._make_working_dir()
    pkg = root / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "requirements.txt").write_text("numpy\n")
    (root / "requirements.txt").write_text("scipy\n")
    ctx = self._make_ctx(pkg)

    self._run(ctx)

    self.assertEqual(ctx.requirements_path, str(pkg / "requirements.txt"))


class TestComputePackagingPlan(absltest.TestCase):
  """Tests for the packaging plan (root detection + remote layout hints)."""

  def setUp(self):
    super().setUp()
    _no_package_root_override(self)

  def _func(self):
    def train():
      return 1

    return train

  def _home(self):
    """Create a temp tree with $HOME pinned inside it."""
    tmp = _make_temp_path(self)
    home = tmp / "home"
    home.mkdir()
    patcher = mock.patch.dict(os.environ, {"HOME": str(home)})
    patcher.start()
    self.addCleanup(patcher.stop)
    return tmp, home

  def _plan(self, working_dir):
    return compute_packaging_plan(self._func(), working_dir=str(working_dir))

  def _mkdirs(self, *paths):
    for path in paths:
      path.mkdir(parents=True, exist_ok=True)

  def test_flat_project_root_is_working_dir(self):
    _, home = self._home()
    proj = home / "proj"
    self._mkdirs(proj)
    (proj / "train.py").write_text("x = 1\n")

    plan = self._plan(proj)

    self.assertIsInstance(plan, execution.PackagingPlan)
    self.assertEqual(plan.package_root, str(proj))
    self.assertEqual(plan.working_dir, str(proj))
    self.assertEqual(plan.entry_rel, "")
    self.assertEqual(plan.sys_path_rel[0], "")

  def test_marker_at_repo_root(self):
    _, home = self._home()
    repo = home / "repo"
    proj = repo / "proj"
    self._mkdirs(proj)
    (repo / "pyproject.toml").write_text("[project]\n")

    plan = self._plan(proj)

    self.assertEqual(plan.package_root, str(repo))
    self.assertEqual(plan.entry_rel, "proj")

  def test_package_escape_to_repo_root(self):
    _, home = self._home()
    repo = home / "repo"
    pkg = repo / "pkg"
    self._mkdirs(pkg)
    (repo / ".git").mkdir()
    (pkg / "__init__.py").write_text("")

    plan = self._plan(pkg)

    self.assertEqual(plan.package_root, str(repo))
    self.assertEqual(plan.entry_rel, "pkg")

  def test_git_as_file_is_a_marker(self):
    _, home = self._home()
    repo = home / "repo"
    proj = repo / "proj"
    self._mkdirs(proj)
    (repo / ".git").write_text("gitdir: /elsewhere\n")

    self.assertEqual(self._plan(proj).package_root, str(repo))

  def test_nested_packages(self):
    _, home = self._home()
    repo = home / "repo"
    sub = repo / "pkg" / "sub"
    self._mkdirs(sub)
    (repo / "setup.py").write_text("")
    (repo / "pkg" / "__init__.py").write_text("")
    (sub / "__init__.py").write_text("")

    plan = self._plan(sub)

    self.assertEqual(plan.package_root, str(repo))
    self.assertEqual(plan.entry_rel, os.path.join("pkg", "sub"))

  def test_src_layout_records_sys_path_entry(self):
    _, home = self._home()
    repo = home / "repo"
    app = repo / "src" / "app"
    self._mkdirs(app)
    (repo / "setup.cfg").write_text("")
    (app / "__init__.py").write_text("")

    with mock.patch.object(sys, "path", [str(repo / "src"), "/usr/lib/py"]):
      plan = self._plan(app)

    self.assertEqual(plan.package_root, str(repo))
    self.assertEqual(plan.entry_rel, os.path.join("src", "app"))
    self.assertEqual(plan.sys_path_rel, ["", "src"])

  def test_sys_path_excludes_site_packages_and_outside_entries(self):
    _, home = self._home()
    repo = home / "repo"
    site = repo / ".venv" / "lib" / "site-packages"
    other = home / "elsewhere"
    self._mkdirs(site, other, repo / "libs")
    (repo / "pyproject.toml").write_text("[project]\n")

    with mock.patch.object(
      sys,
      "path",
      [
        str(site),
        str(other),
        str(repo / "libs"),
        str(repo / "libs"),
        str(repo / "missing"),
      ],
    ):
      plan = self._plan(repo)

    self.assertEqual(plan.sys_path_rel, ["", "libs"])

  def test_no_marker_falls_back_to_dir_above_package(self):
    _, home = self._home()
    proj = home / "proj"
    pkg = proj / "pkg"
    self._mkdirs(pkg)
    (pkg / "__init__.py").write_text("")

    plan = self._plan(pkg)

    self.assertEqual(plan.package_root, str(proj))
    self.assertEqual(plan.entry_rel, "pkg")

  def test_home_is_never_adopted_as_root(self):
    _, home = self._home()
    (home / "pyproject.toml").write_text("[project]\n")
    proj = home / "proj"
    self._mkdirs(proj)

    self.assertEqual(self._plan(proj).package_root, str(proj))

  def test_home_allowed_when_it_is_the_working_dir(self):
    _, home = self._home()
    (home / "pyproject.toml").write_text("[project]\n")

    self.assertEqual(self._plan(home).package_root, str(home))

  def test_env_override_wins_over_detection(self):
    _, home = self._home()
    repo = home / "repo"
    inner = repo / "a" / "b"
    self._mkdirs(inner)
    (inner / "pyproject.toml").write_text("[project]\n")

    with mock.patch.dict(os.environ, {"KINETIC_PACKAGE_ROOT": str(repo)}):
      plan = self._plan(inner)

    self.assertEqual(plan.package_root, str(repo))
    self.assertEqual(plan.entry_rel, os.path.join("a", "b"))

  def test_env_override_must_be_an_existing_directory(self):
    _, home = self._home()
    repo = home / "repo"
    self._mkdirs(repo)
    missing = home / "nope"

    with (
      mock.patch.dict(os.environ, {"KINETIC_PACKAGE_ROOT": str(missing)}),
      self.assertRaisesRegex(ValueError, "not an existing directory"),
    ):
      self._plan(repo)

  def test_sys_path_non_string_entries_ignored(self):
    _, home = self._home()
    repo = home / "repo"
    src = repo / "src"
    self._mkdirs(src)
    (repo / "setup.cfg").write_text("")

    fake_path = [str(src), object(), 42, None]
    with mock.patch.object(sys, "path", fake_path):
      plan = self._plan(repo)

    self.assertEqual(plan.sys_path_rel, ["", "src"])

  def test_env_override_must_be_an_ancestor(self):
    _, home = self._home()
    repo = home / "repo"
    other = home / "other"
    self._mkdirs(repo, other)

    with (
      mock.patch.dict(os.environ, {"KINETIC_PACKAGE_ROOT": str(other)}),
      self.assertRaisesRegex(ValueError, "KINETIC_PACKAGE_ROOT"),
    ):
      self._plan(repo)

  def test_env_override_equal_to_working_dir(self):
    _, home = self._home()
    repo = home / "repo"
    self._mkdirs(repo)

    with mock.patch.dict(os.environ, {"KINETIC_PACKAGE_ROOT": str(repo)}):
      plan = self._plan(repo)

    self.assertEqual(plan.package_root, str(repo))
    self.assertEqual(plan.entry_rel, "")

  def test_client_cwd_rel_when_cwd_inside_root(self):
    _, home = self._home()
    repo = home / "repo"
    proj = repo / "proj"
    self._mkdirs(proj)
    (repo / ".git").mkdir()
    previous = os.getcwd()
    self.addCleanup(os.chdir, previous)
    os.chdir(str(proj))

    self.assertEqual(self._plan(proj).client_cwd_rel, "proj")

  def test_client_cwd_rel_none_when_cwd_outside_root(self):
    _, home = self._home()
    repo = home / "repo"
    outside = home / "outside"
    self._mkdirs(repo, outside)
    (repo / ".git").mkdir()
    previous = os.getcwd()
    self.addCleanup(os.chdir, previous)
    os.chdir(str(outside))

    self.assertIsNone(self._plan(repo).client_cwd_rel)

  def test_working_dir_defaults_to_function_module_dir(self):
    _, home = self._home()
    proj = home / "proj"
    self._mkdirs(proj)
    with mock.patch(
      "kinetic.backend.execution.inspect.getmodule",
      return_value=SimpleNamespace(__file__=str(proj / "train.py")),
    ):
      plan = compute_packaging_plan(self._func())

    self.assertEqual(plan.working_dir, str(proj))
    self.assertEqual(plan.package_root, str(proj))

  def test_notebook_function_packages_the_cwd(self):
    _, home = self._home()
    proj = home / "proj"
    self._mkdirs(proj)
    previous = os.getcwd()
    self.addCleanup(os.chdir, previous)
    os.chdir(str(proj))

    with mock.patch(
      "kinetic.backend.execution.inspect.getmodule",
      return_value=SimpleNamespace(),
    ):
      plan = compute_packaging_plan(self._func())

    self.assertEqual(
      os.path.realpath(plan.working_dir), os.path.realpath(str(proj))
    )
    self.assertEqual(plan.client_cwd_rel, "")


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
