"""Tests for kinetic.utils.packager — zip and payload serialization."""

import collections
import importlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import typing
import zipfile
from unittest import mock

import cloudpickle
import numpy as np
from absl.testing import absltest

from kinetic import version
from kinetic.data import Data
from kinetic.utils import packager
from kinetic.utils.packager import (
  extract_data_refs,
  replace_data_with_refs,
  save_payload,
  zip_working_dir,
)

_REPO_ROOT = os.path.dirname(
  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _make_temp_path(test_case):
  """Create a temp directory that is cleaned up after the test."""
  td = tempfile.TemporaryDirectory()
  test_case.addCleanup(td.cleanup)
  return pathlib.Path(td.name)


class Point(typing.NamedTuple):
  """typing.NamedTuple used to check field preservation."""

  x: int
  data: object


# Deliberately the collections flavour: it reconstructs differently from
# typing.NamedTuple and regressed separately (v2-01/v2-02).
ClassicTuple = collections.namedtuple("ClassicTuple", "a b")  # noqa: PYI024
SingleField = collections.namedtuple("SingleField", "only")  # noqa: PYI024


class ListSubclass(list):
  """Plain list subclass — reconstructible from an iterable."""


class StrictDict(dict):
  """Dict subclass that cannot be rebuilt from a mapping or from nothing."""

  def __init__(self, tag):
    super().__init__()
    self.tag = tag


class TaggedList(list):
  """List subclass whose ``__init__`` accepts but discards the items."""

  def __init__(self, tag):
    super().__init__()
    self.tag = tag


class TestZipWorkingDir(absltest.TestCase):
  def _zip_and_list(self, src, tmp_path, exclude_paths=None, plan_json=None):
    """Zip src directory and return the set of archive member names."""
    out = tmp_path / "context.zip"
    zip_working_dir(
      str(src), str(out), exclude_paths=exclude_paths, plan_json=plan_json
    )
    with zipfile.ZipFile(str(out)) as zf:
      return set(zf.namelist())

  def test_contains_all_files(self):
    tmp_path = _make_temp_path(self)
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("a")
    (src / "b.txt").write_text("b")

    self.assertEqual(self._zip_and_list(src, tmp_path), {"a.py", "b.txt"})

  def test_excludes_git_directory(self):
    tmp_path = _make_temp_path(self)
    src = tmp_path / "src"
    src.mkdir()
    git_dir = src / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("git config")
    (src / "main.py").write_text("code")

    names = self._zip_and_list(src, tmp_path)
    self.assertTrue(all(".git" not in n for n in names))
    self.assertIn("main.py", names)

  def test_excludes_pycache_directory(self):
    tmp_path = _make_temp_path(self)
    src = tmp_path / "src"
    src.mkdir()
    cache_dir = src / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "mod.cpython-312.pyc").write_bytes(b"\x00")
    (src / "mod.py").write_text("code")

    names = self._zip_and_list(src, tmp_path)
    self.assertTrue(all("__pycache__" not in n for n in names))
    self.assertIn("mod.py", names)

  def test_preserves_nested_structure(self):
    tmp_path = _make_temp_path(self)
    src = tmp_path / "src"
    sub = src / "pkg" / "sub"
    sub.mkdir(parents=True)
    (sub / "deep.py").write_text("deep")
    (src / "top.py").write_text("top")

    names = self._zip_and_list(src, tmp_path)
    self.assertIn("top.py", names)
    self.assertIn(os.path.join("pkg", "sub", "deep.py"), names)

  def test_empty_directory(self):
    tmp_path = _make_temp_path(self)
    src = tmp_path / "empty"
    src.mkdir()

    self.assertEqual(self._zip_and_list(src, tmp_path), set())

  def test_exclude_directory(self):
    tmp_path = _make_temp_path(self)
    src = tmp_path / "src"
    src.mkdir()
    data_dir = src / "data"
    data_dir.mkdir()
    (data_dir / "big.csv").write_text("lots of data")
    (src / "main.py").write_text("code")

    names = self._zip_and_list(src, tmp_path, exclude_paths={str(data_dir)})
    self.assertIn("main.py", names)
    self.assertTrue(all("data" not in n for n in names))

  def test_exclude_single_file(self):
    tmp_path = _make_temp_path(self)
    src = tmp_path / "src"
    src.mkdir()
    big_file = src / "weights.h5"
    big_file.write_text("model weights")
    (src / "main.py").write_text("code")

    names = self._zip_and_list(src, tmp_path, exclude_paths={str(big_file)})
    self.assertIn("main.py", names)
    self.assertNotIn("weights.h5", names)

  def test_exclude_multiple_paths(self):
    tmp_path = _make_temp_path(self)
    src = tmp_path / "src"
    src.mkdir()
    d1 = src / "data1"
    d1.mkdir()
    (d1 / "a.csv").write_text("a")
    d2 = src / "data2"
    d2.mkdir()
    (d2 / "b.csv").write_text("b")
    (src / "main.py").write_text("code")

    names = self._zip_and_list(src, tmp_path, exclude_paths={str(d1), str(d2)})
    self.assertEqual(names, {"main.py"})

  def test_symlinked_directory_is_archived(self):
    """v4-01: a symlinked package dir must ship, not be silently dropped."""
    tmp_path = _make_temp_path(self)
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("code")
    external = tmp_path / "shared"
    external.mkdir()
    (external / "lib.py").write_text("shared code")
    os.symlink(str(external), str(src / "shared"))

    out = tmp_path / "context.zip"
    zip_working_dir(str(src), str(out))
    with zipfile.ZipFile(str(out)) as zf:
      self.assertIn("shared/lib.py", set(zf.namelist()))
      self.assertEqual(zf.read("shared/lib.py"), b"shared code")

  def test_symlink_cycle_terminates_with_warning(self):
    tmp_path = _make_temp_path(self)
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("code")
    os.symlink(str(src), str(src / "loop"))

    with mock.patch("kinetic.utils.packager.logging") as mock_logging:
      names = self._zip_and_list(src, tmp_path)

    self.assertIn("main.py", names)
    self.assertTrue(all("loop" not in n for n in names))
    warnings = " ".join(
      str(c.args) for c in mock_logging.warning.call_args_list
    )
    self.assertIn("symlink loop", warnings)

  def test_broken_symlink_is_skipped(self):
    """v4-03: one dangling symlink must not abort the submission."""
    tmp_path = _make_temp_path(self)
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("code")
    os.symlink(str(src / "gone.py"), str(src / "dangling.py"))

    with mock.patch("kinetic.utils.packager.logging") as mock_logging:
      names = self._zip_and_list(src, tmp_path)

    self.assertEqual(names, {"main.py"})
    warnings = " ".join(
      str(c.args) for c in mock_logging.warning.call_args_list
    )
    self.assertIn("broken symlink", warnings)

  def test_executable_bit_recorded(self):
    """v4-04: the archive must carry the POSIX mode so extract can restore."""
    tmp_path = _make_temp_path(self)
    src = tmp_path / "src"
    src.mkdir()
    script = src / "run.sh"
    script.write_text("#!/bin/sh\necho hi\n")
    os.chmod(str(script), 0o755)

    out = tmp_path / "context.zip"
    zip_working_dir(str(src), str(out))
    with zipfile.ZipFile(str(out)) as zf:
      mode = zf.getinfo("run.sh").external_attr >> 16
    self.assertTrue(mode & 0o111)

  def test_empty_directory_is_preserved(self):
    """v4-05: scaffolded output dirs must exist on the pod."""
    tmp_path = _make_temp_path(self)
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("code")
    (src / "checkpoints").mkdir()

    names = self._zip_and_list(src, tmp_path)
    self.assertIn("checkpoints/", names)

  def test_pre_1980_mtime_does_not_abort(self):
    """v4-08: an ancient mtime used to raise ValueError during packaging."""
    tmp_path = _make_temp_path(self)
    src = tmp_path / "src"
    src.mkdir()
    old = src / "ancient.txt"
    old.write_text("old")
    os.utime(str(old), (0, 0))

    self.assertEqual(self._zip_and_list(src, tmp_path), {"ancient.txt"})

  def test_default_excludes_drop_junk_directories(self):
    tmp_path = _make_temp_path(self)
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("code")
    for junk in (".venv", "node_modules", ".pytest_cache"):
      d = src / junk
      d.mkdir()
      (d / "junk.bin").write_text("junk")
    (src / ".DS_Store").write_text("finder")

    self.assertEqual(self._zip_and_list(src, tmp_path), {"main.py"})

  def test_no_default_excludes_env_restores_old_behavior(self):
    tmp_path = _make_temp_path(self)
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("code")
    venv = src / ".venv"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("cfg")

    with mock.patch.dict(os.environ, {"KINETIC_NO_DEFAULT_EXCLUDES": "1"}):
      names = self._zip_and_list(src, tmp_path)
    self.assertIn(os.path.join(".venv", "pyvenv.cfg"), names)

  def test_kineticignore_patterns(self):
    tmp_path = _make_temp_path(self)
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("code")
    (src / "big.ckpt").write_text("weights")
    data = src / "data"
    data.mkdir()
    (data / "train.csv").write_text("rows")
    (src / ".kineticignore").write_text(
      "# junk\n*.ckpt\ndata/\n\n",
    )

    names = self._zip_and_list(src, tmp_path)
    self.assertIn("main.py", names)
    self.assertNotIn("big.ckpt", names)
    self.assertTrue(all("data" not in n for n in names))

  def test_size_warning_lists_largest_files(self):
    tmp_path = _make_temp_path(self)
    src = tmp_path / "src"
    src.mkdir()
    (src / "small.py").write_text("x")
    (src / "huge.bin").write_bytes(os.urandom(200_000))

    with (
      mock.patch.dict(os.environ, {"KINETIC_CONTEXT_SIZE_WARN_MB": "0.01"}),
      mock.patch("kinetic.utils.packager.logging") as mock_logging,
    ):
      self._zip_and_list(src, tmp_path)

    warnings = " ".join(
      str(c.args) for c in mock_logging.warning.call_args_list
    )
    self.assertIn("huge.bin", warnings)
    self.assertIn(".kineticignore", warnings)

  def test_no_size_warning_under_threshold(self):
    tmp_path = _make_temp_path(self)
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("code")

    with mock.patch("kinetic.utils.packager.logging") as mock_logging:
      self._zip_and_list(src, tmp_path)
    mock_logging.warning.assert_not_called()

  def test_secret_shaped_files_warn_but_still_ship(self):
    """v4-06: secrets are not silently dropped, but the user is told."""
    tmp_path = _make_temp_path(self)
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("code")
    (src / ".env").write_text("API_KEY=secret")
    (src / ".env.production").write_text("API_KEY=prod-secret")
    (src / "id_rsa").write_text("key")

    with mock.patch("kinetic.utils.packager.logging") as mock_logging:
      names = self._zip_and_list(src, tmp_path)

    self.assertIn(".env", names)
    warnings = " ".join(
      str(c.args) for c in mock_logging.warning.call_args_list
    )
    self.assertIn("Credential-shaped files", warnings)
    self.assertIn(".env", warnings)
    self.assertIn(".env.production", warnings)
    self.assertIn("id_rsa", warnings)

  def test_plan_json_written_to_reserved_path(self):
    tmp_path = _make_temp_path(self)
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("code")
    plan = {
      "package_root": "/proj",
      "entry_rel": "pkg",
      "client_cwd_rel": None,
      "sys_path_rel": ["", "src"],
    }

    out = tmp_path / "context.zip"
    zip_working_dir(str(src), str(out), plan_json=plan)
    with zipfile.ZipFile(str(out)) as zf:
      self.assertEqual(json.loads(zf.read(".kinetic/plan.json")), plan)

  def test_no_plan_entry_when_plan_json_omitted(self):
    tmp_path = _make_temp_path(self)
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("code")

    names = self._zip_and_list(src, tmp_path)
    self.assertNotIn(".kinetic/plan.json", names)

  def test_plan_json_wins_over_existing_file(self):
    tmp_path = _make_temp_path(self)
    src = tmp_path / "src"
    src.mkdir()
    stale = src / ".kinetic"
    stale.mkdir()
    (stale / "plan.json").write_text('{"package_root": "/stale"}')

    out = tmp_path / "context.zip"
    zip_working_dir(str(src), str(out), plan_json={"package_root": "/fresh"})
    with zipfile.ZipFile(str(out)) as zf:
      self.assertEqual(zf.namelist().count(".kinetic/plan.json"), 1)
      plan = json.loads(zf.read(".kinetic/plan.json"))
    self.assertEqual(plan["package_root"], "/fresh")


class TestSavePayload(absltest.TestCase):
  def _save_and_load(
    self,
    tmp_path,
    func,
    args=(),
    kwargs=None,
    env_vars=None,
    volumes=None,
    **extra,
  ):
    """Save a payload and load it back, returning the deserialized dict."""
    if kwargs is None:
      kwargs = {}
    if env_vars is None:
      env_vars = {}
    out = tmp_path / "payload.pkl"
    save_payload(
      func, args, kwargs, env_vars, str(out), volumes=volumes, **extra
    )
    with open(str(out), "rb") as f:
      return cloudpickle.load(f)

  def test_roundtrip_simple_function(self):
    tmp_path = _make_temp_path(self)

    def add(a, b):
      return a + b

    payload = self._save_and_load(
      tmp_path, add, args=(2, 3), env_vars={"KEY": "val"}
    )

    self.assertEqual(payload["func"](2, 3), 5)
    self.assertEqual(payload["args"], (2, 3))
    self.assertEqual(payload["kwargs"], {})
    self.assertEqual(payload["env_vars"], {"KEY": "val"})

  def test_roundtrip_with_kwargs(self):
    tmp_path = _make_temp_path(self)

    def greet(name, greeting="Hello"):
      return f"{greeting}, {name}"

    payload = self._save_and_load(
      tmp_path, greet, args=("World",), kwargs={"greeting": "Hi"}
    )

    result = payload["func"](*payload["args"], **payload["kwargs"])
    self.assertEqual(result, "Hi, World")

  def test_roundtrip_lambda(self):
    tmp_path = _make_temp_path(self)
    payload = self._save_and_load(tmp_path, lambda x: x * 2, args=(5,))

    self.assertEqual(payload["func"](*payload["args"]), 10)

  def test_roundtrip_closure(self):
    tmp_path = _make_temp_path(self)
    multiplier = 7

    def make_closure(x):
      return x * multiplier

    payload = self._save_and_load(tmp_path, make_closure, args=(6,))

    self.assertEqual(payload["func"](*payload["args"]), 42)

  def test_roundtrip_numpy_args(self):
    tmp_path = _make_temp_path(self)

    def dot(a, b):
      return np.dot(a, b)

    arr_a = np.array([1.0, 2.0, 3.0])
    arr_b = np.array([4.0, 5.0, 6.0])

    payload = self._save_and_load(tmp_path, dot, args=(arr_a, arr_b))

    result = payload["func"](*payload["args"])
    self.assertAlmostEqual(result, 32.0)

  def test_roundtrip_complex_args(self):
    tmp_path = _make_temp_path(self)

    def identity(x):
      return x

    complex_arg = {
      "key": [1, 2, 3],
      "nested": {"a": True, "b": None},
      "tuple": (1, "two", 3.0),
    }

    payload = self._save_and_load(tmp_path, identity, args=(complex_arg,))

    self.assertEqual(payload["func"](*payload["args"]), complex_arg)

  def test_volumes_included_in_payload(self):
    tmp_path = _make_temp_path(self)

    def noop():
      pass

    vol_refs = [
      {
        "__data_ref__": True,
        "uri": "gs://b/data-cache/abc",
        "is_dir": True,
        "mount_path": "/data",
      }
    ]
    payload = self._save_and_load(tmp_path, noop, volumes=vol_refs)

    self.assertIn("volumes", payload)
    self.assertEqual(len(payload["volumes"]), 1)
    self.assertEqual(payload["volumes"][0]["mount_path"], "/data")

  def test_no_volumes_key_when_none(self):
    tmp_path = _make_temp_path(self)

    def noop():
      pass

    payload = self._save_and_load(tmp_path, noop)
    self.assertNotIn("volumes", payload)

  def test_client_fingerprint_present(self):
    tmp_path = _make_temp_path(self)

    def noop():
      pass

    payload = self._save_and_load(tmp_path, noop)
    fingerprint = payload["client_fingerprint"]
    self.assertEqual(
      fingerprint["python"],
      ".".join(str(p) for p in sys.version_info[:3]),
    )
    self.assertEqual(fingerprint["cloudpickle"], cloudpickle.__version__)
    self.assertEqual(fingerprint["kinetic"], version.__version__)

  def test_has_data_refs_false_without_refs(self):
    tmp_path = _make_temp_path(self)

    def noop():
      pass

    payload = self._save_and_load(tmp_path, noop, args=([1, {"a": 2}],))
    self.assertFalse(payload["has_data_refs"])

  def test_has_data_refs_true_when_nested_ref(self):
    tmp_path = _make_temp_path(self)

    def noop():
      pass

    ref = {"__data_ref__": True, "uri": "gs://b/p"}
    payload = self._save_and_load(
      tmp_path, noop, kwargs={"cfg": {"inner": [ref]}}
    )
    self.assertTrue(payload["has_data_refs"])

  def test_has_data_refs_survives_circular_args(self):
    tmp_path = _make_temp_path(self)

    def noop():
      pass

    circular = {"a": 1}
    circular["self"] = circular
    payload = self._save_and_load(tmp_path, noop, args=(circular,))
    self.assertFalse(payload["has_data_refs"])

  def test_package_root_and_payload_extra_merged(self):
    tmp_path = _make_temp_path(self)

    def noop():
      pass

    extra = {"entry_rel": "pkg", "sys_path_rel": ["", "src"]}
    payload = self._save_and_load(
      tmp_path,
      noop,
      package_root=str(tmp_path),
      payload_extra=extra,
    )
    self.assertEqual(payload["package_root"], str(tmp_path))
    self.assertEqual(payload["entry_rel"], "pkg")
    self.assertEqual(payload["sys_path_rel"], ["", "src"])

  def test_unpicklable_argument_names_the_offender(self):
    """v2-10: the error must say which argument is at fault."""
    tmp_path = _make_temp_path(self)

    def noop(a, b):
      pass

    with self.assertRaises(ValueError) as cm:
      self._save_and_load(tmp_path, noop, args=(1, threading.Lock()))
    message = str(cm.exception)
    self.assertIn("argument 1", message)
    self.assertIn("kinetic.Data", message)

  def test_unpicklable_keyword_argument_names_the_offender(self):
    tmp_path = _make_temp_path(self)

    def noop(**kw):
      pass

    with self.assertRaises(ValueError) as cm:
      self._save_and_load(tmp_path, noop, kwargs={"stream": (i for i in [1])})
    self.assertIn("keyword argument 'stream'", str(cm.exception))

  def test_unpicklable_function_names_the_function(self):
    tmp_path = _make_temp_path(self)
    lock = threading.Lock()

    def uses_lock():
      return lock

    with self.assertRaises(ValueError) as cm:
      self._save_and_load(tmp_path, uses_lock)
    self.assertIn("the function 'uses_lock'", str(cm.exception))

  def test_failed_payload_file_is_removed(self):
    tmp_path = _make_temp_path(self)
    out = tmp_path / "payload.pkl"

    def noop(a):
      pass

    with self.assertRaises(ValueError):
      save_payload(noop, (threading.Lock(),), {}, {}, str(out))
    self.assertFalse(out.exists())

  def test_payload_size_warning(self):
    tmp_path = _make_temp_path(self)

    def noop(blob):
      pass

    with (
      mock.patch.dict(os.environ, {"KINETIC_PAYLOAD_SIZE_WARN_MB": "0.01"}),
      mock.patch("kinetic.utils.packager.logging") as mock_logging,
    ):
      self._save_and_load(tmp_path, noop, args=(os.urandom(200_000),))

    warnings = " ".join(
      str(c.args) for c in mock_logging.warning.call_args_list
    )
    self.assertIn("kinetic.Data", warnings)

  def test_invalid_size_threshold_is_ignored(self):
    tmp_path = _make_temp_path(self)

    def noop():
      pass

    with mock.patch.dict(
      os.environ, {"KINETIC_PAYLOAD_SIZE_WARN_MB": "not-a-number"}
    ):
      payload = self._save_and_load(tmp_path, noop)
    self.assertIn("func", payload)


class TestPickleByValueModules(absltest.TestCase):
  """T2.3: first-party modules under the package root ship inside the payload."""

  def _make_project(self, name):
    """Create an importable single-module project in a temp dir."""
    tmp_path = _make_temp_path(self)
    root = tmp_path / "proj"
    root.mkdir()
    (root / f"{name}.py").write_text(
      "def helper(x):\n"
      "  return x * 3\n"
      "\n"
      "\n"
      "def entry(x):\n"
      "  return helper(x) + 1\n"
    )
    sys.path.insert(0, str(root))
    self.addCleanup(lambda: sys.path.remove(str(root)))
    self.addCleanup(lambda: sys.modules.pop(name, None))
    return tmp_path, root, importlib.import_module(name)

  def _load_in_subprocess(self, payload_path):
    """Load and call the payload where the project dir is not importable."""
    script = (
      "import cloudpickle\n"
      f"with open({str(payload_path)!r}, 'rb') as f:\n"
      "  payload = cloudpickle.load(f)\n"
      "print(payload['func'](*payload['args']))\n"
    )
    return subprocess.run(
      [sys.executable, "-c", script],
      capture_output=True,
      text=True,
      cwd=tempfile.gettempdir(),
      env={**os.environ, "PYTHONPATH": _REPO_ROOT},
      check=False,
    )

  def test_module_global_helper_ships_by_value(self):
    tmp_path, root, module = self._make_project("kinetic_byvalue_ok")
    out = tmp_path / "payload.pkl"
    save_payload(module.entry, (5,), {}, {}, str(out), package_root=str(root))

    result = self._load_in_subprocess(out)
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(result.stdout.strip(), "16")

  def test_without_package_root_the_module_must_be_importable(self):
    """Control: this is the failure the by-value registration removes."""
    tmp_path, _, module = self._make_project("kinetic_byvalue_ctl")
    out = tmp_path / "payload.pkl"
    save_payload(module.entry, (5,), {}, {}, str(out))

    result = self._load_in_subprocess(out)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("kinetic_byvalue_ctl", result.stderr)

  def test_kinetic_is_never_registered(self):
    registered = []
    with mock.patch.object(
      cloudpickle,
      "register_pickle_by_value",
      side_effect=registered.append,
    ):
      packager._register_by_value_modules(_REPO_ROOT)
    names = {m.__name__ for m in registered}
    self.assertTrue(all(not n.startswith("kinetic") for n in names), names)

  def test_modules_are_unregistered_after_a_failed_dump(self):
    tmp_path, root, module = self._make_project("kinetic_byvalue_err")
    out = tmp_path / "payload.pkl"

    with self.assertRaises(ValueError):
      save_payload(
        module.entry,
        (threading.Lock(),),
        {},
        {},
        str(out),
        package_root=str(root),
      )

    self.assertNotIn(
      module.__name__, cloudpickle.cloudpickle._PICKLE_BY_VALUE_MODULES
    )
    # A second dump without a root must fall back to by-reference pickling.
    save_payload(module.entry, (5,), {}, {}, str(out))
    result = self._load_in_subprocess(out)
    self.assertNotEqual(result.returncode, 0)

  def test_no_package_root_registers_nothing(self):
    self.assertEqual(packager._register_by_value_modules(None), [])


class TestExtractDataRefs(absltest.TestCase):
  def _data(self, name="data.csv"):
    tmp = _make_temp_path(self)
    f = tmp / name
    f.write_text(name)
    return Data(str(f))

  def test_direct_arg(self):
    d = self._data()

    refs = extract_data_refs((d, 42), {})
    self.assertEqual(len(refs), 1)
    self.assertIs(refs[0][0], d)
    self.assertEqual(refs[0][1], ("arg", 0))

  def test_kwarg(self):
    d = self._data()

    refs = extract_data_refs((), {"train_data": d})
    self.assertEqual(len(refs), 1)
    self.assertIs(refs[0][0], d)
    self.assertEqual(refs[0][1], ("kwarg", "train_data"))

  def test_nested_in_list(self):
    d = self._data()

    refs = extract_data_refs(([d, "other"],), {})
    self.assertEqual(len(refs), 1)
    self.assertEqual(refs[0][1], ("arg", 0, 0))

  def test_nested_in_dict(self):
    d = self._data()

    refs = extract_data_refs((), {"config": {"data": d}})
    self.assertEqual(len(refs), 1)
    self.assertEqual(refs[0][1], ("kwarg", "config", "data"))

  def test_multiple_data_objects(self):
    d1 = self._data("a.csv")
    d2 = self._data("b.csv")

    refs = extract_data_refs((d1, d2), {})
    self.assertEqual(len(refs), 2)

  def test_no_data_objects(self):
    refs = extract_data_refs((1, "hello"), {"lr": 0.01})
    self.assertEqual(len(refs), 0)

  def test_reused_data_reported_once(self):
    """v3-F3: one ref per unique Data, so it is hashed and uploaded once."""
    d = self._data()

    refs = extract_data_refs((d, [d], {"k": d}), {"also": d})
    self.assertEqual(len(refs), 1)
    self.assertEqual(refs[0][1], ("arg", 0))

  def test_data_in_set_raises(self):
    d = self._data()

    with self.assertRaises(ValueError) as cm:
      extract_data_refs(({d, "other"},), {})
    self.assertIn("sets or frozensets", str(cm.exception))
    self.assertIn("arg 0", str(cm.exception))

  def test_data_in_frozenset_raises(self):
    d = self._data()

    with self.assertRaises(ValueError) as cm:
      extract_data_refs((), {"cfg": frozenset({d})})
    self.assertIn("kwarg 'cfg'", str(cm.exception))

  def test_data_in_set_raises_even_when_seen_elsewhere(self):
    """The error must fire before the first upload, not after."""
    d = self._data()

    with self.assertRaises(ValueError):
      extract_data_refs((d, {d}), {})

  def test_data_as_dict_key_raises(self):
    """v2-08: a Data key used to be silently ignored."""
    d = self._data()

    with self.assertRaises(ValueError) as cm:
      extract_data_refs(({d: "label"},), {})
    self.assertIn("not supported as dict keys", str(cm.exception))

  def test_data_as_nested_dict_key_raises(self):
    d = self._data()

    with self.assertRaises(ValueError) as cm:
      extract_data_refs((), {"cfg": {"inner": {d: 1}}})
    self.assertIn("kwarg 'cfg'['inner']", str(cm.exception))

  def test_circular_reference_does_not_recurse(self):
    """Circular structures in args should not cause infinite recursion."""
    circular = {"key": "value"}
    circular["self"] = circular

    refs = extract_data_refs((circular,), {})
    self.assertEqual(len(refs), 0)

  def test_circular_reference_still_finds_data(self):
    d = self._data()
    circular = {"data": d}
    circular["self"] = circular

    refs = extract_data_refs((circular,), {})
    self.assertEqual(len(refs), 1)
    self.assertIs(refs[0][0], d)


class TestReplaceDataWithRefs(absltest.TestCase):
  def setUp(self):
    super().setUp()
    tmp = _make_temp_path(self)
    f = tmp / "data.csv"
    f.write_text("data")
    self.data = Data(str(f))
    self.ref = {"__data_ref__": True, "uri": "gs://b/p"}
    self.ref_map = {id(self.data): self.ref}

  def _replace(self, *args, **kwargs):
    return replace_data_with_refs(args, kwargs, self.ref_map)

  def test_replaces_direct_arg(self):
    new_args, _ = self._replace(self.data, 42)
    self.assertEqual(new_args[0], self.ref)
    self.assertEqual(new_args[1], 42)

  def test_replaces_in_list(self):
    new_args, _ = self._replace([self.data, "other"])
    self.assertEqual(new_args[0][0], self.ref)
    self.assertEqual(new_args[0][1], "other")

  def test_replaces_in_kwargs(self):
    _, new_kwargs = replace_data_with_refs(
      (), {"data": self.data, "lr": 0.01}, self.ref_map
    )
    self.assertEqual(new_kwargs["data"], self.ref)
    self.assertEqual(new_kwargs["lr"], 0.01)

  def test_preserves_non_data(self):
    new_args, new_kwargs = replace_data_with_refs(
      (1, "hello", [1, 2]), {"x": 3}, {}
    )
    self.assertEqual(new_args, (1, "hello", [1, 2]))
    self.assertEqual(new_kwargs, {"x": 3})

  def test_untouched_containers_are_the_same_objects(self):
    payload = [1, {"a": 2}]
    new_args, _ = self._replace(payload, self.data)
    self.assertIs(new_args[0], payload)

  def test_typing_namedtuple_preserved(self):
    """v2-04: a NamedTuple must not degrade to a plain tuple."""
    new_args, _ = self._replace(Point(1, self.data))
    self.assertIsInstance(new_args[0], Point)
    self.assertEqual(new_args[0].x, 1)
    self.assertEqual(new_args[0].data, self.ref)

  def test_collections_namedtuple_preserved(self):
    new_args, _ = self._replace(ClassicTuple(self.data, 2))
    self.assertIsInstance(new_args[0], ClassicTuple)
    self.assertEqual(new_args[0].a, self.ref)

  def test_single_field_namedtuple_preserved(self):
    """v2-02: a 1-field NamedTuple used to receive a generator."""
    new_args, _ = self._replace(SingleField(self.data))
    self.assertIsInstance(new_args[0], SingleField)
    self.assertEqual(new_args[0].only, self.ref)

  def test_plain_tuple_stays_tuple(self):
    new_args, _ = self._replace((self.data, 1))
    self.assertIs(type(new_args[0]), tuple)
    self.assertEqual(new_args[0], (self.ref, 1))

  def test_list_subclass_preserved(self):
    new_args, _ = self._replace(ListSubclass([self.data]))
    self.assertIsInstance(new_args[0], ListSubclass)
    self.assertEqual(new_args[0][0], self.ref)

  def test_ordered_dict_preserved(self):
    ordered = collections.OrderedDict([("z", self.data), ("a", 1)])
    new_args, _ = self._replace(ordered)
    self.assertIsInstance(new_args[0], collections.OrderedDict)
    self.assertEqual(list(new_args[0]), ["z", "a"])

  def test_defaultdict_preserves_factory(self):
    dd = collections.defaultdict(int)
    dd["k"] = self.data
    new_args, _ = self._replace(dd)
    self.assertIsInstance(new_args[0], collections.defaultdict)
    self.assertIs(new_args[0].default_factory, int)
    self.assertEqual(new_args[0]["missing"], 0)

  def test_counter_preserved(self):
    counter = collections.Counter()
    counter["k"] = self.data
    new_args, _ = self._replace(counter)
    self.assertIsInstance(new_args[0], collections.Counter)

  def test_unreconstructible_dict_subclass_falls_back_with_warning(self):
    """v2-09: a hostile __init__ must warn, never crash the submission."""
    strict = StrictDict("tag")
    strict["k"] = self.data

    with mock.patch("kinetic.utils.packager.logging") as mock_logging:
      new_args, _ = self._replace(strict)

    self.assertIs(type(new_args[0]), dict)
    self.assertEqual(new_args[0]["k"], self.ref)
    mock_logging.warning.assert_called_once()

  def test_item_swallowing_list_subclass_falls_back_with_warning(self):
    """A subclass that accepts the items without storing them must not
    silently deliver an empty list to the remote function."""
    tagged = TaggedList("tag")
    tagged.append(self.data)

    with mock.patch("kinetic.utils.packager.logging") as mock_logging:
      new_args, _ = self._replace(tagged)

    self.assertIs(type(new_args[0]), list)
    self.assertEqual(new_args[0], [self.ref])
    mock_logging.warning.assert_called_once()

  def test_set_without_data_is_unchanged(self):
    """v3-F7: an unrelated Data must not turn sets into lists."""
    plain = {1, 2, 3}
    frozen = frozenset({4, 5})
    new_args, _ = self._replace(plain, frozen, self.data)
    self.assertIs(new_args[0], plain)
    self.assertIs(new_args[1], frozen)

  def test_data_in_set_raises(self):
    with self.assertRaises(ValueError) as cm:
      self._replace({self.data})
    self.assertIn("sets or frozensets", str(cm.exception))

  def test_data_in_frozenset_raises(self):
    with self.assertRaises(ValueError):
      self._replace(frozenset({self.data}))

  def test_data_as_dict_key_raises(self):
    with self.assertRaises(ValueError) as cm:
      self._replace({self.data: "label"})
    self.assertIn("not supported as dict keys", str(cm.exception))

  def test_duplicate_data_in_one_argument_all_replaced(self):
    """v2-06 / v3-F1: the second occurrence used to leak as a raw Data."""
    new_args, _ = self._replace([self.data, self.data])
    self.assertEqual(new_args[0], [self.ref, self.ref])
    self.assertNotIsInstance(new_args[0][1], Data)

  def test_duplicate_data_across_containers_all_replaced(self):
    new_args, new_kwargs = replace_data_with_refs(
      ([self.data], {"k": self.data}), {"more": (self.data,)}, self.ref_map
    )
    self.assertEqual(new_args[0][0], self.ref)
    self.assertEqual(new_args[1]["k"], self.ref)
    self.assertEqual(new_kwargs["more"][0], self.ref)

  def test_aliasing_preserved(self):
    """v3-F4: shared sub-objects must stay shared after rebuilding."""
    sub = [self.data, 1]
    new_args, _ = self._replace([sub, sub])
    self.assertIs(new_args[0][0], new_args[0][1])

  def test_aliasing_preserved_across_arguments(self):
    sub = {"data": self.data}
    new_args, new_kwargs = replace_data_with_refs(
      (sub,), {"cfg": sub}, self.ref_map
    )
    self.assertIs(new_args[0], new_kwargs["cfg"])

  def test_shared_ref_dict_is_the_same_object(self):
    new_args, _ = self._replace(self.data, [self.data])
    self.assertIs(new_args[0], new_args[1][0])

  def test_deep_nesting(self):
    nested = {"a": [{"b": (self.data,)}]}
    new_args, _ = self._replace(nested)
    self.assertEqual(new_args[0]["a"][0]["b"][0], self.ref)

  def test_self_referential_list_roundtrips(self):
    """v3-F6: a cycle must not blow the stack."""
    circular = [self.data]
    circular.append(circular)

    new_args, _ = self._replace(circular)
    self.assertEqual(new_args[0][0], self.ref)
    self.assertIs(new_args[0][1], new_args[0])

  def test_self_referential_dict_roundtrips(self):
    circular = {"data": self.data}
    circular["self"] = circular

    new_args, _ = self._replace(circular)
    self.assertEqual(new_args[0]["data"], self.ref)
    self.assertIs(new_args[0]["self"], new_args[0])

  def test_cycle_through_tuple_raises_clear_error(self):
    inner = []
    outer = (inner,)
    inner.append(outer)

    with self.assertRaises(ValueError) as cm:
      self._replace(outer, self.data)
    self.assertIn("self-referential", str(cm.exception))

  def test_circular_reference_does_not_recurse(self):
    """Circular structures should not cause infinite recursion."""
    circular = [1, 2]
    circular.append(circular)

    new_args, _ = replace_data_with_refs((circular,), {}, {})
    self.assertIsInstance(new_args[0], list)

  def test_replaced_structure_pickles(self):
    new_args, new_kwargs = replace_data_with_refs(
      (Point(1, self.data), collections.OrderedDict(k=self.data)),
      {"items": ListSubclass([self.data])},
      self.ref_map,
    )
    restored_args, restored_kwargs = cloudpickle.loads(
      cloudpickle.dumps((new_args, new_kwargs))
    )
    self.assertIsInstance(restored_args[0], Point)
    self.assertIsInstance(restored_args[1], collections.OrderedDict)
    self.assertIsInstance(restored_kwargs["items"], ListSubclass)


class TestGitIntegration(absltest.TestCase):
  """Tests for git ls-files integration."""

  def _make_temp_path(self):
    td = tempfile.TemporaryDirectory()
    self.addCleanup(td.cleanup)
    return pathlib.Path(td.name)

  def test_list_git_files_in_repo(self):
    """Test _list_git_files works in a git repository."""
    import subprocess

    from kinetic.utils.packager import _list_git_files

    tmp_path = self._make_temp_path()
    src = tmp_path / "repo"
    src.mkdir()

    # Initialize git repo
    subprocess.run(["git", "-C", str(src), "init"], check=True, capture_output=True)
    subprocess.run(
      ["git", "-C", str(src), "config", "user.email", "test@example.com"],
      check=True,
      capture_output=True,
    )
    subprocess.run(
      ["git", "-C", str(src), "config", "user.name", "Test User"],
      check=True,
      capture_output=True,
    )

    # Add files
    (src / "file1.py").write_text("code")
    (src / "file2.txt").write_text("data")
    (src / ".gitignore").write_text("*.pyc\n")

    subprocess.run(
      ["git", "-C", str(src), "add", "."], check=True, capture_output=True
    )

    files = _list_git_files(str(src))
    self.assertIsNotNone(files)
    self.assertIn("file1.py", files)
    self.assertIn("file2.txt", files)
    self.assertIn(".gitignore", files)

  def test_list_git_files_not_in_repo(self):
    """Test _list_git_files returns None when not in a git repository."""
    from kinetic.utils.packager import _list_git_files

    tmp_path = self._make_temp_path()
    src = tmp_path / "not_repo"
    src.mkdir()
    (src / "file.py").write_text("code")

    files = _list_git_files(str(src))
    self.assertIsNone(files)

  def test_path_is_excluded(self):
    """Test _path_is_excluded checks exclude paths correctly."""
    from kinetic.utils.packager import _path_is_excluded

    exclude_paths = {"/tmp/data", "/tmp/cache"}

    self.assertTrue(_path_is_excluded("/tmp/data/file.txt", exclude_paths))
    self.assertTrue(_path_is_excluded("/tmp/data", exclude_paths))
    self.assertTrue(_path_is_excluded("/tmp/cache/subdir/file.txt", exclude_paths))
    self.assertFalse(_path_is_excluded("/tmp/other/file.txt", exclude_paths))
    self.assertFalse(_path_is_excluded("/tmp/datafile.txt", exclude_paths))


if __name__ == "__main__":
  absltest.main()
