"""Tests for kinetic.utils.packager — zip and payload serialization."""

import os
import pathlib
import tempfile
import zipfile

import cloudpickle
import numpy as np
from absl.testing import absltest

from kinetic.data import Data
from kinetic.utils.packager import (
  extract_data_refs,
  replace_data_with_refs,
  save_payload,
  zip_working_dir,
)


def _make_temp_path(test_case):
  """Create a temp directory that is cleaned up after the test."""
  td = tempfile.TemporaryDirectory()
  test_case.addCleanup(td.cleanup)
  return pathlib.Path(td.name)


class TestZipWorkingDir(absltest.TestCase):
  def _zip_and_list(self, src, tmp_path, exclude_paths=None):
    """Zip src directory and return the set of archive member names."""
    out = tmp_path / "context.zip"
    zip_working_dir(str(src), str(out), exclude_paths=exclude_paths)
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


class TestSavePayload(absltest.TestCase):
  def _save_and_load(
    self,
    tmp_path,
    func,
    args=(),
    kwargs=None,
    env_vars=None,
    volumes=None,
  ):
    """Save a payload and load it back, returning the deserialized dict."""
    if kwargs is None:
      kwargs = {}
    if env_vars is None:
      env_vars = {}
    out = tmp_path / "payload.pkl"
    save_payload(func, args, kwargs, env_vars, str(out), volumes=volumes)
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
        "gcs_uri": "gs://b/data-cache/abc",
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


class TestExtractDataRefs(absltest.TestCase):
  def test_direct_arg(self):
    tmp = _make_temp_path(self)
    f = tmp / "data.csv"
    f.write_text("data")
    d = Data(str(f))

    refs = extract_data_refs((d, 42), {})
    self.assertEqual(len(refs), 1)
    self.assertIs(refs[0][0], d)
    self.assertEqual(refs[0][1], ("arg", 0))

  def test_kwarg(self):
    tmp = _make_temp_path(self)
    f = tmp / "data.csv"
    f.write_text("data")
    d = Data(str(f))

    refs = extract_data_refs((), {"train_data": d})
    self.assertEqual(len(refs), 1)
    self.assertIs(refs[0][0], d)
    self.assertEqual(refs[0][1], ("kwarg", "train_data"))

  def test_nested_in_list(self):
    tmp = _make_temp_path(self)
    f = tmp / "data.csv"
    f.write_text("data")
    d = Data(str(f))

    refs = extract_data_refs(([d, "other"],), {})
    self.assertEqual(len(refs), 1)
    self.assertEqual(refs[0][1], ("arg", 0, 0))

  def test_nested_in_dict(self):
    tmp = _make_temp_path(self)
    f = tmp / "data.csv"
    f.write_text("data")
    d = Data(str(f))

    refs = extract_data_refs((), {"config": {"data": d}})
    self.assertEqual(len(refs), 1)
    self.assertEqual(refs[0][1], ("kwarg", "config", "data"))

  def test_multiple_data_objects(self):
    tmp = _make_temp_path(self)
    f1 = tmp / "a.csv"
    f1.write_text("a")
    f2 = tmp / "b.csv"
    f2.write_text("b")
    d1 = Data(str(f1))
    d2 = Data(str(f2))

    refs = extract_data_refs((d1, d2), {})
    self.assertEqual(len(refs), 2)

  def test_no_data_objects(self):
    refs = extract_data_refs((1, "hello"), {"lr": 0.01})
    self.assertEqual(len(refs), 0)

  def test_nested_in_set(self):
    tmp = _make_temp_path(self)
    f = tmp / "data.csv"
    f.write_text("data")
    d = Data(str(f))

    refs = extract_data_refs(({d, "other"},), {})
    self.assertEqual(len(refs), 1)
    self.assertIs(refs[0][0], d)

  def test_nested_in_frozenset(self):
    tmp = _make_temp_path(self)
    f = tmp / "data.csv"
    f.write_text("data")
    d = Data(str(f))

    refs = extract_data_refs((frozenset({d}),), {})
    self.assertEqual(len(refs), 1)
    self.assertIs(refs[0][0], d)

  def test_circular_reference_does_not_recurse(self):
    """Circular structures in args should not cause infinite recursion."""
    circular = {"key": "value"}
    circular["self"] = circular

    refs = extract_data_refs((circular,), {})
    self.assertEqual(len(refs), 0)


class TestReplaceDataWithRefs(absltest.TestCase):
  def test_replaces_direct_arg(self):
    tmp = _make_temp_path(self)
    f = tmp / "data.csv"
    f.write_text("data")
    d = Data(str(f))
    ref = {"__data_ref__": True, "gcs_uri": "gs://b/p"}
    ref_map = {id(d): ref}

    new_args, new_kwargs = replace_data_with_refs((d, 42), {}, ref_map)
    self.assertEqual(new_args[0], ref)
    self.assertEqual(new_args[1], 42)

  def test_replaces_in_list(self):
    tmp = _make_temp_path(self)
    f = tmp / "data.csv"
    f.write_text("data")
    d = Data(str(f))
    ref = {"__data_ref__": True, "gcs_uri": "gs://b/p"}
    ref_map = {id(d): ref}

    new_args, _ = replace_data_with_refs(([d, "other"],), {}, ref_map)
    self.assertEqual(new_args[0][0], ref)
    self.assertEqual(new_args[0][1], "other")

  def test_replaces_in_kwargs(self):
    tmp = _make_temp_path(self)
    f = tmp / "data.csv"
    f.write_text("data")
    d = Data(str(f))
    ref = {"__data_ref__": True, "gcs_uri": "gs://b/p"}
    ref_map = {id(d): ref}

    _, new_kwargs = replace_data_with_refs((), {"data": d, "lr": 0.01}, ref_map)
    self.assertEqual(new_kwargs["data"], ref)
    self.assertEqual(new_kwargs["lr"], 0.01)

  def test_preserves_non_data(self):
    new_args, new_kwargs = replace_data_with_refs(
      (1, "hello", [1, 2]), {"x": 3}, {}
    )
    self.assertEqual(new_args, (1, "hello", [1, 2]))
    self.assertEqual(new_kwargs, {"x": 3})

  def test_replaces_in_set(self):
    tmp = _make_temp_path(self)
    f = tmp / "data.csv"
    f.write_text("data")
    d = Data(str(f))
    ref = {"__data_ref__": True, "gcs_uri": "gs://b/p"}
    ref_map = {id(d): ref}

    new_args, _ = replace_data_with_refs(({d},), {}, ref_map)
    self.assertIsInstance(new_args[0], list)
    self.assertIn(ref, new_args[0])

  def test_replaces_in_frozenset(self):
    tmp = _make_temp_path(self)
    f = tmp / "data.csv"
    f.write_text("data")
    d = Data(str(f))
    ref = {"__data_ref__": True, "gcs_uri": "gs://b/p"}
    ref_map = {id(d): ref}

    new_args, _ = replace_data_with_refs((frozenset({d}),), {}, ref_map)
    self.assertIsInstance(new_args[0], list)
    self.assertIn(ref, new_args[0])

  def test_circular_reference_does_not_recurse(self):
    """Circular structures should not cause infinite recursion."""
    circular = [1, 2]
    circular.append(circular)

    new_args, _ = replace_data_with_refs((circular,), {}, {})
    self.assertIsInstance(new_args[0], list)


if __name__ == "__main__":
  absltest.main()
