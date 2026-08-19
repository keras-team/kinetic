"""Tests for kinetic.runner.remote_runner — helpers and execution.

Cloud Storage transport is exercised for real against a fake-gcs-server
emulator (see ``kinetic.utils.fake_gcs_fixture``); the remaining patches
are fault injection and instrumentation, never transport replacement.
"""

import collections
import contextlib
import hashlib
import json
import os
import pathlib
import pickle
import stat
import subprocess
import sys
import tempfile
import threading
import typing
import zipfile
from unittest import mock
from unittest.mock import MagicMock

import cloudpickle
from absl.testing import absltest, parameterized
from google.cloud import exceptions as cloud_exceptions
from google.cloud import storage
from google.cloud.storage import transfer_manager

from kinetic.runner import remote_runner
from kinetic.runner.remote_runner import (
  _ZIP_CREATE_SYSTEM_UNIX,
  _apply_workspace_plan,
  _contains_data_ref,
  _download_data,
  _download_from_gcs,
  _download_hf_data,
  _exit_process,
  _extract_context,
  _install_requirements,
  _payload_has_data_refs,
  _preimport_hf_dependencies,
  _upload_to_gcs,
  _verify_sha256,
  _wait_for_leader_ready_sentinel,
  _warn_on_fingerprint_skew,
  main,
  resolve_data_refs,
  resolve_volumes,
)
from kinetic.utils.fake_gcs_fixture import (
  TEST_PROJECT,
  clear_kinetic_client_cache,
  shared_server,
)

# Mounted ref: resolves to its mount path without touching GCS.
_MOUNTED_REF = {
  "__data_ref__": True,
  "uri": "gs://b/p",
  "is_dir": True,
  "mount_path": "/mnt/data",
}

# Point/OneField use collections.namedtuple on purpose: the typing.NamedTuple
# flavor (TypedPoint below) is rebuilt through a different code path, so both
# need coverage.
Point = collections.namedtuple("Point", ["x", "y"])  # noqa: PYI024
OneField = collections.namedtuple("OneField", ["value"])  # noqa: PYI024


class TypedPoint(typing.NamedTuple):
  x: object
  y: object


class MyList(list):
  pass


class Batch(list):
  """List subclass whose constructor is not a plain iterable wrapper."""

  def __init__(self, items, tag="t"):
    super().__init__(items)
    self.tag = tag


class StrictList(list):
  """List subclass whose constructor swallows the items it is given."""

  def __init__(self, tag):
    super().__init__()
    self.tag = tag


class MyDict(dict):
  pass


class StrictDict(dict):
  """Dict subclass that cannot be rebuilt from a mapping."""

  def __init__(self, tag):
    super().__init__()
    self.tag = tag


class HashableRef(dict):
  """A data-ref dict that can be used as a mapping key."""

  def __hash__(self):
    return 1


class UnserializableResult:
  """A result whose ``__reduce__`` fails with a non-pickling error."""

  def __reduce__(self):
    raise RuntimeError("reduce exploded")

  def __repr__(self):
    return "<UnserializableResult tag=abc>"


def _make_temp_path(test_case):
  """Create a temp directory that is cleaned up after the test.

  The path is resolved so that ``os.getcwd()`` inside it compares equal
  to it on platforms where the temp root is a symlink (macOS).
  """
  td = tempfile.TemporaryDirectory()
  test_case.addCleanup(td.cleanup)
  return pathlib.Path(os.path.realpath(td.name))


def _data_ref(uri="gs://b/p", is_dir=True, **extra):
  """Build a data-ref dict the way the client packager writes one."""
  ref = {"__data_ref__": True, "uri": uri, "is_dir": is_dir}
  ref.update(extra)
  return ref


def _defaultdict_with_ref():
  """A ``defaultdict`` holding a mounted ref under ``"a"``."""
  mapping = collections.defaultdict(list)
  mapping["a"] = _MOUNTED_REF
  return mapping


class _EmulatorTestCase(parameterized.TestCase):
  """Base for tests that talk to the shared fake-gcs-server."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # Raises SkipTest when the fake-gcs-server binary is unavailable.
    cls.server = shared_server()

  def setUp(self):
    super().setUp()
    # Other test modules patch storage.Client and can leave a stale mock
    # in kinetic's per-project client cache; drop it before every test
    # (per-test, because xdist can interleave foreign tests between two
    # methods of one class).
    clear_kinetic_client_cache()

  def real_client(self):
    """A real google-cloud-storage client aimed at the emulator."""
    return storage.Client(project=TEST_PROJECT)

  def seeded_data_bucket(self, blob_names, content=b"data"):
    """A fresh bucket holding *blob_names* (directory markers stay empty)."""
    bucket = self.server.make_bucket(suffix="data")
    for name in blob_names:
      self.server.write_blob(
        bucket, name, b"" if name.endswith("/") else content
      )
    return bucket


def _rendered_warnings(mock_warning):
  """The messages a patched ``logging.warning`` received, formatted."""
  messages = []
  for call in mock_warning.call_args_list:
    template, args = call.args[0], call.args[1:]
    messages.append(template % args if args else template)
  return messages


def _assert_warned(test_case, mock_warning, *fragments):
  """Assert every fragment shows up in some rendered warning."""
  messages = _rendered_warnings(mock_warning)
  for fragment in fragments:
    test_case.assertTrue(
      any(fragment in message for message in messages),
      f"{fragment!r} not in warnings: {messages}",
    )


def _sha256(path):
  """The SHA-256 hex digest of a file."""
  with open(path, "rb") as f:
    return hashlib.sha256(f.read()).hexdigest()


def _make_context_zip(path, entries=None, plan_json=None):
  """Write a context.zip containing *entries* and an optional plan."""
  entries = {"dummy.py": "x = 1"} if entries is None else entries
  with zipfile.ZipFile(path, "w") as zf:
    for name, content in entries.items():
      zf.writestr(name, content)
    if plan_json is not None:
      zf.writestr(".kinetic/plan.json", json.dumps(plan_json))


def _make_mode_zip(path, name, mode, create_system):
  """Write a one-entry archive carrying *mode* in its external attrs.

  Args:
      path: Destination archive path.
      name: Archive member name.
      mode: POSIX mode to store in the high bits of ``external_attr``.
      create_system: ``ZipInfo.create_system`` value to record.
  """
  with zipfile.ZipFile(path, "w") as zf:
    info = zipfile.ZipInfo(name)
    info.create_system = create_system
    info.external_attr = mode << 16
    if create_system != _ZIP_CREATE_SYSTEM_UNIX:
      # Real non-Unix archives carry MS-DOS attribute flags down here.
      info.external_attr |= 0x20
    zf.writestr(info, "#!/bin/sh\necho hi\n")


def _basic_payload(func, args=(), kwargs=None, **extra):
  """Build a payload dict with the additive keys under test."""
  payload = {
    "func": func,
    "args": args,
    "kwargs": kwargs or {},
    "env_vars": {},
  }
  payload.update(extra)
  return payload


def _ghost_payload(test_case, fingerprint):
  """Pickle bytes referencing a module that will not exist at load time."""
  import importlib

  tmp = _make_temp_path(test_case)
  (tmp / "kinetic_ghost_mod.py").write_text("class Ghost:\n  pass\n")
  sys.path.insert(0, str(tmp))
  try:
    ghost = importlib.import_module("kinetic_ghost_mod")
    payload = {
      "func": ghost.Ghost,
      "args": (),
      "kwargs": {},
      "env_vars": {},
    }
    if fingerprint is not None:
      payload["client_fingerprint"] = fingerprint
    return pickle.dumps(payload)
  finally:
    sys.path.remove(str(tmp))
    sys.modules.pop("kinetic_ghost_mod", None)


class _SeededArtifacts(typing.NamedTuple):
  """Emulator URIs and local source paths for one runner invocation."""

  bucket: str
  context_uri: str
  payload_uri: str
  result_uri: str
  context_path: str
  payload_path: str

  @property
  def default_argv(self):
    """The legacy positional argument vector for these artifacts."""
    return [
      "remote_runner.py",
      self.context_uri,
      self.payload_uri,
      self.result_uri,
    ]


def _run_runner(
  test_case,
  payload=None,
  payload_bytes=None,
  zip_entries=None,
  plan_json=None,
  context_bytes=None,
  argv=None,
  patches=(),
):
  """Run ``main()`` against artifacts seeded into the real emulator.

  The context and payload are uploaded to a fresh fake-gcs-server
  bucket, ``main()`` downloads and uploads through its real storage
  client, and the result is read back from the emulator.

  Args:
      test_case: The running test, used for temp dirs and cleanups.
      payload: Payload dict to cloudpickle into ``payload.pkl``.
      payload_bytes: Raw bytes to use as ``payload.pkl`` instead.
      zip_entries: ``{archive path: text}`` for the context archive.
      plan_json: Optional ``.kinetic/plan.json`` content.
      context_bytes: Raw bytes to use as ``context.zip`` instead.
      argv: Argument vector, or a callable taking a ``_SeededArtifacts``
          and returning one.
      patches: Extra unstarted patches entered around ``main()``.

  Returns:
      ``(exit_code, result_payload_or_None)`` — the result unpickled
      from the emulator, or None when no result was uploaded.  The
      seeded ``_SeededArtifacts`` is left on ``test_case.artifacts`` so
      tests can read the other blobs in the job prefix.
  """
  server = shared_server()
  tmp_path = _make_temp_path(test_case)
  test_case.addCleanup(setattr, sys, "path", sys.path[:])
  test_case.addCleanup(os.chdir, os.getcwd())

  context_zip = tmp_path / "context.zip"
  if context_bytes is not None:
    context_zip.write_bytes(context_bytes)
  else:
    _make_context_zip(context_zip, zip_entries, plan_json)

  payload_pkl = tmp_path / "payload.pkl"
  if payload_bytes is not None:
    payload_pkl.write_bytes(payload_bytes)
  else:
    with open(payload_pkl, "wb") as f:
      cloudpickle.dump(payload, f)

  bucket = server.make_bucket(suffix="runner")
  server.write_blob(bucket, "job/context.zip", context_zip.read_bytes())
  server.write_blob(bucket, "job/payload.pkl", payload_pkl.read_bytes())
  artifacts = _SeededArtifacts(
    bucket=bucket,
    context_uri=f"gs://{bucket}/job/context.zip",
    payload_uri=f"gs://{bucket}/job/payload.pkl",
    result_uri=f"gs://{bucket}/job/result.pkl",
    context_path=str(context_zip),
    payload_path=str(payload_pkl),
  )
  test_case.artifacts = artifacts

  if argv is None:
    argv = artifacts.default_argv
  elif callable(argv):
    argv = argv(artifacts)

  with contextlib.ExitStack() as stack:
    stack.enter_context(mock.patch("sys.argv", argv))
    for patch in patches:
      stack.enter_context(patch)

    with test_case.assertRaises(SystemExit) as cm:
      main()

  result_payload = None
  result_bytes = server.read_blob(bucket, "job/result.pkl")
  if result_bytes is not None:
    result_payload = cloudpickle.loads(result_bytes)

  return cm.exception.code, result_payload


class _RunnerTestCase(_EmulatorTestCase):
  """Base class for tests that drive ``main()`` end to end."""

  def run_runner(self, **kwargs):
    """Run ``main()`` against seeded emulator artifacts; see ``_run_runner``."""
    return _run_runner(self, **kwargs)

  def job_blob(self, name):
    """Unpickle a blob from the job prefix, or None when it is absent."""
    raw = self.server.read_blob(self.artifacts.bucket, f"job/{name}")
    return None if raw is None else cloudpickle.loads(raw)

  def job_blob_names(self):
    """The blob names present under the seeded job prefix."""
    return sorted(
      name
      for name in self.server.list_blob_names(self.artifacts.bucket)
      if name.startswith("job/")
    )


# Payload entry points shared by parameterized cases. They must be
# module level so the parameter lists can reference them.
def _noop():
  return None


def _identity(value):
  return value


def _add(a, b):
  return a + b


def _type_name(value):
  return type(value).__name__


def _read_env(name):
  return os.environ.get(name)


def _exit_with_code(code):
  sys.exit(code)


class TestDownloadFromGcs(_EmulatorTestCase):
  @parameterized.named_parameters(
    ("simple", "path/to/file.pkl"),
    ("nested", "a/b/c/deep/file.zip"),
  )
  def test_downloads_the_addressed_blob(self, blob_path):
    bucket = self.server.make_bucket()
    self.server.write_blob(bucket, blob_path, b"artifact bytes")
    local = _make_temp_path(self) / "local.bin"

    _download_from_gcs(
      self.real_client(), f"gs://{bucket}/{blob_path}", str(local)
    )

    self.assertEqual(local.read_bytes(), b"artifact bytes")

  def test_missing_blob_raises_not_found(self):
    bucket = self.server.make_bucket()
    local = _make_temp_path(self) / "local.bin"

    with self.assertRaises(cloud_exceptions.NotFound):
      _download_from_gcs(
        self.real_client(), f"gs://{bucket}/absent.pkl", str(local)
      )


class TestUploadToGcs(_EmulatorTestCase):
  def test_uploads_to_the_addressed_blob(self):
    bucket = self.server.make_bucket()
    local = _make_temp_path(self) / "result.pkl"
    local.write_bytes(b"result bytes")

    _upload_to_gcs(
      self.real_client(), str(local), f"gs://{bucket}/results/result.pkl"
    )

    self.assertEqual(
      self.server.read_blob(bucket, "results/result.pkl"), b"result bytes"
    )


class TestDownloadData(_EmulatorTestCase):
  def setUp(self):
    super().setUp()
    # A spy, not a stub: the real transfer_manager still runs against
    # the emulator; the wrapper only records the batching contract.
    self.spy_download = self.enterContext(
      mock.patch(
        "kinetic.runner.remote_runner.transfer_manager.download_many_to_path",
        wraps=transfer_manager.download_many_to_path,
      )
    )

  @parameterized.named_parameters(
    (
      "skips_directory_entries",
      ["prefix/hash/", "prefix/hash/train.csv"],
      ["train.csv"],
    ),
    ("keeps_subdirectories", ["prefix/hash/sub/deep.csv"], ["sub/deep.csv"]),
  )
  def test_downloads_relative_blob_names(self, blob_names, expected):
    bucket = self.seeded_data_bucket(blob_names)
    target = str(_make_temp_path(self) / "output")

    _download_data(
      _data_ref(f"gs://{bucket}/prefix/hash"),
      target,
      self.real_client(),
    )

    self.spy_download.assert_called_once()
    self.assertEqual(self.spy_download.call_args[0][1], expected)
    kwargs = self.spy_download.call_args.kwargs
    self.assertEqual(kwargs["destination_directory"], target)
    self.assertEqual(kwargs["blob_name_prefix"], "prefix/hash/")
    # A failed blob must raise, not come back in the results list — the
    # emulator's downloads all succeed, so only this assertion guards
    # against partially downloaded data reaching the user function.
    self.assertTrue(kwargs["raise_exception"])
    for rel_path in expected:
      downloaded = pathlib.Path(target, rel_path)
      self.assertEqual(downloaded.read_bytes(), b"data")

  def test_large_listing_downloads_in_batches(self):
    batch_size = 5
    num_blobs = batch_size + 2
    bucket = self.seeded_data_bucket(
      [f"prefix/hash/file_{i}.csv" for i in range(num_blobs)]
    )
    target = str(_make_temp_path(self) / "output")

    with mock.patch(
      "kinetic.runner.remote_runner._DOWNLOAD_BATCH_SIZE", batch_size
    ):
      _download_data(
        _data_ref(f"gs://{bucket}/prefix/hash"), target, self.real_client()
      )

    self.assertEqual(self.spy_download.call_count, 2)
    first_batch = self.spy_download.call_args_list[0][0][1]
    second_batch = self.spy_download.call_args_list[1][0][1]
    self.assertEqual(len(first_batch), batch_size)
    self.assertEqual(len(second_batch), 2)
    for i in range(num_blobs):
      self.assertTrue(pathlib.Path(target, f"file_{i}.csv").exists())

  def test_empty_listing_is_noop(self):
    bucket = self.server.make_bucket()
    target = str(_make_temp_path(self) / "output")

    _download_data(
      _data_ref(f"gs://{bucket}/prefix/hash"), target, self.real_client()
    )

    self.spy_download.assert_not_called()


class TestDownloadSingleObject(_EmulatorTestCase):
  """A ``is_dir=False`` ref, addressing either an object or a hash dir.

  ``Data("gs://bucket/dir/file.h5")`` names the object itself, while an
  uploaded local file gets the content-hash directory holding it. Both
  arrive as ``is_dir=False`` and must land as one file in the target.
  """

  # Siblings prove the download is addressed, not "whatever is nearby".
  SIBLINGS = [
    "datasets/weights.h5",
    "datasets/notes.txt",
    "datasets/other.h5",
    "datasets/nested/deep.bin",
  ]

  def test_gcs_native_object_downloads_beside_its_siblings(self):
    bucket = self.seeded_data_bucket(self.SIBLINGS, content=b"weights")
    target = _make_temp_path(self) / "output"

    _download_data(
      _data_ref(f"gs://{bucket}/datasets/weights.h5", is_dir=False),
      str(target),
      self.real_client(),
    )

    self.assertEqual(sorted(os.listdir(target)), ["weights.h5"])
    self.assertEqual((target / "weights.h5").read_bytes(), b"weights")

  def test_object_at_the_bucket_root_downloads(self):
    bucket = self.seeded_data_bucket(["weights.h5", "notes.txt"])
    target = _make_temp_path(self) / "output"

    _download_data(
      _data_ref(f"gs://{bucket}/weights.h5", is_dir=False),
      str(target),
      self.real_client(),
    )

    self.assertEqual(sorted(os.listdir(target)), ["weights.h5"])

  def test_uploaded_file_still_resolves_through_its_hash_dir(self):
    """The client's own single-file URI names a directory, not an object.

    The exact listing is the assertion that matters: the direct object
    attempt 404s here, and ``download_to_filename`` opens the
    destination before it learns that. Were the empty file to survive,
    ``resolve_data_refs`` would see two entries and hand back the
    directory instead of the file.
    """
    bucket = self.seeded_data_bucket(
      ["ns/data-cache/abc123/config.json", "ns/data-cache/def456/other.json"],
      content=b"{}",
    )
    target = _make_temp_path(self) / "output"

    _download_data(
      _data_ref(f"gs://{bucket}/ns/data-cache/abc123", is_dir=False),
      str(target),
      self.real_client(),
    )

    self.assertEqual(sorted(os.listdir(target)), ["config.json"])

  def test_uploaded_file_ref_resolves_to_the_file_after_the_fallback(self):
    """End to end: the miss-then-list path still yields a file path."""
    tmp = _make_temp_path(self)
    bucket = self.seeded_data_bucket(
      ["ns/data-cache/abc123/config.json"], content=b"{}"
    )
    ref = _data_ref(
      f"gs://{bucket}/ns/data-cache/abc123", is_dir=False, mount_path=None
    )

    args, _ = resolve_data_refs(
      (ref,), {}, self.real_client(), str(tmp / "data")
    )

    self.assertTrue(os.path.isfile(args[0]))
    self.assertEqual(os.path.basename(args[0]), "config.json")

  def test_missing_object_raises_with_the_uri(self):
    bucket = self.seeded_data_bucket(self.SIBLINGS)
    uri = f"gs://{bucket}/datasets/absent.h5"
    target = str(_make_temp_path(self) / "output")

    with self.assertRaises(FileNotFoundError) as cm:
      _download_data(_data_ref(uri, is_dir=False), target, self.real_client())

    self.assertIn(uri, str(cm.exception))

  def test_directory_ref_without_trailing_slash_still_lists(self):
    """A prefix that only *looks* like a file falls back to the listing."""
    bucket = self.seeded_data_bucket(
      ["datasets/train.v2/a.csv", "datasets/train.v2/b.csv"]
    )
    target = _make_temp_path(self) / "output"

    _download_data(
      _data_ref(f"gs://{bucket}/datasets/train.v2", is_dir=False),
      str(target),
      self.real_client(),
    )

    self.assertEqual(sorted(os.listdir(target)), ["a.csv", "b.csv"])


class TestDownloadHfData(parameterized.TestCase):
  @parameterized.named_parameters(
    ("trusted", "hf://imdb?split=train", True),
    (
      "query_param_cannot_grant_trust",
      "hf://imdb?trust_remote_code=true",
      False,
    ),
  )
  def test_trust_remote_code_comes_from_the_argument(self, uri, trust):
    mock_datasets = MagicMock()

    with mock.patch.dict("sys.modules", {"datasets": mock_datasets}):
      _download_hf_data(uri, "/tmp/target", trust_remote_code=trust)

    mock_datasets.load_dataset.assert_called_once()
    kwargs = mock_datasets.load_dataset.call_args.kwargs
    self.assertEqual(kwargs["trust_remote_code"], trust)


class TestResolveDataRefs(_EmulatorTestCase):
  # Mounted and fuse refs never touch GCS, so those cases pass
  # ``client=None`` to prove it; download cases get a real client and a
  # seeded emulator bucket.

  def _seeded_ref(self, blob_names, **ref_kwargs):
    """A data ref addressing a fresh bucket seeded with *blob_names*."""
    bucket = self.seeded_data_bucket(blob_names)
    prefix = blob_names[0].rsplit("/", 1)[0]
    return _data_ref(f"gs://{bucket}/{prefix}", **ref_kwargs)

  def test_replaces_ref_with_path(self):
    tmp = _make_temp_path(self)
    ref = self._seeded_ref(["cache/hash/part.txt"], mount_path=None)

    args, _ = resolve_data_refs(
      (ref, 42),
      {},
      self.real_client(),
      str(tmp / "data"),
    )

    self.assertIsInstance(args[0], str)
    self.assertEqual(pathlib.Path(args[0], "part.txt").read_bytes(), b"data")
    self.assertEqual(args[1], 42)

  def test_nested_refs_in_list(self):
    tmp = _make_temp_path(self)
    ref = self._seeded_ref(["cache/hash/part.txt"], mount_path=None)

    args, _ = resolve_data_refs(
      ([ref, "other"],),
      {},
      self.real_client(),
      str(tmp / "data"),
    )

    self.assertIsInstance(args[0][0], str)
    self.assertEqual(args[0][1], "other")

  def test_kwargs_refs_resolved(self):
    tmp = _make_temp_path(self)
    ref = self._seeded_ref(["cache/hash/part.txt"], mount_path=None)

    _, kwargs = resolve_data_refs(
      (),
      {"data": ref, "lr": 0.01},
      self.real_client(),
      str(tmp / "data"),
    )

    self.assertIsInstance(kwargs["data"], str)
    self.assertEqual(kwargs["lr"], 0.01)

  def test_single_file_returns_file_path(self):
    tmp = _make_temp_path(self)
    ref = self._seeded_ref(
      ["prefix/hash/config.json"], is_dir=False, mount_path=None
    )

    args, _ = resolve_data_refs(
      (ref,), {}, self.real_client(), str(tmp / "data")
    )

    self.assertTrue(args[0].endswith("config.json"))

  def test_gcs_native_object_returns_that_objects_path(self):
    """``Data("gs://b/dir/file.h5")`` resolves to the file, not its siblings."""
    tmp = _make_temp_path(self)
    bucket = self.seeded_data_bucket(
      [
        "datasets/weights.h5",
        "datasets/notes.txt",
        "datasets/other.h5",
      ],
      content=b"weights",
    )
    ref = _data_ref(
      f"gs://{bucket}/datasets/weights.h5", is_dir=False, mount_path=None
    )

    args, _ = resolve_data_refs(
      (ref,), {}, self.real_client(), str(tmp / "data")
    )

    self.assertTrue(os.path.isfile(args[0]))
    self.assertEqual(os.path.basename(args[0]), "weights.h5")
    self.assertEqual(pathlib.Path(args[0]).read_bytes(), b"weights")

  def test_duplicate_uri_downloaded_once(self):
    tmp = _make_temp_path(self)
    ref = self._seeded_ref(["cache/hash/part.txt"], mount_path=None)

    with mock.patch(
      "kinetic.runner.remote_runner._download_data", wraps=_download_data
    ) as spy_dl:
      args, kwargs = resolve_data_refs(
        (ref, ref), {"d": ref}, self.real_client(), str(tmp / "data")
      )

    # Downloaded only once despite three references
    spy_dl.assert_called_once()
    # All resolved paths point to the same directory
    self.assertEqual(args[0], args[1])
    self.assertEqual(args[0], kwargs["d"])

  def test_non_ref_dict_preserved(self):
    args, kwargs = resolve_data_refs(
      ({"key": "value"},), {"x": 1}, None, "/tmp/data"
    )

    self.assertEqual(args[0], {"key": "value"})
    self.assertEqual(kwargs["x"], 1)

  def _fuse_mount(self, entries):
    """A stand-in for a GCS FUSE mount holding *entries*."""
    mount_dir = _make_temp_path(self) / "fuse-mount"
    mount_dir.mkdir()
    for entry in entries:
      (mount_dir / entry).write_text(entry)
    return mount_dir

  def test_fuse_single_file_resolves_to_file_path(self):
    """FUSE-mounted single file ref resolves to the actual file, not dir."""
    mount_dir = self._fuse_mount(["config.json"])
    ref = _data_ref(
      "gs://b/path/to/config.json",
      is_dir=False,
      mount_path=str(mount_dir),
      fuse=True,
    )

    args, _ = resolve_data_refs((ref,), {}, None, "/tmp/data")

    self.assertTrue(args[0].endswith("config.json"))
    self.assertFalse(os.path.isdir(args[0]))

  def test_fuse_gcs_native_object_picked_out_of_its_siblings(self):
    """The mounted parent holds unrelated objects; the URI names the one."""
    mount_dir = self._fuse_mount(
      ["notes.txt", "other.h5", "weights.h5", "zzz.bin"]
    )
    ref = _data_ref(
      "gs://b/datasets/weights.h5",
      is_dir=False,
      mount_path=str(mount_dir),
      fuse=True,
    )

    args, _ = resolve_data_refs((ref,), {}, None, "/tmp/data")

    self.assertEqual(args[0], str(mount_dir / "weights.h5"))
    self.assertEqual(pathlib.Path(args[0]).read_text(), "weights.h5")

  def test_fuse_uploaded_file_resolves_through_its_hash_dir(self):
    """An uploaded file's URI names the hash dir, whose lone entry it is."""
    mount_dir = self._fuse_mount(["config.json"])
    ref = _data_ref(
      "gs://b/ns/data-cache/abc123",
      is_dir=False,
      mount_path=str(mount_dir),
      fuse=True,
    )

    args, _ = resolve_data_refs((ref,), {}, None, "/tmp/data")

    self.assertEqual(args[0], str(mount_dir / "config.json"))

  def test_fuse_object_absent_from_a_populated_mount_raises(self):
    """Guessing a sibling would silently feed the function the wrong file."""
    mount_dir = self._fuse_mount(["notes.txt", "other.h5"])
    ref = _data_ref(
      "gs://b/datasets/weights.h5",
      is_dir=False,
      mount_path=str(mount_dir),
      fuse=True,
    )

    with self.assertRaises(FileNotFoundError) as cm:
      resolve_data_refs((ref,), {}, None, "/tmp/data")

    message = str(cm.exception)
    self.assertIn("gs://b/datasets/weights.h5", message)
    self.assertIn("weights.h5", message)

  def test_fuse_empty_mount_falls_back_to_the_mount_path(self):
    """An unmounted/empty path is reported as-is, not as a bogus file."""
    mount_dir = self._fuse_mount([])
    ref = _data_ref(
      "gs://b/datasets/weights.h5",
      is_dir=False,
      mount_path=str(mount_dir),
      fuse=True,
    )

    args, _ = resolve_data_refs((ref,), {}, None, "/tmp/data")

    self.assertEqual(args[0], str(mount_dir))

  def test_fuse_directory_returns_mount_path(self):
    """FUSE-mounted directory ref returns the mount path unchanged."""
    ref = _data_ref(
      "gs://b/data/train/", mount_path="/tmp/fuse-data/0", fuse=True
    )

    args, _ = resolve_data_refs((ref,), {}, None, "/tmp/data")

    self.assertEqual(args[0], "/tmp/fuse-data/0")

  def test_non_fuse_mount_returns_mount_path(self):
    """Non-FUSE mounted ref returns the mount path unchanged."""
    ref = _data_ref(
      "gs://b/cache/hash", is_dir=False, mount_path="/data/config"
    )

    args, _ = resolve_data_refs((ref,), {}, None, "/tmp/data")

    self.assertEqual(args[0], "/data/config")


class TestResolveVolumes(_EmulatorTestCase):
  @parameterized.named_parameters(
    ("single_volume", ["data"], []),
    ("multiple_volumes", ["data1", "data2"], []),
    ("mixed_with_a_fuse_volume", ["downloaded"], ["/kinetic-test-fuse-mount"]),
  )
  def test_non_fuse_volumes_download_to_their_mount_paths(
    self, downloaded, fuse_mounts
  ):
    tmp = _make_temp_path(self)
    bucket = self.seeded_data_bucket(
      [f"{name}/part.txt" for name in downloaded]
    )
    # No "fuse" key at all: the old ref format still downloads.
    refs = [
      _data_ref(f"gs://{bucket}/{name}", mount_path=str(tmp / name))
      for name in downloaded
    ]
    refs += [
      _data_ref(f"gs://{bucket}/fuse/{i}", mount_path=path, fuse=True)
      for i, path in enumerate(fuse_mounts)
    ]

    resolve_volumes(refs, self.real_client())

    for name in downloaded:
      self.assertEqual((tmp / name / "part.txt").read_bytes(), b"data")
    for path in fuse_mounts:
      self.assertFalse(os.path.exists(path))

  def test_fuse_volume_skips_download(self):
    refs = [_data_ref("gs://b/data/", mount_path="/data", fuse=True)]

    with mock.patch(
      "kinetic.runner.remote_runner._download_data", wraps=_download_data
    ) as spy_dl:
      resolve_volumes(refs, None)

    spy_dl.assert_not_called()


class TestMain(_RunnerTestCase):
  @parameterized.named_parameters(
    ("returns_the_value", _add, (2, 3), 5),
    ("passes_arguments_through_untouched", _identity, (42,), 42),
  )
  def test_success_flow(self, func, args, expected):
    exit_code, result = self.run_runner(payload=_basic_payload(func, args=args))

    self.assertEqual(exit_code, 0)
    self.assertTrue(result["success"])
    self.assertEqual(result["result"], expected)
    self.assertIsNone(result["exception"])
    self.assertEqual(result["phase"], "execute")

  def test_function_exception(self):
    def bad_func():
      raise ValueError("test error")

    exit_code, result = self.run_runner(payload=_basic_payload(bad_func))

    self.assertEqual(exit_code, 1)
    self.assertFalse(result["success"])
    self.assertIsInstance(result["exception"], ValueError)
    self.assertIn("test error", str(result["exception"]))
    self.assertIn("ValueError: test error", result["traceback"])

  def test_env_vars_applied(self):
    self.addCleanup(os.environ.pop, "TEST_REMOTE_VAR", None)

    exit_code, result = self.run_runner(
      payload=_basic_payload(
        _read_env,
        args=("TEST_REMOTE_VAR",),
        env_vars={"TEST_REMOTE_VAR": "hello"},
      )
    )

    self.assertEqual(exit_code, 0)
    self.assertEqual(result["result"], "hello")

  def test_data_ref_resolved_before_execution(self):
    """Data refs in args are resolved to local paths."""

    def check_is_string(data_path):
      assert isinstance(data_path, str), f"Expected str, got {type(data_path)}"
      return "resolved"

    bucket = self.seeded_data_bucket(["cache/hash/part.txt"])

    exit_code, result = self.run_runner(
      payload=_basic_payload(
        check_is_string,
        args=(_data_ref(f"gs://{bucket}/cache/hash", mount_path=None),),
      )
    )

    self.assertEqual(exit_code, 0)
    self.assertEqual(result["result"], "resolved")

  def test_volumes_resolved_before_execution(self):
    """Volumes are downloaded to mount paths before function execution."""
    mount_path = str(_make_temp_path(self) / "mounted_data")

    def check_mount(expected_path):
      assert os.path.isdir(expected_path), (
        f"Mount path should exist: {expected_path}"
      )
      return "mounted"

    bucket = self.seeded_data_bucket(["cache/hash/part.txt"])

    exit_code, result = self.run_runner(
      payload=_basic_payload(
        check_mount,
        args=(mount_path,),
        volumes=[_data_ref(f"gs://{bucket}/cache/hash", mount_path=mount_path)],
      )
    )

    self.assertEqual(exit_code, 0)
    self.assertEqual(result["result"], "mounted")

  def test_unpicklable_exception_produces_fallback_result(self):
    """An exception that can't be pickled becomes a RuntimeError."""

    class UnpicklableError(Exception):
      def __reduce__(self):
        raise TypeError("cannot pickle UnpicklableError")

    def raise_unpicklable():
      raise UnpicklableError("boom")

    exit_code, result = self.run_runner(
      payload=_basic_payload(raise_unpicklable)
    )

    self.assertEqual(exit_code, 1)
    self.assertFalse(result["success"])
    self.assertIsInstance(result["exception"], RuntimeError)
    self.assertIn("Result serialization failed", str(result["exception"]))
    self.assertIn("UnpicklableError", result["traceback"])


class TestInstallRequirements(_EmulatorTestCase):
  @contextlib.contextmanager
  def _seeded_install(self, contents, returncode=0, stderr=""):
    """Seed a real requirements blob and fake only ``uv pip install``.

    Yields ``(temp_dir, requirements_uri, mock_subprocess_run)``.
    """
    tmp = _make_temp_path(self)
    bucket = self.server.make_bucket(suffix="reqs")
    self.server.write_blob(bucket, "requirements.txt", contents)
    uri = f"gs://{bucket}/requirements.txt"

    with mock.patch("kinetic.runner.remote_runner.subprocess.run") as mock_run:
      mock_run.return_value = MagicMock(
        returncode=returncode, stderr=stderr, stdout=""
      )
      yield str(tmp), uri, mock_run

  def test_successful_install(self):
    with self._seeded_install("numpy==1.26\n") as (temp_dir, uri, mock_run):
      _install_requirements(self.real_client(), uri, temp_dir)

    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    self.assertEqual(args[:4], ["uv", "pip", "install", "--system"])
    self.assertTrue(args[-1].endswith("user_requirements.txt"))

  def test_failed_install_raises(self):
    with (
      self._seeded_install(
        "nonexistent-package\n", returncode=1, stderr="ERROR: package not found"
      ) as (temp_dir, uri, _),
      self.assertRaisesRegex(RuntimeError, "Failed to install"),
    ):
      _install_requirements(self.real_client(), uri, temp_dir)

  def test_empty_requirements_skipped(self):
    with self._seeded_install("") as (temp_dir, uri, mock_run):
      _install_requirements(self.real_client(), uri, temp_dir)

    mock_run.assert_not_called()


class TestMainWithRequirements(_RunnerTestCase):
  def test_4th_arg_triggers_install(self):
    """When a 4th arg is provided, _install_requirements is called."""
    with mock.patch(
      "kinetic.runner.remote_runner._install_requirements"
    ) as mock_install:
      exit_code, _ = self.run_runner(
        payload=_basic_payload(_add, args=(2, 3)),
        argv=lambda a: [*a.default_argv, "gs://bucket/requirements.txt"],
      )

    self.assertEqual(exit_code, 0)
    mock_install.assert_called_once_with(
      mock.ANY, "gs://bucket/requirements.txt", mock.ANY
    )
    # The client handed to the installer is main()'s real storage client.
    self.assertIsInstance(mock_install.call_args[0][0], storage.Client)


class TestMainArgValidation(parameterized.TestCase):
  @parameterized.named_parameters(
    ("no_args", []),
    ("one_arg", ["gs://bucket/context.zip"]),
    ("two_args", ["gs://bucket/context.zip", "gs://bucket/payload.pkl"]),
  )
  def test_missing_artifact_uris_exit_nonzero(self, uris):
    with (
      mock.patch("sys.argv", ["remote_runner.py", *uris]),
      self.assertRaises(SystemExit) as cm,
    ):
      main()

    self.assertEqual(cm.exception.code, 1)


class TestHostIndex(parameterized.TestCase):
  """Every pod gets the same command; the env var is what separates them."""

  @parameterized.named_parameters(
    ("unset", {}, 0),
    ("leader", {"TPU_WORKER_ID": "0"}, 0),
    ("worker", {"TPU_WORKER_ID": "3"}, 3),
    ("surrounding_whitespace", {"TPU_WORKER_ID": " 2 "}, 2),
    ("empty_falls_through", {"TPU_WORKER_ID": "", "LWS_WORKER_INDEX": "5"}, 5),
    (
      "blank_falls_through",
      {"TPU_WORKER_ID": "  ", "LWS_WORKER_INDEX": "1"},
      1,
    ),
    ("lws_only", {"LWS_WORKER_INDEX": "7"}, 7),
    ("tpu_wins_over_lws", {"TPU_WORKER_ID": "2", "LWS_WORKER_INDEX": "9"}, 2),
    (
      # The pod spec's "$(LWS_WORKER_INDEX)" only expands when the
      # webhook-injected variable precedes it in the container env.
      "unexpanded_substitution_falls_through",
      {"TPU_WORKER_ID": "$(LWS_WORKER_INDEX)", "LWS_WORKER_INDEX": "4"},
      4,
    ),
  )
  def test_index_resolution(self, env, expected):
    with mock.patch.dict(
      os.environ,
      {"TPU_WORKER_ID": "", "LWS_WORKER_INDEX": "", **env},
      clear=False,
    ):
      # A patched-in empty string is indistinguishable from "unset" here
      # because _host_index() skips blanks either way.
      self.assertEqual(remote_runner._host_index(), expected)

  @parameterized.named_parameters(
    ("not_a_number", "worker-2"),
    ("unexpanded_substitution", "$(LWS_WORKER_INDEX)"),
    ("negative", "-1"),
  )
  def test_unusable_value_degrades_to_leader_with_a_warning(self, raw):
    with (
      mock.patch.dict(os.environ, {"TPU_WORKER_ID": raw}, clear=False),
      mock.patch("kinetic.runner.remote_runner.logging.warning") as warn,
    ):
      index = remote_runner._host_index()

    self.assertEqual(index, 0)
    _assert_warned(self, warn, "TPU_WORKER_ID")


class TestWorkerResultUri(parameterized.TestCase):
  @parameterized.named_parameters(
    (
      "job_prefix",
      "gs://bucket/job-a1b2/result.pkl",
      3,
      "gs://bucket/job-a1b2/result-worker-3.pkl",
    ),
    (
      "nested_prefix",
      "gs://bucket/a/b/c/result.pkl",
      1,
      "gs://bucket/a/b/c/result-worker-1.pkl",
    ),
    ("bare_name", "result.pkl", 2, "result-worker-2.pkl"),
  )
  def test_sibling_uri_in_the_same_prefix(self, result_gcs, index, expected):
    self.assertEqual(
      remote_runner._worker_result_uri(result_gcs, index), expected
    )


class TestMainResultOwnership(_RunnerTestCase):
  """Only the leader writes result.pkl; workers report failures alone."""

  def _host_env(self, index):
    return mock.patch.dict(
      os.environ, {"TPU_WORKER_ID": str(index)}, clear=False
    )

  def test_leader_writes_the_canonical_result(self):
    exit_code, result = self.run_runner(
      payload=_basic_payload(_add, args=(2, 3)),
      patches=(self._host_env(0),),
    )

    self.assertEqual(exit_code, 0)
    self.assertEqual(result["result"], 5)
    self.assertEqual(result["host_index"], 0)
    self.assertEqual(
      self.job_blob_names(),
      ["job/context.zip", "job/payload.pkl", "job/result.pkl"],
    )

  def test_worker_success_uploads_nothing(self):
    exit_code, result = self.run_runner(
      payload=_basic_payload(_add, args=(2, 3)),
      patches=(self._host_env(2),),
    )

    self.assertEqual(exit_code, 0)
    # No result.pkl (the leader owns it) and no per-host blob either:
    # a non-leader's return value is discarded, not raced into GCS.
    self.assertIsNone(result)
    self.assertEqual(
      self.job_blob_names(), ["job/context.zip", "job/payload.pkl"]
    )

  def test_worker_success_never_serializes_its_return_value(self):
    """The discarded value is not pickled — N hosts must not pay for it."""
    with mock.patch(
      "kinetic.runner.remote_runner._dump_result_payload"
    ) as dump:
      exit_code, _ = self.run_runner(
        payload=_basic_payload(_identity, args=(42,)),
        patches=(self._host_env(1),),
      )

    self.assertEqual(exit_code, 0)
    dump.assert_not_called()

  def test_worker_failure_goes_to_its_own_blob(self):
    def bad_func():
      raise ValueError("worker 2 exploded")

    exit_code, result = self.run_runner(
      payload=_basic_payload(bad_func),
      patches=(self._host_env(2),),
    )

    self.assertEqual(exit_code, 1)
    self.assertIsNone(result)  # result.pkl stays untouched.
    worker_payload = self.job_blob("result-worker-2.pkl")
    self.assertFalse(worker_payload["success"])
    self.assertEqual(worker_payload["host_index"], 2)
    self.assertIsInstance(worker_payload["exception"], ValueError)
    self.assertIn("worker 2 exploded", str(worker_payload["exception"]))
    self.assertIn("ValueError: worker 2 exploded", worker_payload["traceback"])

  def test_worker_setup_failure_goes_to_its_own_blob(self):
    """Pre-execution failures follow the same ownership rule."""
    exit_code, result = self.run_runner(
      payload=_basic_payload(_noop),
      context_bytes=b"not a zip archive",
      patches=(self._host_env(3),),
    )

    self.assertEqual(exit_code, 1)
    self.assertIsNone(result)
    worker_payload = self.job_blob("result-worker-3.pkl")
    self.assertFalse(worker_payload["success"])
    self.assertEqual(worker_payload["host_index"], 3)
    self.assertEqual(worker_payload["phase"], "context extract")

  def test_leader_failure_still_goes_to_the_canonical_result(self):
    def bad_func():
      raise ValueError("leader exploded")

    exit_code, result = self.run_runner(
      payload=_basic_payload(bad_func),
      patches=(self._host_env(0),),
    )

    self.assertEqual(exit_code, 1)
    self.assertIsInstance(result["exception"], ValueError)
    self.assertEqual(result["host_index"], 0)
    self.assertNotIn("job/result-worker-0.pkl", self.job_blob_names())

  def test_unusable_host_index_keeps_the_legacy_leader_behavior(self):
    """A malformed index must not leave the job with no result at all."""
    exit_code, result = self.run_runner(
      payload=_basic_payload(_add, args=(2, 3)),
      patches=(
        mock.patch.dict(
          os.environ, {"TPU_WORKER_ID": "not-an-index"}, clear=False
        ),
      ),
    )

    self.assertEqual(exit_code, 0)
    self.assertEqual(result["result"], 5)

  def test_worker_serialization_failure_is_reported_to_its_own_blob(self):
    """A worker that cannot pickle its exception still reports one."""

    class UnpicklableError(Exception):
      def __reduce__(self):
        raise TypeError("cannot pickle UnpicklableError")

    def raise_unpicklable():
      raise UnpicklableError("boom")

    exit_code, result = self.run_runner(
      payload=_basic_payload(raise_unpicklable),
      patches=(self._host_env(4),),
    )

    self.assertEqual(exit_code, 1)
    self.assertIsNone(result)
    worker_payload = self.job_blob("result-worker-4.pkl")
    self.assertTrue(worker_payload["serialization_failed"])
    self.assertEqual(worker_payload["host_index"], 4)


class TestLeaderReadySentinel(_EmulatorTestCase):
  """Workers must fail loudly (not hang) if the leader never signals."""

  def _sentinel_env(self, bucket):
    return mock.patch.dict(
      os.environ,
      {
        "GCS_BUCKET": bucket,
        "JOB_ID": "job-abc",
        # Negative so leader_timeout + 60 buffer yields a small positive
        # timeout that elapses quickly after one poll.
        "KINETIC_DEBUG_WAIT_TIMEOUT": "-59",
      },
      clear=False,
    )

  def test_wait_raises_when_sentinel_never_appears(self):
    bucket = self.server.make_bucket(suffix="sentinel")

    with (
      self._sentinel_env(bucket),
      mock.patch("kinetic.runner.remote_runner.time.sleep"),
      self.assertRaisesRegex(RuntimeError, "Leader did not signal readiness"),
    ):
      _wait_for_leader_ready_sentinel()

  def test_wait_returns_when_sentinel_exists(self):
    bucket = self.server.make_bucket(suffix="sentinel")
    self.server.write_blob(bucket, "job-abc/.leader_ready", b"")

    with (
      self._sentinel_env(bucket),
      mock.patch("kinetic.runner.remote_runner.time.sleep"),
    ):
      _wait_for_leader_ready_sentinel()  # Returns without raising.


class TestVerifySha256(absltest.TestCase):
  def test_hash_match(self):
    file_path = _make_temp_path(self) / "test.pkl"
    file_path.write_text("dummy data")

    # Should not raise
    _verify_sha256(str(file_path), _sha256(str(file_path)), "test.pkl")

  def test_hash_mismatch(self):
    file_path = _make_temp_path(self) / "test.pkl"
    file_path.write_text("dummy data")

    with self.assertRaisesRegex(RuntimeError, "Security verification failed"):
      _verify_sha256(str(file_path), "bad-hash", "test.pkl")


class TestMainHashVerification(_RunnerTestCase):
  """Both artifact digests are checked before anything is unpickled."""

  @parameterized.named_parameters(
    ("both_hashes_match", False, False, 0),
    ("payload_hash_mismatch", True, False, 1),
    ("context_hash_mismatch", False, True, 1),
  )
  def test_hash_verification(self, break_payload, break_context, expected_code):
    def argv(artifacts):
      return [
        "remote_runner.py",
        "--context-gcs",
        artifacts.context_uri,
        "--payload-gcs",
        artifacts.payload_uri,
        "--result-gcs",
        artifacts.result_uri,
        "--payload-sha256",
        "0" * 64 if break_payload else _sha256(artifacts.payload_path),
        "--context-sha256",
        "0" * 64 if break_context else _sha256(artifacts.context_path),
      ]

    exit_code, result = self.run_runner(
      payload=_basic_payload(_add, args=(2, 3)), argv=argv
    )

    self.assertEqual(exit_code, expected_code)
    if expected_code == 0:
      self.assertEqual(result["result"], 5)
    else:
      self.assertEqual(result["phase"], "artifact verification")
      self.assertIn("Security verification failed", str(result["exception"]))


class TestResolveDataRefsTypePreservation(parameterized.TestCase):
  """Containers keep their type, state and aliasing across the walk."""

  def _resolve(self, args=(), kwargs=None):
    return resolve_data_refs(args, kwargs or {}, MagicMock(), "/tmp/data")

  @parameterized.named_parameters(
    dict(
      testcase_name="collections_namedtuple",
      value=Point(_MOUNTED_REF, 2),
      expected=("/mnt/data", 2),
      via_kwargs=False,
    ),
    dict(
      testcase_name="typing_namedtuple",
      value=TypedPoint(1, _MOUNTED_REF),
      expected=(1, "/mnt/data"),
      via_kwargs=False,
    ),
    dict(
      # A one-field namedtuple rebuilt with the wrong call shape comes
      # back holding a generator instead of the resolved path.
      testcase_name="single_field_namedtuple",
      value=OneField(_MOUNTED_REF),
      expected=("/mnt/data",),
      via_kwargs=False,
    ),
    dict(
      testcase_name="namedtuple_in_kwargs",
      value=Point(_MOUNTED_REF, _MOUNTED_REF),
      expected=("/mnt/data", "/mnt/data"),
      via_kwargs=True,
    ),
  )
  def test_namedtuple_type_and_fields_survive(
    self, value, expected, via_kwargs
  ):
    if via_kwargs:
      _, out = self._resolve((), {"p": value})
      rebuilt = out["p"]
    else:
      out, _ = self._resolve((value,))
      rebuilt = out[0]

    self.assertIsInstance(rebuilt, type(value))
    self.assertEqual(tuple(rebuilt), expected)

  @parameterized.named_parameters(
    ("tuple", (_MOUNTED_REF, 1), tuple, ("/mnt/data", 1)),
    ("list", [_MOUNTED_REF, 1], list, ["/mnt/data", 1]),
    ("dict", {"a": _MOUNTED_REF}, dict, {"a": "/mnt/data"}),
  )
  def test_exact_builtin_containers_stay_exact(
    self, value, expected_type, expected
  ):
    args, _ = self._resolve((value,))

    self.assertIs(type(args[0]), expected_type)
    self.assertEqual(args[0], expected)

  @parameterized.named_parameters(
    ("list_subclass", MyList([_MOUNTED_REF, 3]), ["/mnt/data", 3]),
    (
      "list_subclass_with_an_extra_ctor_arg",
      Batch([_MOUNTED_REF], tag="train"),
      ["/mnt/data"],
    ),
  )
  def test_list_subclass_type_survives(self, original, expected):
    args, _ = self._resolve((original,))

    self.assertIsInstance(args[0], type(original))
    self.assertEqual(list(args[0]), expected)

  @parameterized.named_parameters(
    dict(
      testcase_name="dict_subclass",
      original=MyDict(a=_MOUNTED_REF),
      key="a",
      factory=None,
    ),
    dict(
      testcase_name="ordered_dict",
      original=collections.OrderedDict(
        [("z", _MOUNTED_REF), ("a", 1), ("m", 2)]
      ),
      key="z",
      factory=None,
    ),
    dict(
      testcase_name="counter",
      original=collections.Counter(a=_MOUNTED_REF),
      key="a",
      factory=None,
    ),
    dict(
      testcase_name="defaultdict",
      original=_defaultdict_with_ref(),
      key="a",
      factory=list,
    ),
  )
  def test_dict_subclass_type_order_and_state_survive(
    self, original, key, factory
  ):
    args, _ = self._resolve((original,))
    rebuilt = args[0]

    self.assertIsInstance(rebuilt, type(original))
    self.assertEqual(list(rebuilt), list(original))
    self.assertEqual(rebuilt[key], "/mnt/data")
    self.assertIs(getattr(rebuilt, "default_factory", None), factory)

  def test_list_subclass_that_drops_items_falls_back_to_list(self):
    strict = StrictList("train")
    strict.append(_MOUNTED_REF)

    args, _ = self._resolve((strict,))

    self.assertIs(type(args[0]), list)
    self.assertEqual(args[0], ["/mnt/data"])

  def test_dict_subclass_that_drops_items_falls_back_to_dict(self):
    strict = StrictDict("train")
    strict["a"] = _MOUNTED_REF

    args, _ = self._resolve((strict,))

    self.assertIs(type(args[0]), dict)
    self.assertEqual(args[0], {"a": "/mnt/data"})

  def test_sets_are_left_untouched(self):
    plain = {1, 2, 3}
    frozen = frozenset({4, 5})

    args, _ = self._resolve((plain, frozen, _MOUNTED_REF))

    self.assertIs(args[0], plain)
    self.assertIs(args[1], frozen)
    self.assertEqual(args[2], "/mnt/data")

  def test_aliasing_within_one_argument_preserved(self):
    sub = [_MOUNTED_REF]

    args, _ = self._resolve(([sub, sub],))

    self.assertIs(args[0][0], args[0][1])
    self.assertEqual(args[0][0], ["/mnt/data"])

  def test_aliasing_across_args_and_kwargs_preserved(self):
    shared = {"d": _MOUNTED_REF}

    args, kwargs = self._resolve((shared,), {"k": shared})

    self.assertIs(args[0], kwargs["k"])

  @parameterized.named_parameters(
    ("namedtuple", Point(1, 2)),
    ("nested_dict_list_tuple", {"cfg": [1, 2, {"deep": (3, 4)}]}),
    ("plain_list", [1, 2]),
  )
  def test_unchanged_containers_keep_their_identity(self, original):
    args, _ = self._resolve((original, _MOUNTED_REF))

    self.assertIs(args[0], original)
    self.assertEqual(args[1], "/mnt/data")

  def test_self_referential_argument_terminates(self):
    cyclic = [_MOUNTED_REF]
    cyclic.append(cyclic)

    args, _ = self._resolve((cyclic,))

    self.assertEqual(args[0][0], "/mnt/data")
    self.assertIs(args[0][1], args[0])

  def test_mutually_referential_dicts_terminate(self):
    left = {"ref": _MOUNTED_REF}
    right = {"left": left}
    left["right"] = right

    args, _ = self._resolve((left,))

    self.assertEqual(args[0]["ref"], "/mnt/data")
    self.assertIs(args[0]["right"]["left"], args[0])

  def test_deeply_nested_refs_resolved(self):
    nested = _MOUNTED_REF
    for _ in range(60):
      nested = [nested]

    args, _ = self._resolve((nested,))

    current = args[0]
    for _ in range(60):
      current = current[0]
    self.assertEqual(current, "/mnt/data")

  def test_data_ref_as_dict_key_raises(self):
    key = HashableRef(_data_ref())

    with self.assertRaisesRegex(ValueError, "not supported as dict keys"):
      self._resolve(({key: 1},))


class TestPayloadHasDataRefs(parameterized.TestCase):
  """The declared fast path decides whether the walk runs at all."""

  @parameterized.named_parameters(
    ("declared_true_is_honored", {"has_data_refs": True}, (), {}, True),
    (
      "declared_false_skips_the_scan",
      {"has_data_refs": False},
      (_MOUNTED_REF,),
      {},
      False,
    ),
    ("missing_key_scans_args", {}, ([{"a": [_MOUNTED_REF]}],), {}, True),
    ("missing_key_scans_kwargs", {}, (), {"k": (_MOUNTED_REF,)}, True),
    ("missing_key_and_no_refs", {}, ([1, {"a": 2}],), {"k": 3}, False),
  )
  def test_declaration_or_scan(self, payload, args, kwargs, expected):
    self.assertEqual(_payload_has_data_refs(payload, args, kwargs), expected)

  def test_fallback_scan_survives_cycles(self):
    cyclic = [1]
    cyclic.append(cyclic)

    self.assertFalse(_payload_has_data_refs({}, (cyclic,), {}))

    cyclic.append(_MOUNTED_REF)
    self.assertTrue(_payload_has_data_refs({}, (cyclic,), {}))

  def test_fallback_scan_finds_a_ref_used_as_a_key(self):
    self.assertTrue(_contains_data_ref({HashableRef(_data_ref()): 1}))

  def test_fallback_scan_handles_deep_nesting(self):
    nested = _MOUNTED_REF
    for _ in range(5000):
      nested = [nested]

    self.assertTrue(_contains_data_ref(nested))


class TestExtractContext(parameterized.TestCase):
  """Extraction restores the permission bits ``extractall`` drops."""

  @parameterized.named_parameters(
    dict(
      testcase_name="unix_executable",
      create_system=_ZIP_CREATE_SYSTEM_UNIX,
      mode=0o755,
      executable=True,
    ),
    dict(
      testcase_name="unix_plain_file",
      create_system=_ZIP_CREATE_SYSTEM_UNIX,
      mode=0o644,
      executable=False,
    ),
    dict(
      # setuid/setgid/sticky must never survive an archive round trip.
      testcase_name="unix_setuid_is_masked_off",
      create_system=_ZIP_CREATE_SYSTEM_UNIX,
      mode=0o4755,
      executable=True,
    ),
    dict(
      testcase_name="unix_zero_mode_is_left_alone",
      create_system=_ZIP_CREATE_SYSTEM_UNIX,
      mode=0,
      executable=False,
    ),
    dict(
      # Non-Unix creators store unrelated data in the high bits, so the
      # "mode" read out of them is junk and must be ignored.
      testcase_name="non_unix_junk_bits_are_ignored",
      create_system=0,
      mode=0o777,
      executable=False,
    ),
    dict(
      testcase_name="non_unix_restrictive_junk_bits_are_ignored",
      create_system=0,
      mode=0o200,
      executable=False,
    ),
  )
  def test_mode_is_restored_only_from_unix_archives(
    self, create_system, mode, executable
  ):
    tmp = _make_temp_path(self)
    zip_path = tmp / "context.zip"
    _make_mode_zip(zip_path, "entry.sh", mode, create_system)
    dest = tmp / "workspace"

    _extract_context(str(zip_path), str(dest))

    target = dest / "entry.sh"
    applied = stat.S_IMODE(os.stat(target).st_mode)
    archive_mode = mode & 0o777
    if create_system == _ZIP_CREATE_SYSTEM_UNIX and archive_mode:
      self.assertEqual(applied, archive_mode)
    else:
      self.assertNotEqual(applied, archive_mode)
      self.assertTrue(os.access(target, os.R_OK))
    self.assertEqual(bool(applied & 0o111), executable)
    self.assertFalse(applied & 0o7000)

  def test_nested_entries_and_directories_extracted(self):
    tmp = _make_temp_path(self)
    zip_path = tmp / "context.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
      zf.writestr("pkg/__init__.py", "")
      zf.writestr("pkg/mod.py", "VALUE = 7")
      zf.writestr("empty/", "")

    dest = tmp / "workspace"
    _extract_context(str(zip_path), str(dest))

    self.assertEqual((dest / "pkg" / "mod.py").read_text(), "VALUE = 7")
    self.assertTrue((dest / "empty").is_dir())


class TestApplyWorkspacePlan(parameterized.TestCase):
  """T2.2 — sys.path/cwd reconstruction from ``.kinetic/plan.json``."""

  def setUp(self):
    super().setUp()
    self.addCleanup(setattr, sys, "path", sys.path[:])
    self.tmp = _make_temp_path(self)
    self.addCleanup(os.chdir, os.getcwd())
    self.workspace = self.tmp / "workspace"
    (self.workspace / "src").mkdir(parents=True)
    (self.workspace / "tools").mkdir()
    (self.workspace / ".kinetic").mkdir()

  def _write_plan(self, plan, raw=None):
    """Write ``plan.json``, either as JSON or as raw text."""
    path = self.workspace / ".kinetic" / "plan.json"
    path.write_text(raw if raw is not None else json.dumps(plan))

  def _assert_cwd(self, expected):
    self.assertEqual(
      os.path.realpath(os.getcwd()), os.path.realpath(str(expected))
    )

  def test_legacy_zip_without_plan_keeps_old_behavior(self):
    before = os.getcwd()

    inserted = _apply_workspace_plan(str(self.workspace))

    self.assertEqual(inserted, [str(self.workspace)])
    self.assertEqual(sys.path[0], str(self.workspace))
    self.assertEqual(os.getcwd(), before)

  def test_plan_inserts_entries_in_order_and_chdirs(self):
    self._write_plan({"sys_path_rel": ["", "src"], "client_cwd_rel": "tools"})

    inserted = _apply_workspace_plan(str(self.workspace))

    self.assertEqual(
      inserted, [str(self.workspace), str(self.workspace / "src")]
    )
    self.assertEqual(sys.path[0], str(self.workspace))
    self.assertEqual(sys.path[1], str(self.workspace / "src"))
    self._assert_cwd(self.workspace / "tools")

  def test_plan_without_cwd_rel_chdirs_to_the_root(self):
    self._write_plan({"sys_path_rel": [""]})

    _apply_workspace_plan(str(self.workspace))

    self._assert_cwd(self.workspace)

  def test_missing_cwd_dir_falls_back_to_the_root(self):
    self._write_plan({"sys_path_rel": [""], "client_cwd_rel": "gone"})

    _apply_workspace_plan(str(self.workspace))

    self._assert_cwd(self.workspace)

  def test_root_entry_is_added_when_the_plan_omits_it(self):
    self._write_plan({"sys_path_rel": ["src"]})

    inserted = _apply_workspace_plan(str(self.workspace))

    self.assertEqual(inserted[0], str(self.workspace))

  @parameterized.named_parameters(
    ("escapes_the_workspace", "../../etc"),
    ("absolute_path", "/etc"),
    ("not_shipped_in_the_context", "not_shipped"),
  )
  def test_unusable_sys_path_entries_are_skipped(self, rel):
    self._write_plan({"sys_path_rel": ["", rel]})

    inserted = _apply_workspace_plan(str(self.workspace))

    self.assertEqual(inserted, [str(self.workspace)])

  @parameterized.named_parameters(
    ("invalid_json", "{not json"),
    ("json_list", '["not", "a", "dict"]'),
    ("json_string", '"nope"'),
    ("empty_file", ""),
  )
  def test_unusable_plan_falls_back_to_legacy_behavior(self, raw):
    self._write_plan(None, raw=raw)
    before = os.getcwd()

    inserted = _apply_workspace_plan(str(self.workspace))

    self.assertEqual(inserted, [str(self.workspace)])
    self.assertEqual(sys.path[0], str(self.workspace))
    self.assertEqual(os.getcwd(), before)

  @parameterized.named_parameters(
    dict(
      testcase_name="sys_path_rel_is_a_dict",
      sys_path_rel={"src": True},
      expected_subdirs=[],
      fragments=["expected a list of strings", "dict"],
    ),
    dict(
      testcase_name="sys_path_rel_is_a_string",
      sys_path_rel="src",
      expected_subdirs=[],
      fragments=["expected a list of strings", "str"],
    ),
    dict(
      testcase_name="sys_path_rel_holds_non_strings",
      sys_path_rel=["", 3, "src", None],
      expected_subdirs=["src"],
      fragments=["Dropping non-string", "[3, None]"],
    ),
  )
  def test_malformed_sys_path_rel_degrades_with_a_warning(
    self, sys_path_rel, expected_subdirs, fragments
  ):
    self._write_plan({"sys_path_rel": sys_path_rel, "client_cwd_rel": "tools"})

    with mock.patch(
      "kinetic.runner.remote_runner.logging.warning"
    ) as mock_warn:
      inserted = _apply_workspace_plan(str(self.workspace))

    expected = [str(self.workspace)] + [
      str(self.workspace / sub) for sub in expected_subdirs
    ]
    self.assertEqual(inserted, expected)
    self.assertEqual(sys.path[: len(expected)], expected)
    # The rest of the plan still applies: the job is not failed.
    self._assert_cwd(self.workspace / "tools")
    _assert_warned(self, mock_warn, *fragments)

  def test_empty_cwd_rel_chdirs_to_the_root_without_warning(self):
    self._write_plan({"sys_path_rel": ["", "src"], "client_cwd_rel": ""})

    with mock.patch(
      "kinetic.runner.remote_runner.logging.warning"
    ) as mock_warn:
      _apply_workspace_plan(str(self.workspace))

    self._assert_cwd(self.workspace)
    mock_warn.assert_not_called()

  @parameterized.named_parameters(
    ("an_int", 5, True),
    ("a_list", ["tools"], True),
    ("an_explicit_null", None, False),
  )
  def test_invalid_client_cwd_rel_is_ignored(self, cwd_rel, warns):
    self._write_plan({"sys_path_rel": ["", "src"], "client_cwd_rel": cwd_rel})

    with mock.patch(
      "kinetic.runner.remote_runner.logging.warning"
    ) as mock_warn:
      inserted = _apply_workspace_plan(str(self.workspace))

    self.assertEqual(
      inserted, [str(self.workspace), str(self.workspace / "src")]
    )
    self._assert_cwd(self.workspace)
    if warns:
      _assert_warned(self, mock_warn, "client_cwd_rel")
    else:
      mock_warn.assert_not_called()


class TestMainWorkspacePlan(_RunnerTestCase):
  """End-to-end: the plan drives imports and relative file access."""

  def test_plan_enables_src_layout_import_and_relative_open(self):
    self.addCleanup(sys.modules.pop, "kinetic_plan_mod", None)

    def read_it():
      import kinetic_plan_mod

      with open("data.txt") as f:
        return kinetic_plan_mod.VALUE, f.read()

    exit_code, result = self.run_runner(
      payload=_basic_payload(read_it),
      zip_entries={
        "src/kinetic_plan_mod.py": "VALUE = 7",
        "run/data.txt": "hello",
      },
      plan_json={"sys_path_rel": ["", "src"], "client_cwd_rel": "run"},
    )

    self.assertEqual(exit_code, 0)
    self.assertEqual(result["result"], (7, "hello"))

  def test_malformed_plan_still_runs_the_job(self):
    self.addCleanup(sys.modules.pop, "kinetic_broken_plan_mod", None)

    def read_it():
      import kinetic_broken_plan_mod

      return kinetic_broken_plan_mod.VALUE

    exit_code, result = self.run_runner(
      payload=_basic_payload(read_it),
      zip_entries={"kinetic_broken_plan_mod.py": "VALUE = 13"},
      plan_json={"sys_path_rel": {"src": 1}, "client_cwd_rel": 5},
    )

    self.assertEqual(exit_code, 0)
    self.assertEqual(result["result"], 13)

  def test_legacy_zip_does_not_change_the_working_directory(self):
    before = os.path.realpath(os.getcwd())

    def where():
      return os.path.realpath(os.getcwd())

    exit_code, result = self.run_runner(
      payload=_basic_payload(where), zip_entries={"dummy.py": "x = 1"}
    )

    self.assertEqual(exit_code, 0)
    self.assertEqual(result["result"], before)

  def test_workspace_root_is_importable_without_a_plan(self):
    self.addCleanup(sys.modules.pop, "kinetic_legacy_mod", None)

    def read_it():
      import kinetic_legacy_mod

      return kinetic_legacy_mod.VALUE

    exit_code, result = self.run_runner(
      payload=_basic_payload(read_it),
      zip_entries={"kinetic_legacy_mod.py": "VALUE = 11"},
    )

    self.assertEqual(exit_code, 0)
    self.assertEqual(result["result"], 11)


class TestMainDataRefFastPath(_RunnerTestCase):
  """``has_data_refs`` decides whether the argument walk runs."""

  def test_declared_false_skips_the_walk(self):
    with mock.patch(
      "kinetic.runner.remote_runner.resolve_data_refs"
    ) as mock_resolve:
      exit_code, result = self.run_runner(
        payload=_basic_payload(
          _identity, args=(Point(1, 2),), has_data_refs=False
        )
      )

    mock_resolve.assert_not_called()
    self.assertEqual(exit_code, 0)
    self.assertIsInstance(result["result"], Point)

  def test_declared_true_runs_the_walk_and_uses_its_output(self):
    with mock.patch(
      "kinetic.runner.remote_runner.resolve_data_refs",
      return_value=(("resolved",), {}),
    ) as mock_resolve:
      exit_code, result = self.run_runner(
        payload=_basic_payload(_type_name, args=(1,), has_data_refs=True)
      )

    mock_resolve.assert_called_once()
    self.assertEqual(exit_code, 0)
    # The int argument was replaced by the resolver's output.
    self.assertEqual(result["result"], "str")

  def test_legacy_payload_without_the_key_still_resolves(self):
    exit_code, result = self.run_runner(
      payload=_basic_payload(_identity, args=(_MOUNTED_REF,))
    )

    self.assertEqual(exit_code, 0)
    self.assertEqual(result["result"], "/mnt/data")

  def test_namedtuple_argument_survives_a_data_carrying_call(self):
    def fields(point):
      return type(point).__name__, point.x, point.y

    exit_code, result = self.run_runner(
      payload=_basic_payload(
        fields, args=(Point(_MOUNTED_REF, 2),), has_data_refs=True
      )
    )

    self.assertEqual(exit_code, 0)
    self.assertEqual(result["result"], ("Point", "/mnt/data", 2))


class TestMainPhaseFailures(_RunnerTestCase):
  """Every pre-execution failure still produces a result payload."""

  @parameterized.named_parameters(
    dict(
      testcase_name="artifact_download",
      runner_kwargs=lambda: {
        "payload": _basic_payload(_noop),
        "patches": (
          mock.patch(
            "kinetic.runner.remote_runner._download_from_gcs",
            side_effect=RuntimeError("network is down"),
          ),
        ),
      },
      phase="artifact download",
      message="network is down",
      error_name="RuntimeError",
    ),
    dict(
      testcase_name="context_extract",
      runner_kwargs=lambda: {
        "payload": _basic_payload(_noop),
        "context_bytes": b"not a zip file",
      },
      phase="context extract",
      message="kinetic context extract failed",
      error_name="BadZipFile",
    ),
    dict(
      testcase_name="requirements_install",
      runner_kwargs=lambda: {
        "payload": _basic_payload(_noop),
        "argv": lambda a: [*a.default_argv, "gs://bucket/requirements.txt"],
        "patches": (
          mock.patch(
            "kinetic.runner.remote_runner._install_requirements",
            side_effect=RuntimeError("resolution impossible"),
          ),
        ),
      },
      phase="requirements install",
      message="resolution impossible",
      error_name="RuntimeError",
    ),
    dict(
      testcase_name="data_resolve",
      runner_kwargs=lambda: {
        "payload": _basic_payload(
          _identity, args=({HashableRef(_data_ref()): 1},)
        ),
      },
      phase="data resolve",
      message="not supported as dict keys",
      error_name="ValueError",
    ),
  )
  def test_pre_execution_failure_writes_a_result(
    self, runner_kwargs, phase, message, error_name
  ):
    exit_code, result = self.run_runner(**runner_kwargs())

    self.assertEqual(exit_code, 1)
    self.assertFalse(result["success"])
    self.assertIsNone(result["result"])
    self.assertEqual(result["phase"], phase)
    self.assertIn(message, str(result["exception"]))
    self.assertIn("Traceback", result["traceback"])
    self.assertIn(error_name, result["traceback"])

  def test_unpickle_failure_reports_versions_and_missing_module(self):
    payload_bytes = _ghost_payload(
      self,
      {"python": "3.99.0", "cloudpickle": "0.0.1", "kinetic": "9.9.9"},
    )

    exit_code, result = self.run_runner(payload_bytes=payload_bytes)

    self.assertEqual(exit_code, 1)
    self.assertEqual(result["phase"], "payload unpickle")
    message = str(result["exception"])
    self.assertIn("kinetic payload unpickle failed", message)
    self.assertIn("3.99.0", message)
    self.assertIn("cloudpickle 0.0.1", message)
    self.assertIn("kinetic_ghost_mod", message)

  def test_unpickle_failure_without_a_fingerprint_still_explains(self):
    payload_bytes = _ghost_payload(self, None)

    exit_code, result = self.run_runner(payload_bytes=payload_bytes)

    self.assertEqual(exit_code, 1)
    self.assertEqual(result["phase"], "payload unpickle")
    self.assertIn("carries no client fingerprint", str(result["exception"]))

  def test_failure_upload_error_is_not_fatal(self):
    with mock.patch(
      "kinetic.runner.remote_runner._upload_to_gcs",
      side_effect=RuntimeError("upload refused"),
    ):
      exit_code, _ = self.run_runner(
        payload=_basic_payload(_noop), context_bytes=b"not a zip file"
      )

    self.assertEqual(exit_code, 1)


def _current_fingerprint(**overrides):
  """A fingerprint that matches this interpreter unless overridden."""
  fingerprint = {
    "python": ".".join(str(p) for p in sys.version_info[:3]),
    "cloudpickle": cloudpickle.__version__,
  }
  fingerprint.update(overrides)
  return fingerprint


class TestFingerprintSkewWarning(parameterized.TestCase):
  @parameterized.named_parameters(
    ("python_minor_skew", {"python": "2.7.18"}, "Python version skew"),
    ("python_version_as_a_list", {"python": [2, 7, 18]}, "Python version skew"),
    (
      "cloudpickle_skew",
      _current_fingerprint(cloudpickle="0.0.1"),
      "cloudpickle version skew",
    ),
  )
  def test_skew_warns(self, fingerprint, fragment):
    with mock.patch(
      "kinetic.runner.remote_runner.logging.warning"
    ) as mock_warn:
      _warn_on_fingerprint_skew(fingerprint)

    _assert_warned(self, mock_warn, fragment)

  @parameterized.named_parameters(
    ("matching_versions", _current_fingerprint()),
    ("no_fingerprint", None),
    ("empty_fingerprint", {}),
    ("not_a_dict", "3.12.0"),
  )
  def test_no_skew_does_not_warn(self, fingerprint):
    with mock.patch(
      "kinetic.runner.remote_runner.logging.warning"
    ) as mock_warn:
      _warn_on_fingerprint_skew(fingerprint)

    mock_warn.assert_not_called()


class TestMainSystemExit(_RunnerTestCase):
  """``sys.exit()`` inside the user function is an ordinary return."""

  @parameterized.named_parameters(("bare_exit_code", None), ("zero", 0))
  def test_zero_exit_is_success(self, code):
    exit_code, result = self.run_runner(
      payload=_basic_payload(_exit_with_code, args=(code,))
    )

    self.assertEqual(exit_code, 0)
    self.assertTrue(result["success"])
    self.assertIsNone(result["result"])
    self.assertIsNone(result["exception"])

  @parameterized.named_parameters(
    ("nonzero_status", 3, "sys.exit(3)"),
    ("message_string", "bad config", "bad config"),
  )
  def test_nonzero_sys_exit_is_a_failure(self, code, fragment):
    exit_code, result = self.run_runner(
      payload=_basic_payload(_exit_with_code, args=(code,))
    )

    self.assertEqual(exit_code, 1)
    self.assertFalse(result["success"])
    self.assertIsInstance(result["exception"], RuntimeError)
    self.assertIn(fragment, str(result["exception"]))
    self.assertIn("SystemExit", result["traceback"])

  def test_keyboard_interrupt_keeps_its_type_name(self):
    def interrupted():
      raise KeyboardInterrupt()

    exit_code, result = self.run_runner(payload=_basic_payload(interrupted))

    self.assertEqual(exit_code, 1)
    self.assertIn("KeyboardInterrupt", str(result["exception"]))


class TestResultSerializationFallback(_RunnerTestCase):
  """A result that cannot be pickled is a FAILURE with a repr."""

  def test_non_pickling_error_hits_the_fallback(self):
    def build():
      return UnserializableResult()

    exit_code, result = self.run_runner(payload=_basic_payload(build))

    self.assertEqual(exit_code, 1)
    self.assertFalse(result["success"])
    self.assertTrue(result["serialization_failed"])
    self.assertEqual(result["phase"], "result serialization")
    self.assertIn("reduce exploded", str(result["exception"]))
    self.assertIn("UnserializableResult tag=abc", result["result_repr"])

  def test_successful_result_is_not_flagged(self):
    def build():
      return {"loss": 0.5}

    exit_code, result = self.run_runner(payload=_basic_payload(build))

    self.assertEqual(exit_code, 0)
    self.assertFalse(result.get("serialization_failed", False))
    self.assertNotIn("result_repr", result)

  def test_unreprable_result_still_produces_a_payload(self):
    class Hostile:
      def __reduce__(self):
        raise RuntimeError("nope")

      def __repr__(self):
        raise ValueError("no repr for you")

    def build():
      return Hostile()

    exit_code, result = self.run_runner(payload=_basic_payload(build))

    self.assertEqual(exit_code, 1)
    self.assertTrue(result["serialization_failed"])
    self.assertIn("unreprable", result["result_repr"])


class TestExitProcessHygiene(absltest.TestCase):
  """T2.6 — the pod must not hang on a stray non-daemon thread."""

  def _start_thread(self, name, daemon):
    """Start a thread that lives until the test finishes."""
    release = threading.Event()
    thread = threading.Thread(target=release.wait, name=name, daemon=daemon)
    thread.start()
    self.addCleanup(thread.join)
    self.addCleanup(release.set)
    return thread

  def test_clean_exit_uses_sys_exit(self):
    with self.assertRaises(SystemExit) as cm:
      _exit_process(0)

    self.assertEqual(cm.exception.code, 0)

  def test_lingering_thread_forces_os_exit(self):
    self._start_thread("kinetic-test-lingerer", daemon=False)

    with (
      mock.patch("kinetic.runner.remote_runner.os._exit") as mock_exit,
      mock.patch("kinetic.runner.remote_runner.logging.warning") as mock_warn,
      # os._exit never returns for real, so the mocked call falls through.
      self.assertRaises(SystemExit),
    ):
      _exit_process(7)

    mock_exit.assert_called_once_with(7)
    _assert_warned(self, mock_warn, "kinetic-test-lingerer")

  def test_daemon_threads_do_not_force_os_exit(self):
    self._start_thread("kinetic-test-daemon", daemon=True)

    with (
      mock.patch("kinetic.runner.remote_runner.os._exit") as mock_exit,
      self.assertRaises(SystemExit),
    ):
      _exit_process(0)

    mock_exit.assert_not_called()


class TestPreimportHfDependencies(parameterized.TestCase):
  """T2.6 — close the window where a project file shadows ``datasets``."""

  def setUp(self):
    super().setUp()
    self.addCleanup(setattr, sys, "path", sys.path[:])
    self._saved = sys.modules.pop("datasets", None)
    self.addCleanup(self._restore)
    self.tmp = _make_temp_path(self)
    self.addCleanup(os.chdir, os.getcwd())
    self.workspace = self.tmp / "workspace"
    self.workspace.mkdir()
    self.marker = self.tmp / "hijacked.txt"
    (self.workspace / "datasets.py").write_text(
      f"import pathlib\npathlib.Path({str(self.marker)!r}).write_text('x')\n"
    )
    sys.path.insert(0, str(self.workspace))

  def _restore(self):
    sys.modules.pop("datasets", None)
    if self._saved is not None:
      sys.modules["datasets"] = self._saved

  def _reach_workspace_through(self, entry):
    """Leave the workspace reachable only via a relative sys.path entry.

    This is what the runner really faces: the plan chdirs into the
    workspace, and an interpreter-supplied ``""``/``"."`` entry then
    resolves inside it.
    """
    sys.path.remove(str(self.workspace))
    sys.path.insert(0, entry)
    os.chdir(self.workspace)

  def test_hf_volume_import_bypasses_the_workspace(self):
    _preimport_hf_dependencies(
      (), {}, [{"uri": "hf://imdb?split=train"}], [str(self.workspace)]
    )

    self.assertFalse(self.marker.exists())
    self.assertEqual(sys.path[0], str(self.workspace))

  def test_hf_ref_in_args_bypasses_the_workspace(self):
    ref = _data_ref("hf://imdb")

    _preimport_hf_dependencies(([ref],), {}, [], [str(self.workspace)])

    self.assertFalse(self.marker.exists())

  @parameterized.named_parameters(("empty_string", ""), ("dot", "."))
  def test_relative_workspace_entry_is_filtered(self, entry):
    self._reach_workspace_through(entry)

    _preimport_hf_dependencies(
      (), {}, [{"uri": "hf://imdb"}], [str(self.workspace)]
    )

    self.assertFalse(self.marker.exists())
    self.assertEqual(sys.path[0], entry)

  @parameterized.named_parameters(
    ("absolute_entry", None), ("relative_entry", "")
  )
  def test_without_the_guard_the_workspace_would_win(self, entry):
    if entry is not None:
      self._reach_workspace_through(entry)

    _preimport_hf_dependencies((), {}, [{"uri": "hf://imdb"}], [])

    self.assertTrue(self.marker.exists())

  def test_no_hf_uri_means_no_import(self):
    _preimport_hf_dependencies((_MOUNTED_REF,), {}, [{"uri": "gs://b/p"}], [])

    self.assertFalse(self.marker.exists())
    self.assertNotIn("datasets", sys.modules)

  def test_already_imported_datasets_is_left_alone(self):
    sentinel = object()
    sys.modules["datasets"] = sentinel

    _preimport_hf_dependencies((), {}, [{"uri": "hf://imdb"}], [])

    self.assertIs(sys.modules["datasets"], sentinel)
    self.assertFalse(self.marker.exists())


class TestMainSubprocess(_EmulatorTestCase):
  """The interpreter-level entrypoint contract: a real ``python
  remote_runner.py`` process, real argv parsing, real exit codes, and the
  stdout stream the LogStreamer later reads — nothing patched at all."""

  def setUp(self):
    super().setUp()
    # Payload functions must not be pickled by reference: the subprocess
    # cannot import this module under the alias the test runner used.
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    self.addCleanup(
      cloudpickle.unregister_pickle_by_value, sys.modules[__name__]
    )

  def _seed(self, payload):
    """Upload artifacts for *payload*; returns the bucket name."""
    bucket = self.server.make_bucket(suffix="subproc")
    tmp = _make_temp_path(self)
    context_zip = tmp / "context.zip"
    _make_context_zip(context_zip)
    self.server.write_blob(bucket, "job/context.zip", context_zip.read_bytes())
    self.server.write_blob(
      bucket, "job/payload.pkl", cloudpickle.dumps(payload)
    )
    return bucket

  def _run(self, bucket):
    env = {**os.environ, "STORAGE_EMULATOR_HOST": self.server.host}
    return subprocess.run(
      [
        sys.executable,
        remote_runner.__file__,
        "--context-gcs",
        f"gs://{bucket}/job/context.zip",
        "--payload-gcs",
        f"gs://{bucket}/job/payload.pkl",
        "--result-gcs",
        f"gs://{bucket}/job/result.pkl",
      ],
      env=env,
      capture_output=True,
      text=True,
      timeout=120,
    )

  def test_success_roundtrip_exits_zero(self):
    bucket = self._seed(_basic_payload(_add, args=(2, 3)))

    proc = self._run(bucket)

    self.assertEqual(proc.returncode, 0, proc.stderr)
    result = cloudpickle.loads(self.server.read_blob(bucket, "job/result.pkl"))
    self.assertTrue(result["success"])
    self.assertEqual(result["result"], 5)

  def test_user_exception_exits_one_with_result_payload(self):
    bucket = self._seed(_basic_payload(_exit_with_code, args=(3,)))

    proc = self._run(bucket)

    self.assertEqual(proc.returncode, 1, proc.stderr)
    result = cloudpickle.loads(self.server.read_blob(bucket, "job/result.pkl"))
    self.assertFalse(result["success"])
    self.assertIn("sys.exit(3)", str(result["exception"]))


if __name__ == "__main__":
  absltest.main()
