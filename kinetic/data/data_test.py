"""Tests for kinetic.data — Data class and helpers."""

import os
import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest

from kinetic.data import Data, _make_data_ref, is_data_ref
from kinetic.data.data import _PARALLEL_HASH_THRESHOLD


def _make_temp_path(test_case):
  """Create a temp directory that is cleaned up after the test."""
  td = tempfile.TemporaryDirectory()
  test_case.addCleanup(td.cleanup)
  return pathlib.Path(td.name)


class TestDataConstructor(absltest.TestCase):
  def test_local_file(self):
    tmp = _make_temp_path(self)
    f = tmp / "data.csv"
    f.write_text("a,b\n1,2\n")
    d = Data(str(f))

    self.assertEqual(d.path, str(f))
    self.assertFalse(d.is_gcs)
    self.assertFalse(d.is_dir)

  def test_local_directory(self):
    tmp = _make_temp_path(self)
    d_dir = tmp / "dataset"
    d_dir.mkdir()
    (d_dir / "train.csv").write_text("data")
    d = Data(str(d_dir))

    self.assertEqual(d.path, str(d_dir))
    self.assertFalse(d.is_gcs)
    self.assertTrue(d.is_dir)

  def test_gcs_uri_directory(self):
    d = Data("gs://my-bucket/data/")
    self.assertEqual(d.path, "gs://my-bucket/data/")
    self.assertTrue(d.is_gcs)
    self.assertTrue(d.is_dir)

  def test_gcs_uri_file(self):
    d = Data("gs://my-bucket/data/file.csv")
    self.assertTrue(d.is_gcs)
    self.assertFalse(d.is_dir)

  def test_empty_path_raises(self):
    with self.assertRaises(ValueError):
      Data("")

  def test_nonexistent_path_raises(self):
    with self.assertRaises(FileNotFoundError) as cm:
      Data("/nonexistent/path/to/data")
    self.assertIn("/nonexistent/path/to/data", str(cm.exception))

  def test_relative_path_resolved(self):
    tmp = _make_temp_path(self)
    f = tmp / "file.txt"
    f.write_text("content")
    # Use relative-like path by going through expanduser
    d = Data(str(f))
    self.assertTrue(os.path.isabs(d.path))

  def test_repr(self):
    d = Data("gs://bucket/path/")
    self.assertEqual(repr(d), "Data('gs://bucket/path/')")


class TestContentHash(absltest.TestCase):
  def test_deterministic_file_hash(self):
    tmp = _make_temp_path(self)
    f = tmp / "data.csv"
    f.write_text("hello,world\n")

    d1 = Data(str(f))
    d2 = Data(str(f))
    self.assertEqual(d1.content_hash(), d2.content_hash())

  def test_different_content_different_hash(self):
    tmp = _make_temp_path(self)
    f1 = tmp / "a.csv"
    f1.write_text("content_a")
    f2 = tmp / "b.csv"
    f2.write_text("content_b")

    self.assertNotEqual(
      Data(str(f1)).content_hash(), Data(str(f2)).content_hash()
    )

  def test_deterministic_dir_hash(self):
    tmp = _make_temp_path(self)
    d = tmp / "dataset"
    d.mkdir()
    (d / "train.csv").write_text("train data")
    (d / "val.csv").write_text("val data")

    d1 = Data(str(d))
    d2 = Data(str(d))
    self.assertEqual(d1.content_hash(), d2.content_hash())

  def test_dir_content_change_changes_hash(self):
    tmp = _make_temp_path(self)
    d = tmp / "dataset"
    d.mkdir()
    (d / "train.csv").write_text("original")

    h1 = Data(str(d)).content_hash()
    (d / "train.csv").write_text("modified")
    h2 = Data(str(d)).content_hash()

    self.assertNotEqual(h1, h2)

  def test_file_vs_dir_different_hash(self):
    """A single file and a directory containing only that file should
    produce different hashes due to the type prefix."""
    tmp = _make_temp_path(self)

    # Single file
    f = tmp / "file.csv"
    f.write_text("same content")

    # Directory containing only that file
    d = tmp / "dir"
    d.mkdir()
    (d / "file.csv").write_text("same content")

    file_hash = Data(str(f)).content_hash()
    dir_hash = Data(str(d)).content_hash()
    self.assertNotEqual(file_hash, dir_hash)

  def test_empty_directory(self):
    tmp = _make_temp_path(self)
    d = tmp / "empty"
    d.mkdir()

    # Should not raise, should return a valid hash
    h = Data(str(d)).content_hash()
    self.assertIsInstance(h, str)
    self.assertEqual(len(h), 64)  # SHA-256 hex digest

  def test_gcs_uri_raises(self):
    d = Data("gs://bucket/data/")
    with self.assertRaises(ValueError):
      d.content_hash()

  def test_nested_directory_hash(self):
    tmp = _make_temp_path(self)
    d = tmp / "nested"
    sub = d / "sub"
    sub.mkdir(parents=True)
    (d / "a.txt").write_text("a")
    (sub / "b.txt").write_text("b")

    h = Data(str(d)).content_hash()
    self.assertIsInstance(h, str)
    self.assertEqual(len(h), 64)

  def test_filename_content_boundary(self):
    """Filename/content collisions must produce different hashes.

    Without a delimiter, file "a" with content "bc" and file "ab" with
    content "c" would both hash the byte sequence "abc".
    """
    tmp = _make_temp_path(self)
    d1 = tmp / "dir1"
    d1.mkdir()
    (d1 / "a").write_text("bc")

    d2 = tmp / "dir2"
    d2.mkdir()
    (d2 / "ab").write_text("c")

    self.assertNotEqual(
      Data(str(d1)).content_hash(), Data(str(d2)).content_hash()
    )

  def test_file_boundary_across_entries(self):
    """Consecutive file entries must not collide.

    Without a delimiter between entries, two files ["x" -> "y", "z" -> ""]
    and ["x" -> "", "yz" -> ""] would produce the same hash input.
    """
    tmp = _make_temp_path(self)
    d1 = tmp / "dir1"
    d1.mkdir()
    (d1 / "x").write_text("y")
    (d1 / "z").write_text("")

    d2 = tmp / "dir2"
    d2.mkdir()
    (d2 / "x").write_text("")
    (d2 / "yz").write_text("")

    self.assertNotEqual(
      Data(str(d1)).content_hash(), Data(str(d2)).content_hash()
    )

  def test_path_included_in_hash(self):
    """Files with same content but different names produce different
    hashes."""
    tmp = _make_temp_path(self)
    d1 = tmp / "dir1"
    d1.mkdir()
    (d1 / "alpha.csv").write_text("same")

    d2 = tmp / "dir2"
    d2.mkdir()
    (d2 / "beta.csv").write_text("same")

    self.assertNotEqual(
      Data(str(d1)).content_hash(), Data(str(d2)).content_hash()
    )

  def test_parallel_determinism_many_files(self):
    """Directory with many files exercises the thread pool path and
    must still produce deterministic hashes."""
    tmp = _make_temp_path(self)
    d = tmp / "large_dir"
    d.mkdir()
    num_files = _PARALLEL_HASH_THRESHOLD + 30
    for i in range(num_files):
      (d / f"file_{i:04d}.txt").write_text(f"content_{i}")

    hashes = [Data(str(d)).content_hash() for _ in range(5)]
    self.assertTrue(all(h == hashes[0] for h in hashes))

  def test_parallel_threshold_boundary(self):
    """Directories at and just above the threshold both produce valid
    deterministic hashes."""
    tmp = _make_temp_path(self)
    for count in (_PARALLEL_HASH_THRESHOLD, _PARALLEL_HASH_THRESHOLD + 1):
      d = tmp / f"dir_{count}"
      d.mkdir()
      for i in range(count):
        (d / f"f{i}.txt").write_text(f"data{i}")

      h1 = Data(str(d)).content_hash()
      h2 = Data(str(d)).content_hash()
      self.assertEqual(h1, h2)
      self.assertEqual(len(h1), 64)

  def test_streaming_does_not_materialise_full_file_list(self):
    """The peak number of live file tuples must be bounded, not O(total).

    We wrap os.walk so that every filename it yields increments a
    counter, and wrap _hash_file_batch so that every file it consumes
    decrements the same counter.  The peak value of that counter is the
    high-water mark for live tuples.  A non-streaming implementation
    would hit peak == total_files; the streaming one must stay well
    below that.
    """
    tmp = _make_temp_path(self)
    d = tmp / "big"
    d.mkdir()
    # Spread files across many subdirectories so os.walk yields entries
    # incrementally rather than in one giant batch.
    num_subdirs = 20
    files_per_subdir = 5
    for i in range(num_subdirs):
      sub = d / f"sub_{i:03d}"
      sub.mkdir()
      for j in range(files_per_subdir):
        (sub / f"f{j}.txt").write_text(f"v{i}_{j}")
    total_files = num_subdirs * files_per_subdir

    # Reference hash with default settings.
    h_ref = Data(str(d)).content_hash()

    # Track live file tuples: +1 when yielded by os.walk, -1 when
    # consumed by _hash_file_batch.  Access is single-threaded within
    # the main loop; worker threads only decrement after the main
    # thread has stopped incrementing for that batch, so plain ints
    # are safe here.
    import threading

    lock = threading.Lock()
    alive = 0
    peak_alive = 0
    real_walk = os.walk
    from kinetic.data import data as _data_mod

    real_hash = _data_mod._hash_file_batch

    def tracking_walk(*args, **kwargs):
      nonlocal alive, peak_alive
      for root, dirs, files in real_walk(*args, **kwargs):
        with lock:
          alive += len(files)
          peak_alive = max(peak_alive, alive)
        yield root, dirs, files

    def tracking_hash(batch):
      nonlocal alive
      result = real_hash(batch)
      with lock:
        alive -= len(batch)
      return result

    # Shrink batch size so more submit/drain cycles occur and the
    # bounded-drain logic has a chance to reclaim tuples mid-walk.
    # cpu_count=1 → max_workers=5 → drain threshold=10 futures.
    with (
      mock.patch("kinetic.data.data._HASH_BATCH_SIZE", 4),
      mock.patch("kinetic.data.data.os.walk", side_effect=tracking_walk),
      mock.patch(
        "kinetic.data.data._hash_file_batch",
        side_effect=tracking_hash,
      ),
      mock.patch("kinetic.data.data.os.cpu_count", return_value=1),
    ):
      h = Data(str(d)).content_hash()

    self.assertEqual(h_ref, h)
    # A non-streaming implementation would hit peak_alive == total_files
    # because all walk events fire before any hash events.  The
    # streaming implementation keeps peak_alive well below total.
    self.assertLess(
      peak_alive,
      total_files,
      f"Peak live file count ({peak_alive}) equals total files "
      f"({total_files}); the full file list was materialised.",
    )


class TestMakeDataRef(absltest.TestCase):
  def test_basic_ref(self):
    ref = _make_data_ref("gs://b/prefix", True)
    self.assertTrue(ref["__data_ref__"])
    self.assertEqual(ref["gcs_uri"], "gs://b/prefix")
    self.assertTrue(ref["is_dir"])
    self.assertIsNone(ref["mount_path"])

  def test_with_mount_path(self):
    ref = _make_data_ref("gs://b/p", False, mount_path="/data")
    self.assertEqual(ref["mount_path"], "/data")
    self.assertFalse(ref["is_dir"])


class TestIsDataRef(absltest.TestCase):
  def test_valid_ref(self):
    ref = {"__data_ref__": True, "gcs_uri": "gs://b/p", "is_dir": True}
    self.assertTrue(is_data_ref(ref))

  def test_plain_dict(self):
    self.assertFalse(is_data_ref({"key": "value"}))

  def test_non_dict(self):
    self.assertFalse(is_data_ref("string"))
    self.assertFalse(is_data_ref(42))
    self.assertFalse(is_data_ref(None))


if __name__ == "__main__":
  absltest.main()
