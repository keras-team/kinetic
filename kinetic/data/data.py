"""Data class for declaring data dependencies in remote functions.

Wraps local file/directory paths or GCS URIs. On the remote side, Data
resolves to a plain filesystem path — the user's function only sees paths.
"""

import hashlib
import itertools
import os
import posixpath
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from absl import logging

# Directories with more files than this threshold are hashed in parallel
# using a thread pool. Below this, sequential hashing avoids pool overhead.
_PARALLEL_HASH_THRESHOLD = 16
_HASH_BATCH_SIZE = 512


def _hash_single_file(fpath: str, relpath: str) -> bytes:
  """SHA-256 of relpath + \\0 + file contents. Returns raw 32-byte digest."""
  h = hashlib.sha256()
  h.update(relpath.encode("utf-8"))
  h.update(b"\0")
  # 256 KB: matches hashlib.file_digest's default buffer size.
  with open(fpath, "rb") as f:
    for chunk in iter(partial(f.read, 2**18), b""):
      h.update(chunk)
  return h.digest()


def _hash_file_batch(batch: list[tuple[str, str]]) -> list[bytes]:
  """Hash a batch of (relpath, fpath) pairs. Returns list of 32-byte digests."""
  return [_hash_single_file(fpath, relpath) for relpath, fpath in batch]


class Data:
  """A reference to data that should be available on the remote pod.

  Wraps a local file/directory path or a GCS URI. When passed as a function
  argument or used in the ``volumes`` decorator parameter, Data is resolved
  to a plain filesystem path on the remote side. The user's function code
  never needs to know about Data — it just receives paths.

  Args:
      path: Local file/directory path (absolute or relative) or GCS URI
            (``gs://bucket/prefix``).

  .. note::

      For GCS URIs, a trailing slash indicates a directory (prefix).
      ``Data("gs://my-bucket/dataset/")`` is treated as a directory,
      while ``Data("gs://my-bucket/dataset")`` is treated as a single
      object. If you intend to reference a GCS directory, always
      include the trailing slash.

  Examples::

      # Local directory
      Data("./my_dataset/")

      # Local file
      Data("./config.json")

      # GCS directory — trailing slash required
      Data("gs://my-bucket/datasets/imagenet/")

      # GCS single object
      Data("gs://my-bucket/datasets/weights.h5")
  """

  def __init__(self, path: str):
    if not path:
      raise ValueError("Data path must not be empty")
    self._raw_path = path
    if self.is_gcs:
      self._resolved_path = path
      _warn_if_missing_trailing_slash(path)
    else:
      self._resolved_path = os.path.abspath(os.path.expanduser(path))
      if not os.path.exists(self._resolved_path):
        raise FileNotFoundError(
          f"Data path does not exist: {path} "
          f"(resolved to {self._resolved_path})"
        )

  @property
  def path(self) -> str:
    return self._resolved_path

  @property
  def is_gcs(self) -> bool:
    return self._raw_path.startswith("gs://")

  @property
  def is_dir(self) -> bool:
    if self.is_gcs:
      return self._raw_path.endswith("/")
    return os.path.isdir(self._resolved_path)

  def content_hash(self) -> str:
    """SHA-256 hash of all file contents in deterministic order.

    Uses two-level hashing for parallelism: each file is hashed
    independently (SHA-256 of relpath + contents), then per-file
    digests are combined in sorted-walk (DFS) order into a final hash.

    Includes a type prefix ("dir:" or "file:") to prevent collisions
    between a single file and a directory containing only that file.

    Symlinked directories are not recursed into (followlinks=False)
    to prevent infinite recursion from circular symlinks. Symlinked
    files are read and their resolved contents are hashed, so the
    hash reflects the actual data visible at runtime.
    """
    if self.is_gcs:
      raise ValueError("Cannot compute content hash for GCS URI")
    if os.path.isdir(self._resolved_path):
      return self._content_hash_dir()
    return self._content_hash_file()

  def _content_hash_file(self) -> str:
    h = hashlib.sha256()
    h.update(b"file:")
    h.update(
      _hash_single_file(
        self._resolved_path,
        os.path.basename(self._resolved_path),
      )
    )
    return h.hexdigest()

  def _content_hash_dir(self) -> str:
    resolved = self._resolved_path

    # Walk in sorted order for determinism. Sorting dirs in-place
    # controls os.walk's traversal order; sorting files within each
    # directory yields a deterministic DFS order without materializing
    # the full file list — critical for datasets with millions of files.
    def _iter_files():
      for root, dirs, files in os.walk(resolved, followlinks=False):
        dirs.sort()
        for fname in sorted(files):
          fpath = os.path.join(root, fname)
          relpath = os.path.relpath(fpath, resolved)
          yield (relpath, fpath)

    file_iter = _iter_files()
    first_batch = list(
      itertools.islice(file_iter, _PARALLEL_HASH_THRESHOLD + 1)
    )

    h = hashlib.sha256()
    h.update(b"dir:")

    if len(first_batch) <= _PARALLEL_HASH_THRESHOLD:
      # Small directory — hash sequentially, no pool overhead.
      for digest in _hash_file_batch(first_batch):
        h.update(digest)
    else:
      # Large directory — stream batches to a thread pool.
      # Futures are kept in a bounded deque so at most a few batches
      # worth of file tuples reside in memory at any time.
      max_workers = min(32, (os.cpu_count() or 4) + 4)
      with ThreadPoolExecutor(max_workers=max_workers) as pool:
        pending = deque()
        batch = []
        for item in itertools.chain(first_batch, file_iter):
          batch.append(item)
          if len(batch) >= _HASH_BATCH_SIZE:
            pending.append(pool.submit(_hash_file_batch, batch))
            batch = []
            # Drain oldest completed futures to bound memory.
            while len(pending) > max_workers * 2:
              for digest in pending.popleft().result():
                h.update(digest)
        if batch:
          pending.append(pool.submit(_hash_file_batch, batch))
        for future in pending:
          for digest in future.result():
            h.update(digest)

    return h.hexdigest()

  def __repr__(self):
    return f"Data({self._raw_path!r})"


def _warn_if_missing_trailing_slash(path: str) -> None:
  """Log a warning if a GCS path looks like a directory but has no trailing slash."""
  if path.endswith("/"):
    return
  gcs_path = path.split("//", 1)[1]  # strip gs://
  last_segment = posixpath.basename(gcs_path)
  if last_segment and "." not in last_segment:
    logging.warning(
      "GCS path %r does not end with '/' but the last segment "
      "(%r) has no file extension. If this is a directory "
      "(prefix), add a trailing slash: %r",
      path,
      last_segment,
      path + "/",
    )


def _make_data_ref(
  gcs_uri: str, is_dir: bool, mount_path: str | None = None
) -> dict[str, object]:
  """Create a serializable data reference dict.

  These dicts replace Data objects in the payload before serialization.
  The remote runner identifies them by the __data_ref__ key.
  """
  return {
    "__data_ref__": True,
    "gcs_uri": gcs_uri,
    "is_dir": is_dir,
    "mount_path": mount_path,
  }


def is_data_ref(obj: object) -> bool:
  """Check if an object is a serialized data reference."""
  return isinstance(obj, dict) and obj.get("__data_ref__") is True
