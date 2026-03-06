"""Data class for declaring data dependencies in remote functions.

Wraps local file/directory paths or GCS URIs. On the remote side, Data
resolves to a plain filesystem path — the user's function only sees paths.
"""

import hashlib
import os
import posixpath

from absl import logging


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
    """SHA-256 hash of all file contents, sorted by relative path.

    Includes a type prefix ("dir:" or "file:") to prevent collisions
    between a single file and a directory containing only that file.

    Symlinked directories are not recursed into (followlinks=False)
    to prevent infinite recursion from circular symlinks. Symlinked
    files are read and their resolved contents are hashed, so the
    hash reflects the actual data visible at runtime.
    """
    if self.is_gcs:
      raise ValueError("Cannot compute content hash for GCS URI")

    h = hashlib.sha256()
    if os.path.isdir(self._resolved_path):
      h.update(b"dir:")
      for root, dirs, files in os.walk(self._resolved_path, followlinks=False):
        dirs.sort()
        for fname in sorted(files):
          fpath = os.path.join(root, fname)
          relpath = os.path.relpath(fpath, self._resolved_path)
          h.update(relpath.encode("utf-8"))
          h.update(b"\0")
          with open(fpath, "rb") as f:
            while True:
              chunk = f.read(65536)  # 64 KB chunks
              if not chunk:
                break
              h.update(chunk)
          h.update(b"\0")
    else:
      h.update(b"file:")
      h.update(os.path.basename(self._resolved_path).encode("utf-8"))
      h.update(b"\0")
      with open(self._resolved_path, "rb") as f:
        while True:
          chunk = f.read(65536)
          if not chunk:
            break
          h.update(chunk)
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
