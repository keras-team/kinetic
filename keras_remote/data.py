"""Data class for declaring data dependencies in remote functions.

Wraps local file/directory paths or GCS URIs. On the remote side, Data
resolves to a plain filesystem path — the user's function only sees paths.
"""

import hashlib
import os


class Data:
  """A reference to data that should be available on the remote pod.

  Wraps a local file/directory path or a GCS URI. When passed as a function
  argument or used in the ``volumes`` decorator parameter, Data is resolved
  to a plain filesystem path on the remote side. The user's function code
  never needs to know about Data — it just receives paths.

  Args:
      path: Local file/directory path (absolute or relative) or GCS URI
            (``gs://bucket/prefix``).

  Examples::

      # Local directory
      Data("./my_dataset/")

      # Local file
      Data("./config.json")

      # GCS URI
      Data("gs://my-bucket/datasets/imagenet/")
  """

  def __init__(self, path: str):
    self._raw_path = path
    if self.is_gcs:
      self._resolved_path = path
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
    Symlinks are not followed (followlinks=False) to ensure
    deterministic hashing and prevent circular symlink infinite
    recursion. Users with symlinked data should pass the resolved
    target path.
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
          with open(fpath, "rb") as f:
            while True:
              chunk = f.read(65536)  # 64 KB chunks
              if not chunk:
                break
              h.update(chunk)
    else:
      h.update(b"file:")
      h.update(os.path.basename(self._resolved_path).encode("utf-8"))
      with open(self._resolved_path, "rb") as f:
        while True:
          chunk = f.read(65536)
          if not chunk:
            break
          h.update(chunk)
    return h.hexdigest()

  def __repr__(self):
    return f"Data({self._raw_path!r})"


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
