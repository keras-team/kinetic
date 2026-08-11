"""Packaging utilities for serializing functions, args, and working directories.

Handles zipping the user's working directory, serializing the function
payload with cloudpickle, and extracting/replacing Data objects in
arbitrarily nested arg structures.
"""

import contextlib
import fnmatch
import json
import os
import posixpath
import subprocess
import sys
import zipfile
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import cloudpickle
from absl import logging

from kinetic import version
from kinetic.data import Data

# Type alias for a position path through nested args, e.g. ("arg", 0, "key").
PositionPath = tuple[str | int, ...]

# Directories that are never archived, even when default excludes are off.
_ALWAYS_EXCLUDED_DIRS = frozenset({".git", "__pycache__"})

# Directories/files excluded unless KINETIC_NO_DEFAULT_EXCLUDES=1.
_DEFAULT_EXCLUDED_DIRS = frozenset(
  {
    ".venv",
    "venv",
    "node_modules",
    ".tox",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".ipynb_checkpoints",
  }
)
_DEFAULT_EXCLUDED_FILES = frozenset({".DS_Store"})

# Filenames that look like credentials; archived, but warned about.
_SECRET_FILE_PATTERNS = (".env*", "*.pem", "id_rsa*")

_KINETICIGNORE = ".kineticignore"

# Reserved archive path carrying the client's packaging plan.
_PLAN_ARCHIVE_NAME = ".kinetic/plan.json"


def _list_git_files(base_dir: str) -> list[str] | None:
  """List tracked and non-ignored untracked files under ``base_dir``."""
  try:
    result = subprocess.run(
      [
        "git",
        "-C",
        base_dir,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        ".",
      ],
      check=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.DEVNULL,
    )
  except (OSError, subprocess.CalledProcessError):
    return None

  return [os.fsdecode(path) for path in result.stdout.split(b"\0") if path]


def _path_is_excluded(path: str, exclude_paths: set[str]) -> bool:
  """Check if a path should be excluded from archiving."""
  if not exclude_paths:
    return False
  normalized_path = os.path.normpath(path)
  return any(
    normalized_path == excluded or normalized_path.startswith(excluded + os.sep)
    for excluded in exclude_paths
  )


def _write_git_files(
  zipf: zipfile.ZipFile,
  base_dir: str,
  git_files: list[str],
  exclude_paths: set[str],
  archive_prefix: str = "",
) -> None:
  """Write files from git ls-files to ZIP, respecting exclusions."""
  for relative_path in git_files:
    file_path = os.path.join(base_dir, relative_path)
    if _path_is_excluded(file_path, exclude_paths) or not os.path.lexists(
      file_path
    ):
      continue

    archive_name = posixpath.join(archive_prefix, relative_path)
    if os.path.isdir(file_path) and not os.path.islink(file_path):
      nested_files = _list_git_files(file_path)
      if nested_files is not None:
        _write_git_files(
          zipf,
          file_path,
          nested_files,
          exclude_paths,
          archive_prefix=archive_name,
        )
      continue
    try:
      zipf.write(file_path, archive_name)
    except OSError as e:
      logging.warning("Could not archive %s: %s", file_path, e)


_MB = 1024 * 1024
_DEFAULT_CONTEXT_SIZE_WARN_MB = 100.0
_DEFAULT_PAYLOAD_SIZE_WARN_MB = 50.0

# Marker stored in the replacement memo while an immutable container is
# being rebuilt. Immutable containers cannot be patched after the fact, so
# re-entering one means the argument graph is self-referential through a
# tuple/set, which cannot be reconstructed.
_IN_PROGRESS = object()


def _format_path(path: PositionPath) -> str:
  """Render a position path as user-facing text, e.g. ``kwarg 'cfg'[0]``."""
  if not path:
    return "the arguments"
  kind = path[0]
  head = f"arg {path[1]}" if kind == "arg" else f"kwarg {path[1]!r}"
  return head + "".join(f"[{part!r}]" for part in path[2:])


def _size_warn_bytes(env_name: str, default_mb: float) -> float:
  """Read an MB threshold from the environment, falling back to a default."""
  raw = os.environ.get(env_name)
  if raw:
    try:
      return float(raw) * _MB
    except ValueError:
      logging.warning(
        "Ignoring invalid %s=%r (expected a number of megabytes)", env_name, raw
      )
  return default_mb * _MB


def _read_kineticignore(base_dir: str) -> list[tuple[str, bool]]:
  """Parse ``.kineticignore`` at *base_dir* into (pattern, dir_only) pairs."""
  ignore_path = os.path.join(base_dir, _KINETICIGNORE)
  if not os.path.isfile(ignore_path):
    return []
  patterns: list[tuple[str, bool]] = []
  try:
    with open(ignore_path, encoding="utf-8", errors="replace") as f:
      lines = f.read().splitlines()
  except OSError as e:
    logging.warning("Could not read %s: %s", ignore_path, e)
    return []
  for line in lines:
    pattern = line.strip()
    if not pattern or pattern.startswith("#"):
      continue
    dir_only = pattern.endswith("/")
    pattern = pattern.rstrip("/").lstrip("/")
    if pattern:
      patterns.append((pattern, dir_only))
  if patterns:
    logging.info("Applying %d pattern(s) from %s", len(patterns), ignore_path)
  return patterns


def _matches_ignore(
  rel_path: str, patterns: list[tuple[str, bool]], is_dir: bool
) -> bool:
  """Return True when *rel_path* matches a ``.kineticignore`` pattern."""
  posix_path = rel_path.replace(os.sep, "/")
  name = posix_path.rsplit("/", 1)[-1]
  for pattern, dir_only in patterns:
    if dir_only and not is_dir:
      continue
    if fnmatch.fnmatch(posix_path, pattern) or fnmatch.fnmatch(name, pattern):
      return True
  return False


def _is_secret_name(name: str) -> bool:
  """Return True when a filename looks like a credential."""
  return any(fnmatch.fnmatch(name, pat) for pat in _SECRET_FILE_PATTERNS)


def _dir_identity(path: str) -> tuple[int, int] | None:
  """Return ``(st_dev, st_ino)`` for a directory, or None when unreadable."""
  try:
    st = os.stat(path)
  except OSError:
    return None
  return (st.st_dev, st.st_ino)


def zip_working_dir(
  base_dir: str,
  output_path: str,
  exclude_paths: set[str] | None = None,
  plan_json: dict[str, Any] | None = None,
) -> None:
  """Zip a directory into a ZIP archive, excluding common non-source files.

  When in a git repository, respects ``.gitignore`` and uses git ls-files
  to determine which files to include. Falls back to directory traversal
  with ``.kineticignore`` patterns when not in a git repo.

  Symlinked directories are followed (with a cycle guard), empty
  directories are preserved, and files that cannot be archived (broken
  symlinks, unreadable files) are skipped with a warning instead of
  aborting the submission.

  Excludes ``.git`` and ``__pycache__`` always, plus virtualenv/cache
  directories unless ``KINETIC_NO_DEFAULT_EXCLUDES=1`` is set. A
  ``.kineticignore`` file at *base_dir* adds fnmatch-style patterns.

  Args:
      base_dir: Root directory to zip.
      output_path: Destination path for the ZIP file.
      exclude_paths: Absolute paths to skip during archiving.
      plan_json: Optional packaging plan, written into the archive at the
          reserved path ``.kinetic/plan.json`` for the remote runner.
  """
  exclude_paths = exclude_paths or set()
  normalized_excludes = {os.path.normpath(p) for p in exclude_paths}
  no_defaults = os.environ.get("KINETIC_NO_DEFAULT_EXCLUDES") == "1"
  excluded_dirs = _ALWAYS_EXCLUDED_DIRS
  excluded_files: frozenset[str] = frozenset()
  if not no_defaults:
    excluded_dirs = excluded_dirs | _DEFAULT_EXCLUDED_DIRS
    excluded_files = _DEFAULT_EXCLUDED_FILES
  ignore_patterns = _read_kineticignore(base_dir)

  seen_dirs = set()
  base_identity = _dir_identity(base_dir)
  if base_identity is not None:
    seen_dirs.add(base_identity)

  archived: list[tuple[int, str]] = []
  secrets: list[str] = []

  with zipfile.ZipFile(
    output_path, "w", zipfile.ZIP_DEFLATED, strict_timestamps=False
  ) as zipf:
    # Try git ls-files first if in a git repository
    git_files = _list_git_files(base_dir)
    if git_files is not None:
      for relative_path in git_files:
        file_path = os.path.join(base_dir, relative_path)
        if _path_is_excluded(file_path, normalized_excludes) or not os.path.lexists(
          file_path
        ):
          continue

        archive_name = relative_path
        if os.path.isdir(file_path) and not os.path.islink(file_path):
          # Empty directories are preserved in git mode
          rel_dir = archive_name.replace(os.sep, "/")
          info = zipfile.ZipInfo(rel_dir + "/")
          info.external_attr = (0o40755 << 16) | 0x10
          zipf.writestr(info, b"")
          continue

        try:
          size = os.path.getsize(file_path)
          zipf.write(file_path, archive_name)
          archived.append((size, archive_name))
          if _is_secret_name(os.path.basename(file_path)):
            secrets.append(archive_name)
        except (OSError, ValueError, UnicodeEncodeError) as e:
          logging.warning("Skipping %s: %s", file_path, e)

      if plan_json is not None:
        zipf.writestr(
          _PLAN_ARCHIVE_NAME, json.dumps(plan_json, indent=2, default=str)
        )
      _report_context_size(output_path, archived)
      if secrets:
        logging.warning(
          "Credential-shaped files are being uploaded with your code: %s. They "
          "will be stored in the job's Cloud Storage bucket. Add them to a "
          "%s file at %s to keep them out of the archive.",
          ", ".join(sorted(secrets)),
          _KINETICIGNORE,
          base_dir,
        )
      return

    # Fall back to directory traversal with os.walk
    for root, dirs, files in os.walk(base_dir, followlinks=True):
      kept_dirs = []
      for name in dirs:
        dir_path = os.path.join(root, name)
        if name in excluded_dirs:
          continue
        if os.path.normpath(dir_path) in normalized_excludes:
          continue
        rel_dir = os.path.relpath(dir_path, base_dir)
        if _matches_ignore(rel_dir, ignore_patterns, is_dir=True):
          continue
        identity = _dir_identity(dir_path)
        if identity is None:
          logging.warning("Skipping unreadable directory %s", dir_path)
          continue
        if identity in seen_dirs:
          logging.warning(
            "Skipping %s: it links back to a directory already in the "
            "archive (symlink loop)",
            dir_path,
          )
          continue
        seen_dirs.add(identity)
        kept_dirs.append(name)
      dirs[:] = kept_dirs

      wrote_file = False
      for name in files:
        file_path = os.path.join(root, name)
        if name in excluded_files:
          continue
        if os.path.normpath(file_path) in normalized_excludes:
          continue
        archive_name = os.path.relpath(file_path, base_dir)
        if _matches_ignore(archive_name, ignore_patterns, is_dir=False):
          continue
        if (
          plan_json is not None
          and archive_name.replace(os.sep, "/") == _PLAN_ARCHIVE_NAME
        ):
          continue
        if os.path.islink(file_path) and not os.path.exists(file_path):
          logging.warning(
            "Skipping broken symlink %s -> %s",
            file_path,
            os.readlink(file_path),
          )
          continue
        try:
          size = os.path.getsize(file_path)
          zipf.write(file_path, archive_name)
        except (OSError, ValueError, UnicodeEncodeError) as e:
          logging.warning("Skipping %s: %s", file_path, e)
          continue
        wrote_file = True
        archived.append((size, archive_name))
        if _is_secret_name(name):
          secrets.append(archive_name)

      is_base = os.path.normpath(root) == os.path.normpath(base_dir)
      if not wrote_file and not dirs and not is_base:
        rel_dir = os.path.relpath(root, base_dir).replace(os.sep, "/")
        info = zipfile.ZipInfo(rel_dir + "/")
        info.external_attr = (0o40755 << 16) | 0x10
        zipf.writestr(info, b"")

    if plan_json is not None:
      zipf.writestr(
        _PLAN_ARCHIVE_NAME, json.dumps(plan_json, indent=2, default=str)
      )

  _report_context_size(output_path, archived)
  if secrets:
    logging.warning(
      "Credential-shaped files are being uploaded with your code: %s. They "
      "will be stored in the job's Cloud Storage bucket. Add them to a "
      "%s file at %s to keep them out of the archive.",
      ", ".join(sorted(secrets)),
      _KINETICIGNORE,
      base_dir,
    )


def _report_context_size(
  output_path: str, archived: list[tuple[int, str]]
) -> None:
  """Log the archive size and warn when it exceeds the configured limit."""
  try:
    total = os.path.getsize(output_path)
  except OSError:
    return
  logging.info(
    "Packaged %d file(s) into %s (%.1f MB compressed)",
    len(archived),
    output_path,
    total / _MB,
  )
  limit = _size_warn_bytes(
    "KINETIC_CONTEXT_SIZE_WARN_MB", _DEFAULT_CONTEXT_SIZE_WARN_MB
  )
  if limit <= 0 or total <= limit:
    return
  largest = sorted(archived, reverse=True)[:5]
  listing = "\n".join(
    f"  {size / _MB:8.1f} MB  {name}" for size, name in largest
  )
  logging.warning(
    "The working-directory archive is %.1f MB and is uploaded on every "
    "run. Largest files:\n%s\nAdd a %s file to exclude paths, or pass big "
    "datasets with kinetic.Data(...) instead.",
    total / _MB,
    listing,
    _KINETICIGNORE,
  )


def _client_fingerprint() -> dict[str, str]:
  """Versions of the client-side toolchain that produced the payload."""
  return {
    "python": ".".join(str(part) for part in sys.version_info[:3]),
    "cloudpickle": getattr(cloudpickle, "__version__", "unknown"),
    "kinetic": version.__version__,
  }


def _contains_data_ref(obj: Any, seen: set[int] | None = None) -> bool:
  """Return True when a ``__data_ref__`` dict is reachable from *obj*."""
  if seen is None:
    seen = set()
  obj_id = id(obj)
  if obj_id in seen:
    return False
  if isinstance(obj, dict):
    if obj.get("__data_ref__"):
      return True
    seen.add(obj_id)
    return any(_contains_data_ref(v, seen) for v in obj.values())
  if isinstance(obj, (list, tuple, set, frozenset)):
    seen.add(obj_id)
    return any(_contains_data_ref(item, seen) for item in obj)
  return False


def _register_by_value_modules(package_root: str | None) -> list[Any]:
  """Register first-party modules under *package_root* for by-value pickling.

  Modules living under the packaging root are shipped inside the payload
  instead of being imported on the pod, so helper modules do not have to be
  importable remotely. ``kinetic`` itself is never registered (dev checkouts
  place it under the same root).

  Returns:
      The list of module objects that were registered, for unregistration.
  """
  if not package_root:
    return []
  try:
    root = os.path.realpath(package_root)
  except OSError:
    return []
  registered = []
  for name, module in list(sys.modules.items()):
    if module is None or name == "__main__":
      continue
    if name == "kinetic" or name.startswith("kinetic."):
      continue
    module_file = getattr(module, "__file__", None)
    if not module_file:
      continue
    try:
      if os.path.commonpath([os.path.realpath(module_file), root]) != root:
        continue
    except (OSError, ValueError):
      continue
    try:
      cloudpickle.register_pickle_by_value(module)
    except Exception as e:  # noqa: BLE001 - never fail packaging over this
      logging.debug("Could not ship module %s by value: %s", name, e)
      continue
    registered.append(module)
  if registered:
    logging.info(
      "Shipping %d module(s) from %s inside the payload",
      len(registered),
      package_root,
    )
  return registered


def _unregister_by_value_modules(modules: list[Any]) -> None:
  """Undo :func:`_register_by_value_modules` (best effort)."""
  for module in modules:
    try:
      cloudpickle.unregister_pickle_by_value(module)
    except Exception as e:  # noqa: BLE001 - registry is process-global state
      logging.debug("Could not unregister module %s: %s", module, e)


def _payload_components(
  func: Callable, args: tuple, kwargs: dict[str, Any]
) -> list[tuple[str, Any]]:
  """Label each independently-picklable piece of the payload."""
  components: list[tuple[str, Any]] = [
    (f"the function {getattr(func, '__name__', repr(func))!r}", func)
  ]
  components.extend((f"argument {i}", arg) for i, arg in enumerate(args))
  components.extend(
    (f"keyword argument {key!r}", value) for key, value in kwargs.items()
  )
  return components


def _pickling_error(
  exc: BaseException, func: Callable, args: tuple, kwargs: dict[str, Any]
) -> ValueError:
  """Bisect the payload to name the component that cannot be serialized."""
  advice = (
    "Everything passed to a remote function must be picklable. Move locks, "
    "sockets, database connections, open file handles and generators inside "
    "the function, or pass a plain value (a path, a list, or "
    "kinetic.Data(...)) instead."
  )
  for label, value in _payload_components(func, args, kwargs):
    try:
      cloudpickle.dumps(value)
    except Exception as inner:  # noqa: BLE001 - any failure identifies it
      return ValueError(
        f"kinetic could not serialize {label} "
        f"(type {type(value).__name__}): {inner}\n{advice}"
      )
  return ValueError(
    f"kinetic could not serialize the job payload: {exc}\n{advice}"
  )


def _warn_on_payload_size(output_path: str) -> None:
  """Warn when the serialized payload is unusually large."""
  try:
    size = os.path.getsize(output_path)
  except OSError:
    return
  limit = _size_warn_bytes(
    "KINETIC_PAYLOAD_SIZE_WARN_MB", _DEFAULT_PAYLOAD_SIZE_WARN_MB
  )
  if limit > 0 and size > limit:
    logging.warning(
      "The serialized function payload is %.1f MB. Arguments and any "
      "module-level globals your function references are captured by value; "
      "pass large datasets with kinetic.Data(...) instead.",
      size / _MB,
    )


def save_payload(
  func: Callable,
  args: tuple,
  kwargs: dict[str, Any],
  env_vars: dict[str, str],
  output_path: str,
  volumes: list[dict[str, Any]] | None = None,
  working_dir: str | None = None,
  package_root: str | None = None,
  payload_extra: dict[str, Any] | None = None,
) -> None:
  """Serialize a function call payload with cloudpickle.

  The resulting pickle file contains a dict with keys ``func``, ``args``,
  ``kwargs``, ``env_vars``, ``has_data_refs``, ``client_fingerprint``, and
  optionally ``volumes``, ``working_dir``, ``package_root`` plus anything
  in *payload_extra*. All keys beyond the first four are additive: older
  runners ignore them.

  Args:
      func: The user function to execute remotely.
      args: Positional arguments (Data objects should already be replaced).
      kwargs: Keyword arguments.
      env_vars: Environment variables to set on the remote pod.
      output_path: Destination path for the pickle file.
      volumes: Optional list of volume data-ref dicts.
      working_dir: Optional client-side working directory to preserve.
      package_root: Optional packaging root. Modules imported from under it
          are shipped by value so the pod does not need to import them.
      payload_extra: Optional extra payload keys, merged verbatim.

  Raises:
      ValueError: If the payload cannot be pickled. The message names the
          offending function/argument.
  """
  payload: dict[str, Any] = {
    "func": func,
    "args": args,
    "kwargs": kwargs,
    "env_vars": env_vars,
    "has_data_refs": _contains_data_ref(args) or _contains_data_ref(kwargs),
    "client_fingerprint": _client_fingerprint(),
  }
  if volumes:
    payload["volumes"] = volumes
  if working_dir:
    payload["working_dir"] = working_dir
  if package_root:
    payload["package_root"] = package_root
  if payload_extra:
    payload.update(payload_extra)

  registered = _register_by_value_modules(package_root)
  try:
    with open(output_path, "wb") as f:
      cloudpickle.dump(payload, f)
  except Exception as e:
    with contextlib.suppress(OSError):
      os.remove(output_path)
    raise _pickling_error(e, func, args, kwargs) from e
  finally:
    _unregister_by_value_modules(registered)

  _warn_on_payload_size(output_path)


def extract_data_refs(
  args: tuple, kwargs: dict[str, Any]
) -> list[tuple[Data, PositionPath]]:
  """Scan args and kwargs for Data objects at any nesting depth.

  Returns a list of ``(data_obj, position_path)`` tuples, one per *unique*
  Data object (identity-deduplicated across the whole call, so a Data reused
  in several places is uploaded and hashed once). The position path encodes
  where each Data object was first found, e.g. ``("arg", 0)`` or
  ``("kwarg", "config", "data")``.

  Circular references are handled safely via an ``id()``-based visited set.

  Raises:
      ValueError: If a Data object is used as a dict key or lives inside a
          set/frozenset — neither survives the round trip, because the
          replacement ref is an unhashable dict.
  """
  refs: list[tuple[Data, PositionPath]] = []
  visited: dict[int, bool] = {}
  seen_data: set[int] = set()
  for i, arg in enumerate(args):
    _scan_for_data(arg, ("arg", i), refs, visited, seen_data)
  for key, val in kwargs.items():
    _scan_for_data(val, ("kwarg", key), refs, visited, seen_data)
  return refs


def _data_in_set_error(path: PositionPath) -> ValueError:
  """Error raised for a Data object nested inside a set or frozenset."""
  return ValueError(
    f"kinetic does not support Data objects inside sets or frozensets — "
    f"found at {_format_path(path)}. A Data object is replaced by a dict, "
    f"which is not hashable. Pass the Data as its own argument or inside a "
    f"list, tuple or dict instead."
  )


def _data_as_key_error(path: PositionPath) -> ValueError:
  """Error raised for a Data object used as a dict key."""
  return ValueError(
    f"Data objects are not supported as dict keys — found at "
    f"{_format_path(path)}. Pass the Data as a dict value instead."
  )


def _scan_for_data(
  obj: Any,
  path: PositionPath,
  refs: list[tuple[Data, PositionPath]],
  visited: dict[int, bool] | None = None,
  seen_data: set[int] | None = None,
) -> bool:
  """Recursively collect Data objects from a nested structure.

  Returns:
      True when a Data object is reachable from *obj*. The answer is
      memoized per container so a container reached twice reports the same
      answer without re-collecting its (already deduplicated) Data objects.
  """
  if visited is None:
    visited = {}
  if seen_data is None:
    seen_data = set()
  if isinstance(obj, Data):
    if id(obj) not in seen_data:
      seen_data.add(id(obj))
      refs.append((obj, path))
    return True
  obj_id = id(obj)
  if obj_id in visited:
    return visited[obj_id]
  found = False
  if isinstance(obj, (list, tuple)):
    visited[obj_id] = False
    for i, item in enumerate(obj):
      found |= _scan_for_data(item, path + (i,), refs, visited, seen_data)
  elif isinstance(obj, (set, frozenset)):
    visited[obj_id] = False
    for i, item in enumerate(obj):
      found |= _scan_for_data(item, path + (i,), refs, visited, seen_data)
    if found:
      raise _data_in_set_error(path)
  elif isinstance(obj, dict):
    visited[obj_id] = False
    for key, val in obj.items():
      if isinstance(key, Data):
        raise _data_as_key_error(path)
      found |= _scan_for_data(val, path + (key,), refs, visited, seen_data)
  else:
    return False
  visited[obj_id] = found
  return found


def replace_data_with_refs(
  args: tuple,
  kwargs: dict[str, Any],
  ref_map: dict[int, dict[str, Any]],
) -> tuple[tuple, dict[str, Any]]:
  """Replace Data objects in args/kwargs with serializable ref dicts.

  Container types are preserved (NamedTuple, list/tuple/dict subclasses,
  OrderedDict, defaultdict, Counter) and object identity is preserved: two
  arguments that referenced the same object still reference one shared
  object afterwards, and every occurrence of a reused Data object is
  replaced. Containers that contain no Data are returned unchanged.

  Args:
      args: Positional arguments, possibly containing Data objects.
      kwargs: Keyword arguments, possibly containing Data objects.
      ref_map: Mapping from ``id(Data)`` to the replacement ref dict.

  Returns:
      ``(new_args, new_kwargs)`` with all matched Data objects replaced.
  """
  memo: dict[int, Any] = {}
  new_args = tuple(
    _replace_in_value(a, ref_map, memo, ("arg", i)) for i, a in enumerate(args)
  )
  new_kwargs = {
    k: _replace_in_value(v, ref_map, memo, ("kwarg", k))
    for k, v in kwargs.items()
  }
  return new_args, new_kwargs


def _try_rebuild(factory: Callable, expected: int) -> Any:
  """Call *factory*, returning the result only when it kept every item.

  A subclass whose ``__init__`` takes an unrelated positional argument
  accepts the item collection without storing it, which would silently
  deliver an empty container to the remote function.
  """
  try:
    rebuilt = factory()
    if len(rebuilt) == expected:
      return rebuilt
  except Exception:
    # Subclass constructors run arbitrary user code; any failure here
    # falls back to the base container type rather than aborting submit.
    pass
  return None


def _rebuild_sequence(obj: Any, items: list, path: PositionPath) -> Any:
  """Rebuild a list/tuple (including subclasses and NamedTuples)."""
  if type(obj) is list:
    return items
  if type(obj) is tuple:
    return tuple(items)
  if isinstance(obj, tuple) and hasattr(obj, "_fields"):
    rebuilt = _try_rebuild(lambda: type(obj)(*items), len(items))
  else:
    rebuilt = _try_rebuild(lambda: type(obj)(items), len(items))
  if rebuilt is not None:
    return rebuilt
  fallback = tuple(items) if isinstance(obj, tuple) else items
  logging.warning(
    "Could not rebuild %s at %s after replacing a Data object; it is passed "
    "as a plain %s instead.",
    type(obj).__name__,
    _format_path(path),
    type(fallback).__name__,
  )
  return fallback


def _updated(mapping: Any, items: dict) -> Any:
  """``mapping.update(items)`` as an expression, for :func:`_try_rebuild`."""
  mapping.update(items)
  return mapping


def _rebuild_mapping(obj: Any, items: dict, path: PositionPath) -> Any:
  """Rebuild a dict, preserving subclass identity and state where possible."""
  if type(obj) is dict:
    return items
  if isinstance(obj, defaultdict):
    rebuilt = _try_rebuild(
      lambda: _updated(type(obj)(obj.default_factory), items), len(items)
    )
  else:
    rebuilt = _try_rebuild(lambda: type(obj)(items), len(items))
    if rebuilt is None:
      rebuilt = _try_rebuild(lambda: _updated(type(obj)(), items), len(items))
  if rebuilt is not None:
    return rebuilt
  logging.warning(
    "Could not rebuild %s at %s after replacing a Data object; it is passed "
    "as a plain dict instead.",
    type(obj).__name__,
    _format_path(path),
  )
  return items


def _replace_in_value(
  obj: Any,
  ref_map: dict[int, dict[str, Any]],
  memo: dict[int, Any] | None = None,
  path: PositionPath = (),
) -> Any:
  """Recursively replace Data objects with their ref dicts.

  *memo* maps ``id(original) -> rebuilt`` so shared sub-objects are rebuilt
  once and stay shared. Mutable containers are memoized before their items
  are walked, which makes self-referential lists and dicts round-trip;
  cycles running through a tuple or set cannot be reconstructed and raise.
  """
  if memo is None:
    memo = {}
  if isinstance(obj, Data):
    return ref_map.get(id(obj), obj)
  obj_id = id(obj)
  if obj_id in memo:
    if memo[obj_id] is _IN_PROGRESS:
      raise ValueError(
        f"kinetic cannot serialize self-referential arguments that loop "
        f"through a tuple or set — found at {_format_path(path)}."
      )
    return memo[obj_id]

  if isinstance(obj, list):
    rebuilt_items: list = []
    memo[obj_id] = rebuilt_items
    changed = False
    for i, item in enumerate(obj):
      replaced = _replace_in_value(item, ref_map, memo, path + (i,))
      changed = changed or replaced is not item
      rebuilt_items.append(replaced)
    if not changed:
      memo[obj_id] = obj
      return obj
    result = _rebuild_sequence(obj, rebuilt_items, path)
    memo[obj_id] = result
    return result

  if isinstance(obj, (tuple, set, frozenset)):
    memo[obj_id] = _IN_PROGRESS
    items = []
    changed = False
    for i, item in enumerate(obj):
      replaced = _replace_in_value(item, ref_map, memo, path + (i,))
      changed = changed or replaced is not item
      items.append(replaced)
    if not changed:
      memo[obj_id] = obj
      return obj
    if isinstance(obj, (set, frozenset)):
      raise _data_in_set_error(path)
    result = _rebuild_sequence(obj, items, path)
    memo[obj_id] = result
    return result

  if isinstance(obj, dict):
    rebuilt_map: dict = {}
    memo[obj_id] = rebuilt_map
    changed = False
    for key, val in obj.items():
      if isinstance(key, Data):
        raise _data_as_key_error(path)
      replaced = _replace_in_value(val, ref_map, memo, path + (key,))
      changed = changed or replaced is not val
      rebuilt_map[key] = replaced
    if not changed:
      memo[obj_id] = obj
      return obj
    result = _rebuild_mapping(obj, rebuilt_map, path)
    memo[obj_id] = result
    return result

  return obj
