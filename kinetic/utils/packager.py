"""Packaging utilities for serializing functions, args, and working directories.

Handles zipping the user's working directory, serializing the function
payload with cloudpickle, and extracting/replacing Data objects in
arbitrarily nested arg structures.
"""

import os
import posixpath
import subprocess
import zipfile
from collections.abc import Callable
from typing import Any

import cloudpickle

from kinetic.data import Data

# Type alias for a position path through nested args, e.g. ("arg", 0, "key").
PositionPath = tuple[str | int, ...]


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
    zipf.write(file_path, archive_name)


def zip_working_dir(
  base_dir: str, output_path: str, exclude_paths: set[str] | None = None
) -> None:
  """Zip a directory into a ZIP archive, respecting .gitignore.

  Excludes ``.git``, ``__pycache__``, and any paths in *exclude_paths*
  (which may be files or directories). Uses git ls-files when available
  to respect .gitignore, otherwise falls back to directory traversal.

  Args:
      base_dir: Root directory to zip.
      output_path: Destination path for the ZIP file.
      exclude_paths: Absolute paths to skip during archiving.
  """
  exclude_paths = exclude_paths or set()

  with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    # Try to use git ls-files to respect .gitignore
    git_files = _list_git_files(base_dir)
    if git_files is not None:
      _write_git_files(zipf, base_dir, git_files, exclude_paths)
      return

    # Fallback to directory traversal if git is not available
    normalized_excludes = {os.path.normpath(p) for p in exclude_paths}
    for root, dirs, files in os.walk(base_dir):
      # Exclude .git, __pycache__, and Data-referenced directories
      dirs[:] = [
        d
        for d in dirs
        if d not in [".git", "__pycache__"]
        and os.path.normpath(os.path.join(root, d)) not in normalized_excludes
      ]

      for file in files:
        file_path = os.path.join(root, file)
        if os.path.normpath(file_path) in normalized_excludes:
          continue
        archive_name = os.path.relpath(file_path, base_dir)
        zipf.write(file_path, archive_name)


def save_payload(
  func: Callable,
  args: tuple,
  kwargs: dict[str, Any],
  env_vars: dict[str, str],
  output_path: str,
  volumes: list[dict[str, Any]] | None = None,
  working_dir: str | None = None,
) -> None:
  """Serialize a function call payload with cloudpickle.

  The resulting pickle file contains a dict with keys ``func``, ``args``,
  ``kwargs``, ``env_vars``, and optionally ``volumes``.

  Args:
      func: The user function to execute remotely.
      args: Positional arguments (Data objects should already be replaced).
      kwargs: Keyword arguments.
      env_vars: Environment variables to set on the remote pod.
      output_path: Destination path for the pickle file.
      volumes: Optional list of volume data-ref dicts.
      working_dir: Optional client-side working directory to preserve.
  """
  payload: dict[str, Any] = {
    "func": func,
    "args": args,
    "kwargs": kwargs,
    "env_vars": env_vars,
  }
  if volumes:
    payload["volumes"] = volumes
  if working_dir:
    payload["working_dir"] = working_dir
  with open(output_path, "wb") as f:
    cloudpickle.dump(payload, f)


def extract_data_refs(
  args: tuple, kwargs: dict[str, Any]
) -> list[tuple[Data, PositionPath]]:
  """Scan args and kwargs for Data objects at any nesting depth.

  Returns a list of ``(data_obj, position_path)`` tuples. The position
  path encodes where each Data object was found, e.g.
  ``("arg", 0)`` or ``("kwarg", "config", "data")``.

  Circular references are handled safely via an ``id()``-based visited
  set.
  """
  refs: list[tuple[Data, PositionPath]] = []
  for i, arg in enumerate(args):
    _scan_for_data(arg, ("arg", i), refs)
  for key, val in kwargs.items():
    _scan_for_data(val, ("kwarg", key), refs)
  return refs


def _scan_for_data(
  obj: Any,
  path: PositionPath,
  refs: list[tuple[Data, PositionPath]],
  visited: set[int] | None = None,
) -> None:
  """Recursively collect Data objects from a nested structure."""
  if visited is None:
    visited = set()
  obj_id = id(obj)
  if obj_id in visited:
    return
  visited.add(obj_id)
  if isinstance(obj, Data):
    refs.append((obj, path))
  elif isinstance(obj, (list, tuple, set, frozenset)):
    for i, item in enumerate(obj):
      _scan_for_data(item, path + (i,), refs, visited)
  elif isinstance(obj, dict):
    for key, val in obj.items():
      _scan_for_data(val, path + (key,), refs, visited)


def replace_data_with_refs(
  args: tuple,
  kwargs: dict[str, Any],
  ref_map: dict[int, dict[str, Any]],
) -> tuple[tuple, dict[str, Any]]:
  """Replace Data objects in args/kwargs with serializable ref dicts.

  Args:
      args: Positional arguments, possibly containing Data objects.
      kwargs: Keyword arguments, possibly containing Data objects.
      ref_map: Mapping from ``id(Data)`` to the replacement ref dict.

  Returns:
      ``(new_args, new_kwargs)`` with all matched Data objects replaced.
  """
  new_args = tuple(_replace_in_value(a, ref_map) for a in args)
  new_kwargs = {k: _replace_in_value(v, ref_map) for k, v in kwargs.items()}
  return new_args, new_kwargs


def _replace_in_value(
  obj: Any,
  ref_map: dict[int, dict[str, Any]],
  visited: set[int] | None = None,
) -> Any:
  """Recursively replace Data objects with their ref dicts."""
  if visited is None:
    visited = set()
  obj_id = id(obj)
  if obj_id in visited:
    return obj
  visited.add(obj_id)
  if isinstance(obj, Data) and obj_id in ref_map:
    return ref_map[obj_id]
  elif isinstance(obj, list):
    return [_replace_in_value(item, ref_map, visited) for item in obj]
  elif isinstance(obj, tuple):
    return tuple(_replace_in_value(item, ref_map, visited) for item in obj)
  elif isinstance(obj, (set, frozenset)):
    return [_replace_in_value(item, ref_map, visited) for item in obj]
  elif isinstance(obj, dict):
    return {k: _replace_in_value(v, ref_map, visited) for k, v in obj.items()}
  return obj
