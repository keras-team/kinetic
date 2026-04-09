"""Internal helpers for kinetic.collections.

Input-expansion utilities and manifest construction live here to keep
collections.py focused on the public BatchHandle / map / attach_batch
API.
"""

import keyword
from typing import Any

from kinetic.jobs import JobHandle, _utcnow_iso


def is_valid_kwargs_dict(item: Any) -> bool:
  """Check if a dict's keys are all valid Python identifiers."""
  if not isinstance(item, dict):
    return False
  return all(
    isinstance(k, str) and k.isidentifier() and not keyword.iskeyword(k)
    for k in item
  )


def call_with_input(submit_fn, item, input_mode: str) -> JobHandle:
  """Call *submit_fn* with the appropriate argument unpacking."""
  if input_mode == "single":
    return submit_fn(item)
  elif input_mode == "args":
    if not isinstance(item, (list, tuple)):
      raise TypeError(
        f"input_mode='args' requires list or tuple items, "
        f"got {type(item).__name__}"
      )
    return submit_fn(*item)
  elif input_mode == "kwargs":
    if not isinstance(item, dict):
      raise TypeError(
        f"input_mode='kwargs' requires dict items, got {type(item).__name__}"
      )
    return submit_fn(**item)
  elif input_mode == "auto":
    if isinstance(item, dict) and is_valid_kwargs_dict(item):
      return submit_fn(**item)
    elif isinstance(item, (list, tuple)):
      return submit_fn(*item)
    else:
      return submit_fn(item)
  else:
    raise ValueError(f"Unknown input_mode: {input_mode!r}")


def build_initial_manifest(
  group_id: str,
  group_kind: str,
  group_name: str | None,
  tags: dict[str, str] | None,
  total_expected: int,
  submit_fn_name: str,
) -> dict:
  """Build the initial manifest dict with an empty children list."""
  return {
    "group_id": group_id,
    "group_kind": group_kind,
    "group_name": group_name,
    "tags": tags or {},
    "created_at": _utcnow_iso(),
    "total_expected": total_expected,
    "submit_fn_name": submit_fn_name,
    "children": [],
  }


def append_child_to_manifest(
  manifest: dict,
  group_index: int,
  job_id: str,
  attempts: int = 1,
) -> None:
  """Add or update a child entry in the manifest (in-place)."""
  for child in manifest["children"]:
    if child["group_index"] == group_index:
      child["job_id"] = job_id
      child["attempts"] = attempts
      return
  manifest["children"].append(
    {
      "group_index": group_index,
      "job_id": job_id,
      "attempts": attempts,
    }
  )
