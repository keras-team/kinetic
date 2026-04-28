"""Persisted, kinetic-wide settings that are not tied to a profile.

Holds defaults like ``state_backend`` that should survive across all profile
operations. Stored at ``~/.kinetic/settings.json`` (overridable via
``KINETIC_SETTINGS_FILE``) with a flat ``{key: value}`` shape:

    {
      "state_backend": "gcs"
    }

Kept separate from ``profiles.json`` so wiping profiles does not also wipe
user-global preferences.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path

from kinetic.cli.constants import SETTINGS_FILE

SETTINGS_KEYS = {"state_backend"}


class SettingsError(Exception):
  """Raised for settings I/O or validation errors."""


def _settings_path() -> Path:
  """Resolve the settings file path, honoring env override at call time."""
  override = os.environ.get("KINETIC_SETTINGS_FILE")
  return Path(override) if override else Path(SETTINGS_FILE)


def load() -> dict:
  """Load all settings. Returns {} if the file is missing.

  Raises SettingsError if the file exists but is malformed.
  """
  path = _settings_path()
  if not path.exists():
    return {}
  try:
    with path.open("r", encoding="utf-8") as f:
      data = json.load(f)
  except (OSError, json.JSONDecodeError) as e:
    raise SettingsError(f"Failed to read {path}: {e}") from e
  if not isinstance(data, dict):
    raise SettingsError(f"Malformed settings file {path}: expected object")
  return data


def get(key: str) -> str | None:
  """Convenience wrapper: return the value for ``key`` or None."""
  value = load().get(key)
  if value is None:
    return None
  if not isinstance(value, str):
    raise SettingsError(
      f"Malformed settings file: {key!r} must be a string, got {type(value).__name__}"
    )
  return value


def set_(key: str, value: str) -> None:
  """Persist ``key=value``. Atomic write. Rejects unknown keys."""
  if key not in SETTINGS_KEYS:
    raise SettingsError(
      f"Unknown setting {key!r}. Known settings: {sorted(SETTINGS_KEYS)}"
    )
  if not isinstance(value, str):
    raise SettingsError(f"Setting {key!r} must be a string.")
  data = load()
  data[key] = value
  _save(data)


def unset(key: str) -> None:
  """Remove ``key`` from settings. No-op if not present."""
  data = load()
  if key not in data:
    return
  del data[key]
  _save(data)


def _save(data: dict) -> None:
  """Atomically write the settings store to disk."""
  path = _settings_path()
  path.parent.mkdir(parents=True, exist_ok=True)

  fd, tmp_path = tempfile.mkstemp(
    prefix=".settings-", suffix=".json.tmp", dir=str(path.parent)
  )
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
      json.dump(data, f, indent=2, sort_keys=True)
      f.write("\n")
    os.replace(tmp_path, path)
  except Exception:
    with contextlib.suppress(OSError):
      os.unlink(tmp_path)
    raise
