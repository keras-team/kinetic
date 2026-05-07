"""Named profiles for kinetic CLI configuration.

A profile bundles the infrastructure target fields (project, zone, cluster,
namespace) under a single name so users can switch between configurations
without re-exporting environment variables.

Storage: a single JSON file at ``~/.kinetic/profiles.json`` with the shape:

    {
      "current": "dev-tpu",
      "profiles": {
        "dev-tpu": {"project": "...", "zone": "...", ...},
        ...
      }
    }

Resolution order (handled by the CLI root group, not here):
  CLI flag  >  KINETIC_* env var  >  active profile field  >  built-in default
"""

import contextlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from kinetic.cli.constants import PROFILES_FILE

_PROFILE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,63}$")


class ProfileError(Exception):
  """Raised for profile validation or I/O errors."""


@dataclass
class Profile:
  """An infrastructure target saved under a name.

  Fields mirror the KINETIC_* env vars and InfraConfig so profile values
  can slot directly into the existing config precedence chain.
  """

  name: str
  project: str
  zone: str
  cluster: str
  namespace: str = "default"

  def to_dict(self):
    d = asdict(self)
    d.pop("name")
    return d


def validate_name(name):
  """Check that ``name`` is a valid profile identifier.

  Raises ProfileError if invalid.
  """
  if not isinstance(name, str) or not _PROFILE_NAME_RE.match(name):
    raise ProfileError(
      f"Invalid profile name {name!r}: must start with an alphanumeric "
      "character and contain only letters, digits, '-', or '_' (max 64 chars)."
    )


def _profiles_path():
  """Resolve the profiles file path, honoring env override at call time."""
  override = os.environ.get("KINETIC_PROFILES_FILE")
  return Path(override) if override else Path(PROFILES_FILE)


def load_store():
  """Load the full profile store. Returns (current, {name: Profile}).

  Missing file -> (None, {}). Malformed file raises ProfileError.
  """
  path = _profiles_path()
  if not path.exists():
    return None, {}
  try:
    with path.open("r", encoding="utf-8") as f:
      data = json.load(f)
  except (OSError, json.JSONDecodeError) as e:
    raise ProfileError(f"Failed to read {path}: {e}") from e

  if not isinstance(data, dict):
    raise ProfileError(f"Malformed profiles file {path}: expected object")

  current = data.get("current")
  if current is not None and not isinstance(current, str):
    raise ProfileError(
      f"Malformed profiles file {path}: 'current' must be a string"
    )

  raw = data.get("profiles", {})
  if not isinstance(raw, dict):
    raise ProfileError(
      f"Malformed profiles file {path}: 'profiles' must be an object"
    )

  profiles = {}
  for name, fields in raw.items():
    if not isinstance(fields, dict):
      raise ProfileError(f"Malformed profile {name!r}: expected object")
    try:
      profiles[name] = Profile(
        name=name,
        project=fields["project"],
        zone=fields["zone"],
        cluster=fields["cluster"],
        namespace=fields.get("namespace", "default"),
      )
    except KeyError as e:
      raise ProfileError(
        f"Malformed profile {name!r}: missing field {e.args[0]!r}"
      ) from e

  # Drop stale 'current' pointer rather than error out.
  if current is not None and current not in profiles:
    current = None

  return current, profiles


def _save_store(current, profiles):
  """Atomically write the profile store to disk."""
  path = _profiles_path()
  path.parent.mkdir(parents=True, exist_ok=True)

  payload = {
    "current": current,
    "profiles": {name: p.to_dict() for name, p in profiles.items()},
  }

  # Atomic write: tempfile in the same dir + rename.
  fd, tmp_path = tempfile.mkstemp(
    prefix=".profiles-", suffix=".json.tmp", dir=str(path.parent)
  )
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
      json.dump(payload, f, indent=2, sort_keys=True)
      f.write("\n")
    os.replace(tmp_path, path)
  except Exception:
    # Best-effort cleanup.
    with contextlib.suppress(OSError):
      os.unlink(tmp_path)
    raise


def list_profiles():
  """Return (current_name, list[Profile]) sorted by name."""
  current, profiles = load_store()
  return current, sorted(profiles.values(), key=lambda p: p.name)


def get_profile(name):
  """Return the named Profile, or raise ProfileError if it does not exist."""
  _, profiles = load_store()
  if name not in profiles:
    raise ProfileError(f"Profile {name!r} does not exist.")
  return profiles[name]


def get_current():
  """Return the active Profile, or None if no profile is active."""
  current, profiles = load_store()
  if current is None:
    return None
  return profiles.get(current)


def upsert_profile(profile, *, make_current_if_first=True):
  """Create or overwrite a profile. Returns True if it is the active one."""
  validate_name(profile.name)
  current, profiles = load_store()
  is_new = profile.name not in profiles
  profiles[profile.name] = profile
  if current is None and is_new and make_current_if_first:
    current = profile.name
  _save_store(current, profiles)
  return current == profile.name


def set_current(name):
  """Mark ``name`` as the active profile. Raises if it does not exist."""
  current, profiles = load_store()
  if name not in profiles:
    raise ProfileError(f"Profile {name!r} does not exist.")
  if current == name:
    return
  _save_store(name, profiles)


def remove_profile(name):
  """Delete a profile. Raises if it does not exist.

  If the removed profile was active, the active pointer is cleared.
  """
  current, profiles = load_store()
  if name not in profiles:
    raise ProfileError(f"Profile {name!r} does not exist.")
  del profiles[name]
  if current == name:
    current = None
  _save_store(current, profiles)


def resolve_active(explicit_name=None):
  """Determine which profile (if any) should apply for the current invocation.

  Precedence for *which profile*:
      explicit_name (e.g. --profile)  >  $KINETIC_PROFILE  >  stored 'current'

  Returns the Profile, or None if no active profile is selected.
  Raises ProfileError if a name is requested but does not exist.
  """
  name = explicit_name or os.environ.get("KINETIC_PROFILE")
  current, profiles = load_store()
  if name:
    if name not in profiles:
      raise ProfileError(
        f"Profile {name!r} does not exist. Run 'kinetic profile ls' to see "
        "available profiles."
      )
    return profiles[name]
  if current is None:
    return None
  return profiles.get(current)
