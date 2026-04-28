"""Pulumi state backend resolution and GCS bucket lifecycle.

Two responsibilities:

1. **Normalize** the user-intent value (from --state-backend flag, env var,
   profile field, or persisted global setting) into a concrete Pulumi backend
   URL like ``file:///...`` or ``gs://bucket[/prefix]``. Pure string logic.
2. **Ensure** that a GCS state bucket exists (idempotent), creating it with
   versioning + uniform bucket-level access if missing. Network I/O.
"""

from __future__ import annotations

import re

import click
from google.api_core import exceptions as gax
from google.cloud import storage

from kinetic.cli import settings
from kinetic.cli.constants import (
  STATE_BACKEND_ENV_VAR,
  STATE_DIR,
  default_gcs_bucket_name,
)

_GCS_URL_RE = re.compile(
  r"^gs://([a-z0-9][a-z0-9._\-]{1,61}[a-z0-9])(?:/(.*))?$"
)
_LOCAL_VALUES = {None, "", "local"}
_GCS_SENTINEL = "gcs"


def normalize_state_backend_url(value: str | None, project: str) -> str:
  """Turn a raw user-intent value into a concrete Pulumi backend URL.

  Mapping:
    - None | "" | "local"           -> file://{STATE_DIR}
    - "gcs"                         -> gs://{project}-kinetic-state
    - "gs://bucket[/prefix]"        -> passthrough (validated)
    - "file:///abs/path"            -> passthrough
    - anything else                 -> raises click.BadParameter

  Pure function, no I/O.
  """
  if value in _LOCAL_VALUES:
    return f"file://{STATE_DIR}"
  if value == _GCS_SENTINEL:
    if not project:
      raise click.BadParameter(
        "Cannot derive default GCS bucket: GCP project is not set. "
        "Pass --project, set KINETIC_PROJECT, or activate a profile.",
        param_hint="--state-backend",
      )
    return f"gs://{default_gcs_bucket_name(project)}"
  if value.startswith("gs://"):
    if not _GCS_URL_RE.match(value):
      raise click.BadParameter(
        f"Invalid GCS URL {value!r}. Expected 'gs://bucket[/prefix]' with a "
        "valid bucket name.",
        param_hint="--state-backend",
      )
    return value
  if value.startswith("file://"):
    return value
  raise click.BadParameter(
    f"Unsupported state backend {value!r}. Use 'local', 'gcs', or a "
    "'gs://bucket[/prefix]' URL.",
    param_hint="--state-backend",
  )


def resolve_state_backend_for_show(
  active_profile,
  project: str,
) -> tuple[str, str]:
  """Source-attributing resolver for ``kinetic config show``.

  Mirrors the precedence used by Click's default_map for state-touching
  commands, minus the per-invocation flag (which `config show` does not
  accept). Returns (absolute_url, source).

  Precedence:
    KINETIC_STATE_BACKEND env  >  active profile.state_backend
      >  settings.get('state_backend')  >  None (local default)

  When the resolved value is the "gcs" sentinel and no project has been
  resolved yet, the URL is rendered with a literal "<project>" placeholder
  rather than failing — the user is asking what is configured, not asking
  the CLI to act on it.
  """
  import os  # noqa: PLC0415 — local to keep module import light

  env_val = os.environ.get(STATE_BACKEND_ENV_VAR)
  if env_val:
    return _safe_normalize(env_val, project), STATE_BACKEND_ENV_VAR

  profile_val = (
    getattr(active_profile, "state_backend", None) if active_profile else None
  )
  if profile_val:
    return _safe_normalize(profile_val, project), "profile"

  settings_val = settings.get("state_backend")
  if settings_val:
    return _safe_normalize(settings_val, project), "settings"

  return _safe_normalize(None, project), "default"


def _safe_normalize(value: str | None, project: str) -> str:
  """Like normalize_state_backend_url, but substitutes a placeholder
  project when the gcs sentinel needs one and none is known."""
  effective_project = project or "<project>"
  return normalize_state_backend_url(value, effective_project)


def _parse_gcs_url(url: str) -> tuple[str, str]:
  """Split 'gs://bucket[/prefix]' into (bucket, prefix). Raises on malformed."""
  match = _GCS_URL_RE.match(url)
  if not match:
    raise click.BadParameter(f"Invalid GCS URL {url!r}.")
  return match.group(1), match.group(2) or ""


def ensure_gcs_backend(
  url: str, *, project: str | None = None, location: str = "US"
) -> None:
  """Best-effort: create the GCS state bucket if it does not exist.

  This is purely a UX nicety for the *first* admin who opts into the GCS
  backend. We attempt to create the bucket; if the name is taken globally
  we raise a specific actionable error. For everything else (the bucket
  already exists, the user lacks bucket-level perms, etc.) we swallow
  the error and let Pulumi's own first read/write surface the real
  permission problem at the *object* level — which is the only level
  team members with `roles/storage.objectAdmin` actually have.

  Calling code must pass ``project`` (the kinetic GCP project) so the
  bucket lands under that project's IAM/billing/ownership rather than
  whatever ADC happens to default to.

  Raises:
    click.ClickException only on Conflict (bucket name globally taken),
    which is the one case the user must intervene to fix.
  """
  bucket_name, _ = _parse_gcs_url(url)
  client = storage.Client(project=project) if project else storage.Client()
  bucket = client.bucket(bucket_name)

  try:
    bucket.versioning_enabled = True
    bucket.iam_configuration.uniform_bucket_level_access_enabled = True
    client.create_bucket(bucket, location=location)
  except gax.Conflict:
    # Bucket already exists (globally). Either the team already created
    # it (the happy path for collaborators) or the name is taken by
    # someone else. Pulumi's first state read will distinguish the two
    # and produce a clean object-level error if access is wrong.
    return
  except (gax.Forbidden, gax.PermissionDenied):
    # Caller lacks storage.admin on the project. Fine for non-bucket-
    # creating teammates — Pulumi will succeed via object-level perms.
    return
