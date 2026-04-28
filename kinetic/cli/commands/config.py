"""kinetic config command — show/set configuration."""

import os

import click
from rich.table import Table

from kinetic.cli import settings
from kinetic.cli.constants import (
  DEFAULT_CLUSTER_NAME,
  DEFAULT_ZONE,
  STATE_DIR,
)
from kinetic.cli.infra.state_backend import (
  normalize_state_backend_url,
  resolve_state_backend_for_show,
)
from kinetic.cli.output import banner, console, success


@click.group(invoke_without_command=True)
@click.pass_context
def config(ctx):
  """Show or manage kinetic configuration."""
  if ctx.invoked_subcommand is None:
    ctx.invoke(show)


def _resolve(env_var, profile_value, default):
  """Return (value, source) following the CLI precedence chain.

  CLI flag is not visible to `config show`, so the effective precedence
  reported here is: env var > active profile > built-in default.
  """
  env_val = os.environ.get(env_var)
  if env_val:
    return env_val, env_var
  if profile_value is not None:
    return profile_value, "profile"
  return default, f"default ({default})" if default else ""


@config.command()
@click.pass_context
def show(ctx):
  """Show current configuration."""
  banner("kinetic Configuration")

  # Root group resolves the active profile (respecting --profile, env,
  # and stored 'current') and stashes it in ctx.obj.
  active = None
  if ctx.obj:
    active = ctx.obj.get("active_profile")

  if active is not None:
    console.print(f"Active profile: [bold]{active.name}[/bold]")
  else:
    console.print("[dim]No active profile.[/dim]")

  table = Table()
  table.add_column("Setting", style="bold")
  table.add_column("Value", style="green")
  table.add_column("Source", style="dim")

  project, src = _resolve(
    "KINETIC_PROJECT",
    active.project if active else None,
    None,
  )
  table.add_row("Project", project or "(not set)", src or "")

  zone, src = _resolve(
    "KINETIC_ZONE",
    active.zone if active else None,
    DEFAULT_ZONE,
  )
  table.add_row("Zone", zone, src)

  cluster, src = _resolve(
    "KINETIC_CLUSTER",
    active.cluster if active else None,
    DEFAULT_CLUSTER_NAME,
  )
  table.add_row("Cluster Name", cluster, src)

  namespace, src = _resolve(
    "KINETIC_NAMESPACE",
    active.namespace if active else None,
    "default",
  )
  table.add_row("Namespace", namespace, src)

  # Output directory
  output_dir = os.environ.get("KINETIC_OUTPUT_DIR")
  table.add_row(
    "Output Dir",
    output_dir or "(not set)",
    "KINETIC_OUTPUT_DIR" if output_dir else "",
  )

  # State backend (flag-less precedence: env > profile > settings > default)
  backend_url, backend_src = resolve_state_backend_for_show(
    active, project or ""
  )
  table.add_row("State Backend", backend_url, backend_src)

  # Local state directory — only useful when the resolved backend is file://.
  if backend_url.startswith("file://"):
    state_dir = os.environ.get("KINETIC_STATE_DIR")
    table.add_row(
      "Pulumi State Dir",
      state_dir or STATE_DIR,
      "KINETIC_STATE_DIR" if state_dir else "default",
    )

  console.print()
  console.print(table)
  console.print()
  console.print(
    "Precedence: CLI flag > KINETIC_* env var > active profile > "
    "global setting > default."
  )
  console.print(
    "  https://kinetic.readthedocs.io/en/latest/configuration.html#precedence"
  )
  console.print("Manage profiles with 'kinetic profile create|ls|use|show|rm'.")
  console.print(
    "Manage global settings with 'kinetic config set|unset <key> [value]'."
  )
  console.print()


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_context
def config_set(ctx, key, value):
  """Persist a global setting in ~/.kinetic/settings.json.

  Known keys:
      state-backend    Pulumi state backend (local | gcs | gs://bucket[/prefix])
  """
  internal_key = _to_internal_key(key)

  if internal_key == "state_backend":
    # Use a placeholder project so the "gcs" sentinel passes validation
    # even when no profile is active. The real project substitution
    # happens at command time, when the user actually invokes `up`/`pool`/
    # etc. with a resolved project.
    try:
      normalize_state_backend_url(value, "_placeholder")
    except click.BadParameter as e:
      raise click.ClickException(e.message) from e

  try:
    settings.set_(internal_key, value)
  except settings.SettingsError as e:
    raise click.ClickException(str(e)) from e
  success(f"Saved global setting: {key} = {value!r}")


@config.command("unset")
@click.argument("key")
def config_unset(key):
  """Remove a global setting from ~/.kinetic/settings.json."""
  internal_key = _to_internal_key(key)
  try:
    settings.unset(internal_key)
  except settings.SettingsError as e:
    raise click.ClickException(str(e)) from e
  success(f"Removed global setting: {key}")


_KEY_ALIASES = {
  "state-backend": "state_backend",
  "state_backend": "state_backend",
}


def _to_internal_key(key):
  """Map a user-facing key (CLI dash form) to the internal underscore form."""
  internal = _KEY_ALIASES.get(key)
  if internal is None:
    raise click.BadParameter(
      f"Unknown setting {key!r}. Known: {sorted(_KEY_ALIASES)}",
      param_hint="KEY",
    )
  return internal
