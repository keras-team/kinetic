"""kinetic CLI entry point."""

import click

from kinetic.cli.commands.accelerators import accelerators
from kinetic.cli.commands.build_base import build_base
from kinetic.cli.commands.config import config
from kinetic.cli.commands.doctor import doctor
from kinetic.cli.commands.down import down
from kinetic.cli.commands.jobs import jobs
from kinetic.cli.commands.pool import pool
from kinetic.cli.commands.profile import profile
from kinetic.cli.commands.status import status
from kinetic.cli.commands.up import up
from kinetic.cli.output import error
from kinetic.cli.profiles import ProfileError, resolve_active

# Param names, on commands using common_options / jobs_options, that
# receive values from the active profile. Used to inject profile fields
# as Click default_map entries — CLI flags and KINETIC_* env vars still win.
_PROFILE_TO_PARAM = {
  "project": "project",
  "zone": "zone",
  "cluster": "cluster_name",
  "namespace": "namespace",
}


@click.group()
@click.option(
  "--profile",
  "profile_name",
  envvar="KINETIC_PROFILE",
  default=None,
  help="Use a named profile for this invocation [env: KINETIC_PROFILE].",
)
@click.version_option(package_name="keras-kinetic")
@click.pass_context
def cli(ctx, profile_name):
  """kinetic: Provision and manage GCP infrastructure for remote Keras
  execution."""
  ctx.ensure_object(dict)

  try:
    active = resolve_active(explicit_name=profile_name)
  except ProfileError as e:
    error(str(e))
    raise click.exceptions.Exit(1) from e

  # Stash the resolved profile so subcommands (e.g. `config show`,
  # `profile show`, `profile ls`) can respect --profile / KINETIC_PROFILE
  # without re-resolving.
  ctx.obj["active_profile"] = active

  # The 'profile' command group must not have profile defaults injected
  # into its own options — we don't want 'profile create' to auto-fill
  # from the currently-active profile. Skip default_map, but keep the
  # resolved selection on ctx.obj above.
  if ctx.invoked_subcommand == "profile":
    return

  if active is None:
    return

  defaults = {
    param: getattr(active, field) for field, param in _PROFILE_TO_PARAM.items()
  }
  ctx.default_map = _spread_defaults(cli, defaults)


def _spread_defaults(group, defaults):
  """Build a nested default_map that applies ``defaults`` to every command.

  Click resolves defaults per command node, so for a subcommand like
  ``jobs list`` we need ``{"jobs": {"list": {...}}}``. This walks the
  group tree and copies the same defaults dict into every leaf.
  """
  result = {}
  for name, cmd in group.commands.items():
    if name == "profile":
      # Keep the profile management group free of injected defaults.
      continue
    if isinstance(cmd, click.Group):
      result[name] = _spread_defaults(cmd, defaults)
    else:
      result[name] = dict(defaults)
  return result


cli.add_command(accelerators)
cli.add_command(up)
cli.add_command(down)
cli.add_command(status)
cli.add_command(config)
cli.add_command(pool)
cli.add_command(jobs)
cli.add_command(doctor)
cli.add_command(build_base)
cli.add_command(profile)
