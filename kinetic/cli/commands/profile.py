"""kinetic profile commands — manage named infrastructure-target profiles."""

import click
from rich.table import Table

from kinetic.cli.constants import DEFAULT_CLUSTER_NAME, DEFAULT_ZONE
from kinetic.cli.output import banner, console, error, success, warning
from kinetic.cli.profiles import (
  Profile,
  ProfileError,
  get_profile,
  list_profiles,
  load_store,
  remove_profile,
  set_current,
  upsert_profile,
  validate_name,
)


@click.group()
def profile():
  """Manage named kinetic profiles (project/zone/cluster/namespace bundles)."""


@profile.command("create")
@click.argument("name", required=False)
@click.option(
  "--project",
  envvar="KINETIC_PROJECT",
  default=None,
  help="GCP project ID [env: KINETIC_PROJECT]",
)
@click.option(
  "--zone",
  envvar="KINETIC_ZONE",
  default=None,
  help="GCP zone [env: KINETIC_ZONE]",
)
@click.option(
  "--cluster",
  "cluster_name",
  envvar="KINETIC_CLUSTER",
  default=None,
  help="GKE cluster name [env: KINETIC_CLUSTER]",
)
@click.option(
  "--namespace",
  envvar="KINETIC_NAMESPACE",
  default=None,
  help="Kubernetes namespace [env: KINETIC_NAMESPACE]",
)
@click.option(
  "--force",
  is_flag=True,
  help="Overwrite an existing profile with the same name.",
)
def profile_create(name, project, zone, cluster_name, namespace, force):
  """Create a new profile.

  Any unset field is resolved in this order:
      --<flag>  >  KINETIC_* env var  >  interactive prompt.
  """
  banner("Create kinetic profile")

  if name is None:
    name = click.prompt("Profile name", type=str)
  try:
    validate_name(name)
  except ProfileError as e:
    raise click.BadParameter(str(e), param_hint="NAME") from e

  _, existing = load_store()
  if name in existing and not force:
    raise click.ClickException(
      f"Profile {name!r} already exists. Use --force to overwrite "
      "or 'kinetic profile rm' to delete it first."
    )

  if project is None:
    project = click.prompt("GCP project ID", type=str)
  if zone is None:
    zone = click.prompt("GCP zone", default=DEFAULT_ZONE, type=str)
  if cluster_name is None:
    cluster_name = click.prompt(
      "GKE cluster name", default=DEFAULT_CLUSTER_NAME, type=str
    )
  if namespace is None:
    namespace = click.prompt(
      "Kubernetes namespace", default="default", type=str
    )

  p = Profile(
    name=name,
    project=project,
    zone=zone,
    cluster=cluster_name,
    namespace=namespace,
  )
  became_current = upsert_profile(p)
  success(f"Saved profile '{name}'.")
  if became_current:
    console.print(f"Profile '{name}' is now active.")
  else:
    console.print(f"Run 'kinetic profile use {name}' to activate it.")


@profile.command("ls")
@click.pass_context
def profile_ls(ctx):
  """List all saved profiles."""
  try:
    stored_current, profiles = list_profiles()
  except ProfileError as e:
    error(str(e))
    raise click.exceptions.Exit(1) from e

  if not profiles:
    console.print(
      "No profiles saved. Create one with 'kinetic profile create <name>'."
    )
    return

  # If --profile / KINETIC_PROFILE selected a profile for this invocation,
  # mark that as active instead of the stored 'current'.
  effective = ctx.obj.get("active_profile") if ctx.obj else None
  effective_name = effective.name if effective is not None else stored_current

  table = Table()
  table.add_column("", width=1)
  table.add_column("NAME", style="bold")
  table.add_column("PROJECT", style="green")
  table.add_column("ZONE")
  table.add_column("CLUSTER")
  table.add_column("NAMESPACE")
  for p in profiles:
    marker = "*" if p.name == effective_name else ""
    table.add_row(marker, p.name, p.project, p.zone, p.cluster, p.namespace)

  console.print()
  console.print(table)
  console.print()
  if effective_name:
    if effective is not None and stored_current != effective_name:
      console.print(
        f"Active profile: [bold]{effective_name}[/bold] "
        f"[dim](override; stored: {stored_current or 'none'})[/dim]"
      )
    else:
      console.print(f"Active profile: [bold]{effective_name}[/bold]")
  else:
    console.print(
      "[dim]No active profile. Run 'kinetic profile use <name>'.[/dim]"
    )


@profile.command("use")
@click.argument("name")
def profile_use(name):
  """Set ``name`` as the active profile."""
  try:
    set_current(name)
  except ProfileError as e:
    raise click.ClickException(str(e)) from e
  success(f"Active profile: {name}")
  # Echo what resolves now so the user sees the effect immediately.
  p = get_profile(name)
  _print_profile(p)


@profile.command("show")
@click.argument("name", required=False)
@click.pass_context
def profile_show(ctx, name):
  """Show a profile's settings.

  Default target: the profile selected by --profile / KINETIC_PROFILE, or
  the stored active profile if neither was provided.
  """
  try:
    if name is not None:
      p = get_profile(name)
    else:
      # Prefer the --profile / KINETIC_PROFILE selection (stashed by the
      # root group) over the persistent 'current' pointer.
      effective = ctx.obj.get("active_profile") if ctx.obj else None
      if effective is None:
        warning(
          "No active profile. Pass a NAME or run 'kinetic profile use <name>'."
        )
        raise click.exceptions.Exit(1)
      p = effective
  except ProfileError as e:
    raise click.ClickException(str(e)) from e
  _print_profile(p)


@profile.command("rm")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation.")
def profile_rm(name, yes):
  """Delete a profile."""
  try:
    p = get_profile(name)
  except ProfileError as e:
    raise click.ClickException(str(e)) from e

  if not yes:
    click.confirm(
      f"Delete profile '{name}' ({p.project}/{p.zone}/{p.cluster})?",
      abort=True,
    )

  try:
    remove_profile(name)
  except ProfileError as e:
    raise click.ClickException(str(e)) from e
  success(f"Removed profile '{name}'.")


def _print_profile(p):
  table = Table(title=f"Profile: {p.name}")
  table.add_column("Setting", style="bold")
  table.add_column("Value", style="green")
  table.add_row("Project", p.project)
  table.add_row("Zone", p.zone)
  table.add_row("Cluster", p.cluster)
  table.add_row("Namespace", p.namespace)
  console.print()
  console.print(table)
  console.print()
