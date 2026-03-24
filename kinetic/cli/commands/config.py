"""kinetic config command — show/set configuration."""

import os

import click
from rich.table import Table

from kinetic.cli.constants import (
  DEFAULT_CLUSTER_NAME,
  DEFAULT_ZONE,
  STATE_DIR,
)
from kinetic.cli.output import banner, console


@click.group(invoke_without_command=True)
@click.pass_context
def config(ctx):
  """Show or manage kinetic configuration."""
  if ctx.invoked_subcommand is None:
    ctx.invoke(show)


@config.command()
def show():
  """Show current configuration."""
  banner("kinetic Configuration")

  table = Table()
  table.add_column("Setting", style="bold")
  table.add_column("Value", style="green")
  table.add_column("Source", style="dim")

  # Project
  project = os.environ.get("KERAS_REMOTE_PROJECT")
  table.add_row(
    "Project",
    project or "(not set)",
    "KERAS_REMOTE_PROJECT" if project else "",
  )

  # Zone
  zone = os.environ.get("KERAS_REMOTE_ZONE")
  table.add_row(
    "Zone",
    zone or DEFAULT_ZONE,
    "KERAS_REMOTE_ZONE" if zone else f"default ({DEFAULT_ZONE})",
  )

  # Cluster name
  cluster = os.environ.get("KERAS_REMOTE_CLUSTER")
  table.add_row(
    "Cluster Name",
    cluster or DEFAULT_CLUSTER_NAME,
    "KERAS_REMOTE_CLUSTER" if cluster else f"default ({DEFAULT_CLUSTER_NAME})",
  )

  # Namespace
  namespace = os.environ.get("KERAS_REMOTE_NAMESPACE")
  table.add_row(
    "Namespace",
    namespace or "default",
    "KERAS_REMOTE_NAMESPACE" if namespace else "default (default)",
  )

  # State directory
  state_dir = os.environ.get("KERAS_REMOTE_STATE_DIR")
  table.add_row(
    "Pulumi State Dir",
    state_dir or STATE_DIR,
    "KERAS_REMOTE_STATE_DIR" if state_dir else "default",
  )

  console.print()
  console.print(table)
  console.print()
  console.print("Set values via environment variables:")
  console.print("  export KERAS_REMOTE_PROJECT=my-project")
  console.print(f"  export KERAS_REMOTE_ZONE={DEFAULT_ZONE}")
  console.print("  export KERAS_REMOTE_CLUSTER=kinetic-cluster")
  console.print("  export KERAS_REMOTE_NAMESPACE=my-namespace")
  console.print()
