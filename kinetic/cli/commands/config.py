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
  project = os.environ.get("KINETIC_PROJECT")
  table.add_row(
    "Project",
    project or "(not set)",
    "KINETIC_PROJECT" if project else "",
  )

  # Zone
  zone = os.environ.get("KINETIC_ZONE")
  table.add_row(
    "Zone",
    zone or DEFAULT_ZONE,
    "KINETIC_ZONE" if zone else f"default ({DEFAULT_ZONE})",
  )

  # Cluster name
  cluster = os.environ.get("KINETIC_CLUSTER")
  table.add_row(
    "Cluster Name",
    cluster or DEFAULT_CLUSTER_NAME,
    "KINETIC_CLUSTER" if cluster else f"default ({DEFAULT_CLUSTER_NAME})",
  )

  # Namespace
  namespace = os.environ.get("KINETIC_NAMESPACE")
  table.add_row(
    "Namespace",
    namespace or "default",
    "KINETIC_NAMESPACE" if namespace else "default (default)",
  )

  # Output directory
  output_dir = os.environ.get("KINETIC_OUTPUT_DIR")
  table.add_row(
    "Output Dir",
    output_dir or "(not set)",
    "KINETIC_OUTPUT_DIR" if output_dir else "",
  )

  # State directory
  state_dir = os.environ.get("KINETIC_STATE_DIR")
  table.add_row(
    "Pulumi State Dir",
    state_dir or STATE_DIR,
    "KINETIC_STATE_DIR" if state_dir else "default",
  )

  console.print()
  console.print(table)
  console.print()
  console.print("Set values via environment variables:")
  console.print("  export KINETIC_PROJECT=my-project")
  console.print(f"  export KINETIC_ZONE={DEFAULT_ZONE}")
  console.print("  export KINETIC_CLUSTER=kinetic-cluster")
  console.print("  export KINETIC_NAMESPACE=my-namespace")
  console.print()
