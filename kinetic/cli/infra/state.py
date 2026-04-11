"""Centralized Pulumi state loading and updates for kinetic.

Provides a single load_state() that reads ALL state dimensions, and a single
apply_update() that runs a Pulumi update with a complete InfraConfig. This
prevents commands from accidentally omitting state (e.g. forgetting to load
namespaces), which would cause Pulumi to delete the omitted resources.
"""

from dataclasses import dataclass, field

import click
import pulumi.automation as auto

from kinetic.cli.config import InfraConfig, NodePoolConfig
from kinetic.cli.constants import DEFAULT_CLUSTER_NAME, DEFAULT_ZONE
from kinetic.cli.infra.program import create_program
from kinetic.cli.infra.stack_manager import (
  get_current_node_pools,
  get_stack,
)
from kinetic.cli.output import LiveOutputPanel, console, success, warning
from kinetic.cli.prerequisites_check import check_all
from kinetic.cli.prompts import resolve_project


@dataclass
class StackState:
  """Complete state loaded from a Pulumi stack."""

  project: str
  zone: str
  cluster_name: str
  node_pools: list[NodePoolConfig] = field(default_factory=list)
  stack: auto.Stack | None = None


def load_state(
  project,
  zone,
  cluster_name,
  *,
  allow_missing=False,
  check_prerequisites=True,
):
  """Load full infrastructure state from the Pulumi stack.

  Args:
      project: GCP project ID (or None to resolve interactively).
      zone: GCP zone (or None for default).
      cluster_name: GKE cluster name (or None for default).
      allow_missing: If True, return empty state when no stack exists
          instead of raising an error. Useful for first-run scenarios.
      check_prerequisites: If True, run prerequisite checks (gcloud, etc.).

  Returns:
      A StackState with all state dimensions populated.

  Raises:
      click.ClickException: If no stack exists and allow_missing is False.
  """
  if check_prerequisites:
    check_all()

  project = project or resolve_project()
  zone = zone or DEFAULT_ZONE
  cluster_name = cluster_name or DEFAULT_CLUSTER_NAME

  base_config = InfraConfig(
    project=project, zone=zone, cluster_name=cluster_name
  )

  try:
    program = create_program(base_config)
    stack = get_stack(program, base_config)
  except auto.errors.CommandError as e:
    if allow_missing:
      return StackState(project=project, zone=zone, cluster_name=cluster_name)
    raise click.ClickException(
      f"No Pulumi stack found for project '{project}': {e}\n"
      "Run 'kinetic up' to provision infrastructure first."
    ) from e

  refresh_failed = False
  with LiveOutputPanel("Refreshing state", transient=True) as panel:
    try:
      stack.refresh(on_output=panel.on_output)
    except auto.errors.CommandError:
      panel.mark_error()
      refresh_failed = True
  if refresh_failed:
    warning("State refresh encountered an issue (using cached state).")

  node_pools = get_current_node_pools(stack)

  return StackState(
    project=project,
    zone=zone,
    cluster_name=cluster_name,
    node_pools=node_pools,
    stack=stack,
  )


def apply_update(config):
  """Run a Pulumi update with the given complete config.

  Args:
      config: A fully populated InfraConfig.

  Returns:
      True if the update succeeded, False otherwise.
  """
  program = create_program(config)
  stack = get_stack(program, config)

  ok = True
  with LiveOutputPanel("Updating infrastructure", transient=True) as panel:
    try:
      result = stack.up(on_output=panel.on_output)
    except auto.errors.CommandError:
      panel.mark_error()
      ok = False

  console.print()
  if ok:
    success(
      f"Infrastructure update complete. {result.summary.resource_changes}"
    )
    return True
  warning("Infrastructure update encountered an issue.")
  return False


def _format_changes(changes):
  """Format a Pulumi change summary dict into a readable string."""
  parts = []
  for action in ("create", "update", "delete", "replace"):
    count = changes.get(action, 0)
    if count:
      parts.append(f"{count} to {action}")
  same = changes.get("same", 0)
  if same:
    parts.append(f"{same} unchanged")
  return ", ".join(parts) if parts else "no changes"


def apply_preview(config):
  """Run a Pulumi preview with the given complete config.

  Args:
      config: A fully populated InfraConfig.

  Returns:
      True if the preview succeeded, False otherwise.
  """
  program = create_program(config)
  stack = get_stack(program, config)

  ok = True
  with LiveOutputPanel("Previewing infrastructure changes") as panel:
    try:
      result = stack.preview(on_output=panel.on_output)
    except auto.errors.CommandError as e:
      panel.mark_error()
      ok = False
      preview_error = e

  console.print()
  if ok:
    summary = _format_changes(result.change_summary)
    success(f"Preview complete: {summary}.")
    console.print(
      "\nNo changes were applied. Run without [bold]--preview[/bold] to apply."
    )
    return True
  warning(f"Preview failed: {preview_error}")
  return False


def apply_destroy(config):
  """Destroy all Pulumi-managed resources for the given config.

  Args:
      config: A fully populated InfraConfig.

  Returns:
      True if the destroy succeeded, False otherwise.
  """
  program = create_program(config)
  stack = get_stack(program, config)

  ok = True
  with LiveOutputPanel("Destroying infrastructure", transient=True) as panel:
    try:
      result = stack.destroy(on_output=panel.on_output)
    except auto.errors.CommandError:
      panel.mark_error()
      ok = False

  console.print()
  if ok:
    success(
      f"Infrastructure destroy complete. {result.summary.resource_changes}"
    )
    return True
  warning("Infrastructure destroy encountered an issue.")
  return False
