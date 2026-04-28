"""kinetic status command — show current infrastructure state."""

import click

from kinetic.cli.infra.state import load_state
from kinetic.cli.options import infra_options
from kinetic.cli.output import (
  banner,
  console,
  infrastructure_state,
  warning,
)


@click.command()
@infra_options
def status(project, zone, cluster_name, state_backend):
  """Show current kinetic infrastructure state."""
  banner("kinetic Status")

  state = load_state(
    project,
    zone,
    cluster_name,
    allow_missing=True,
    state_backend=state_backend,
  )

  if state.stack is None:
    warning("No Pulumi stack found.")
    console.print("Run 'kinetic up' to provision infrastructure.")
    return

  outputs = state.stack.outputs()

  if not outputs:
    warning("No infrastructure found. Run 'kinetic up' first.")
    return

  infrastructure_state(outputs)
