"""keras-remote status command — show current infrastructure state."""

import click

from keras_remote.cli.infra.state import load_state
from keras_remote.cli.options import common_options
from keras_remote.cli.output import (
  banner,
  console,
  infrastructure_state,
  warning,
)


@click.command()
@common_options
def status(project, zone, cluster_name):
  """Show current keras-remote infrastructure state."""
  banner("keras-remote Status")

  state = load_state(project, zone, cluster_name, allow_missing=True)

  if state.stack is None:
    warning("No Pulumi stack found.")
    console.print("Run 'keras-remote up' to provision infrastructure.")
    return

  outputs = state.stack.outputs()

  if not outputs:
    warning("No infrastructure found. Run 'keras-remote up' first.")
    return

  infrastructure_state(outputs)
