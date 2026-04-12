"""kinetic pool commands — add, remove, and list accelerator node pools."""

import click

from kinetic.cli.config import InfraConfig, NodePoolConfig
from kinetic.cli.infra.state import apply_update, load_state
from kinetic.cli.options import common_options
from kinetic.cli.output import (
  banner,
  console,
  infrastructure_state,
  warning,
)
from kinetic.core import accelerators
from kinetic.core.accelerators import generate_pool_name


@click.group()
def pool():
  """Manage accelerator node pools."""


@pool.command("add")
@common_options
@click.option(
  "--accelerator",
  required=True,
  help="Accelerator spec: t4, l4, a100, a100-80gb, h100, "
  "v5litepod, v5p, v6e, v3 (with optional count/topology)",
)
@click.option(
  "--min-nodes",
  default=0,
  type=int,
  help="Minimum node count for the accelerator node pool (default: 0)",
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option("--spot", is_flag=True, help="Use Spot VMs for node pool")
@click.option(
  "--reservation",
  default=None,
  envvar="KINETIC_RESERVATION",
  help="GCP capacity reservation name to consume when provisioning nodes [env: KINETIC_RESERVATION]",
)
def pool_add(
  project, zone, cluster_name, accelerator, min_nodes, yes, spot, reservation
):
  """Add an accelerator node pool to the cluster."""
  banner("kinetic Pool Add")

  # Parse the accelerator spec first to fail fast on bad input.
  try:
    accel_config = accelerators.parse_accelerator(accelerator, spot=spot)
  except ValueError as e:
    raise click.BadParameter(str(e), param_hint="--accelerator") from e

  if accel_config is None:
    raise click.BadParameter(
      "Cannot add a CPU node pool. Use 'kinetic up' instead.",
      param_hint="--accelerator",
    )

  if reservation and spot:
    raise click.BadParameter(
      "Reservations cannot be used with Spot VMs.",
      param_hint="--reservation",
    )

  new_pool_name = generate_pool_name(accel_config)
  new_pool = NodePoolConfig(
    new_pool_name, accel_config, min_nodes=min_nodes, reservation=reservation
  )

  state = load_state(project, zone, cluster_name)
  all_pools = state.node_pools + [new_pool]

  console.print(f"\nAdding pool [bold]{new_pool_name}[/bold] ({accelerator})")
  console.print(f"Total pools after add: {len(all_pools)}\n")

  if not yes:
    click.confirm("Proceed?", abort=True)

  config = InfraConfig(
    project=state.project,
    zone=state.zone,
    cluster_name=state.cluster_name,
    node_pools=all_pools,
  )
  update_succeeded = apply_update(config)

  console.print()
  if update_succeeded:
    banner("Pool Added")
  else:
    banner("Pool Update Failed")
    console.print()
    console.print(
      "You may re-run the command to retry, or use"
      " [bold]kinetic pool list[/bold] to check current state."
    )
  console.print()


@pool.command("remove")
@common_options
@click.argument("pool_name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def pool_remove(project, zone, cluster_name, pool_name, yes):
  """Remove an accelerator node pool from the cluster."""
  banner("kinetic Pool Remove")

  state = load_state(project, zone, cluster_name)

  remaining = [p for p in state.node_pools if p.name != pool_name]
  if len(remaining) == len(state.node_pools):
    existing_names = [p.name for p in state.node_pools]
    raise click.ClickException(
      f"Node pool '{pool_name}' not found. "
      f"Existing pools: {', '.join(existing_names) or '(none)'}"
    )

  console.print(f"\nRemoving pool [bold]{pool_name}[/bold]")
  console.print(f"Remaining pools after remove: {len(remaining)}\n")

  if not yes:
    click.confirm("Proceed?", abort=True)

  config = InfraConfig(
    project=state.project,
    zone=state.zone,
    cluster_name=state.cluster_name,
    node_pools=remaining,
  )
  update_succeeded = apply_update(config)

  console.print()
  if update_succeeded:
    banner("Pool Removed")
  else:
    banner("Pool Update Failed")
    console.print()
    console.print(
      "You may re-run the command to retry, or use"
      " [bold]kinetic pool list[/bold] to check current state."
    )
  console.print()


@pool.command("list")
@common_options
def pool_list(project, zone, cluster_name):
  """List accelerator node pools on the cluster."""
  banner("kinetic Node Pools")

  state = load_state(project, zone, cluster_name, allow_missing=True)

  if state.stack is None:
    warning("No Pulumi stack found.")
    console.print("Run 'kinetic up' to provision infrastructure.")
    return

  outputs = state.stack.outputs()
  if not outputs:
    warning("No infrastructure found. Run 'kinetic up' first.")
    return

  infrastructure_state(outputs)
