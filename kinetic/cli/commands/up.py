"""kinetic up command — provision infrastructure."""

import subprocess

import click

from kinetic.cli.config import InfraConfig, NodePoolConfig
from kinetic.cli.constants import DEFAULT_CLUSTER_NAME, DEFAULT_ZONE
from kinetic.cli.infra.post_deploy import configure_kubectl
from kinetic.cli.infra.state import apply_update, load_state
from kinetic.cli.options import common_options
from kinetic.cli.output import (
  banner,
  config_summary,
  console,
  warning,
)
from kinetic.cli.prerequisites_check import check_all
from kinetic.cli.prompts import prompt_accelerator, resolve_project
from kinetic.core import accelerators
from kinetic.core.accelerators import generate_pool_name


@click.command()
@common_options
@click.option(
  "--accelerator",
  default=None,
  help="Accelerator spec: cpu, t4, l4, a100, a100-80gb, h100, "
  "v5litepod, v5p, v6e, v3",
)
@click.option(
  "--min-nodes",
  default=0,
  type=int,
  help="Minimum node count for accelerator node pools (default: 0, scale-to-zero)",
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def up(project, zone, accelerator, cluster_name, min_nodes, yes):
  """Provision GCP infrastructure for kinetic."""
  banner("kinetic Setup")

  # Check prerequisites
  check_all()

  # Resolve configuration
  project = project or resolve_project()
  zone = zone or DEFAULT_ZONE
  cluster_name = cluster_name or DEFAULT_CLUSTER_NAME

  # Resolve accelerator (interactive if not provided)
  if accelerator and accelerator.strip().lower() == "cpu":
    accel_config = None
  elif accelerator:
    try:
      accel_config = accelerators.parse_accelerator(accelerator)
    except ValueError as e:
      raise click.BadParameter(str(e), param_hint="--accelerator") from e
  else:
    accel_config = prompt_accelerator()

  # If a stack already exists, preserve its node pools as-is.
  # Users should manage pools via `kinetic pool add/remove` after
  # initial setup.
  state = load_state(
    project,
    zone,
    cluster_name,
    allow_missing=True,
    check_prerequisites=False,
  )

  config = InfraConfig(
    project=project,
    zone=zone,
    cluster_name=cluster_name,
  )

  if state.node_pools:
    config.node_pools = list(state.node_pools)
    console.print(
      f"\nFound {len(state.node_pools)} existing node pool(s)."
      "\nUse 'kinetic pool add/remove/list' to manage node pools.\n"
    )
  elif accel_config is not None:
    config.node_pools.append(
      NodePoolConfig(
        generate_pool_name(accel_config), accel_config, min_nodes=min_nodes
      )
    )

  # Show summary and confirm
  config_summary(config)
  if not yes:
    click.confirm("\nProceed with setup?", abort=True)

  console.print()

  pulumi_ok = apply_update(config)

  # Configure local kubectl context so the user can interact with the
  # cluster immediately.  Non-fatal — the user can always run
  # `gcloud container clusters get-credentials` manually.
  try:
    configure_kubectl(cluster_name, zone, project)
  except subprocess.CalledProcessError:
    warning(
      "kubectl configuration failed. Run manually:\n"
      f"  gcloud container clusters get-credentials {cluster_name}"
      f" --zone={zone} --project={project}"
    )

  # Final summary
  console.print()
  if not pulumi_ok:
    banner("Setup Completed With Warnings")
    console.print()
    warning("Pulumi provisioning encountered errors (see above).")
    console.print()
    console.print("You may re-run [bold]kinetic up[/bold] to retry.")
  else:
    banner("Setup Complete")

  console.print()
  console.print("Add these environment variables to your shell config:")
  console.print(f"  export KINETIC_PROJECT={project}")
  console.print(f"  export KINETIC_ZONE={zone}")
  console.print(f"  export KINETIC_CLUSTER={cluster_name}")
  console.print()
  console.print("View quotas:")
  console.print(
    f"  https://console.cloud.google.com/iam-admin/quotas?project={project}"
  )
  console.print()
