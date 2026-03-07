"""keras-remote up command — provision infrastructure."""

import subprocess

import click

from keras_remote.cli.config import InfraConfig, NodePoolConfig
from keras_remote.cli.constants import DEFAULT_CLUSTER_NAME, DEFAULT_ZONE
from keras_remote.cli.infra.post_deploy import (
  configure_kubectl,
  install_gpu_drivers,
  install_lws,
)
from keras_remote.cli.infra.state import apply_update, load_state
from keras_remote.cli.options import common_options
from keras_remote.cli.output import (
  LiveOutputPanel,
  banner,
  config_summary,
  console,
  warning,
)
from keras_remote.cli.prerequisites_check import check_all
from keras_remote.cli.prompts import prompt_accelerator, resolve_project
from keras_remote.core import accelerators
from keras_remote.core.accelerators import GpuConfig, generate_pool_name


@click.command()
@common_options
@click.option(
  "--accelerator",
  default=None,
  help="Accelerator spec: cpu, t4, l4, a100, a100-80gb, h100, "
  "v5litepod, v5p, v6e, v3",
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def up(project, zone, accelerator, cluster_name, yes):
  """Provision GCP infrastructure for keras-remote."""
  banner("keras-remote Setup")

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
  # Users should manage pools via `keras-remote pool add/remove` after
  # initial setup.
  state = load_state(
    project,
    zone,
    cluster_name,
    allow_missing=True,
    check_prerequisites=False,
  )

  config = InfraConfig(project=project, zone=zone, cluster_name=cluster_name)

  if state.node_pools:
    config.node_pools = list(state.node_pools)
    console.print(
      f"\nFound {len(state.node_pools)} existing node pool(s)."
      "\nUse 'keras-remote pool add/remove/list' to manage node pools.\n"
    )
  elif accel_config is not None:
    config.node_pools.append(
      NodePoolConfig(generate_pool_name(accel_config), accel_config)
    )

  # Show summary and confirm
  config_summary(config)
  if not yes:
    click.confirm("\nProceed with setup?", abort=True)

  console.print()

  pulumi_ok = apply_update(config)
  pulumi_failed = not pulumi_ok

  if pulumi_failed:
    warning("Attempting post-deploy configuration anyway...")

  # Post-deploy steps
  steps = [
    (
      "kubectl configuration",
      lambda: configure_kubectl(
        cluster_name,
        zone,
        project,
      ),
    ),
    ("LWS CRD installation", install_lws),
  ]
  if any(isinstance(np.accelerator, GpuConfig) for np in config.node_pools):
    steps.append(("GPU driver installation", install_gpu_drivers))

  failures = []
  with LiveOutputPanel("Post-deploy configuration", transient=True) as panel:
    for name, fn in steps:
      panel.on_output(f"{name}...")
      try:
        fn()
        panel.on_output(f"{name} complete.")
      except subprocess.CalledProcessError as e:
        failures.append(name)
        panel.on_output(f"{name} failed: {e}")
        panel.mark_error()

  # Final summary
  console.print()
  if pulumi_failed or failures:
    banner("Setup Completed With Warnings")
    console.print()
    if pulumi_failed:
      warning("Pulumi provisioning encountered errors (see above).")
    if failures:
      warning(f"Post-deploy steps failed: {', '.join(failures)}")
    console.print()
    console.print(
      "You may re-run [bold]keras-remote up[/bold] to retry failed steps."
    )
  else:
    banner("Setup Complete")

  console.print()
  console.print("Add these environment variables to your shell config:")
  console.print(f"  export KERAS_REMOTE_PROJECT={project}")
  console.print(f"  export KERAS_REMOTE_ZONE={zone}")
  console.print(f"  export KERAS_REMOTE_CLUSTER={cluster_name}")
  console.print()
  console.print("View quotas:")
  console.print(
    f"  https://console.cloud.google.com/iam-admin/quotas?project={project}"
  )
  console.print()
