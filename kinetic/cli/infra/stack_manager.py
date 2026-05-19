"""Pulumi Automation API wrapper for kinetic."""

import click
import pulumi.automation as auto

from kinetic.cli.config import NodePoolConfig
from kinetic.cli.constants import PULUMI_ROOT, RESOURCE_NAME_PREFIX
from kinetic.cli.infra.state_backend import (
  ensure_gcs_backend,
  state_backend_url,
)
from kinetic.cli.output import loading
from kinetic.core import accelerators


def get_stack(program_fn, config):
  """Create or select a Pulumi stack on the GCS backend for ``config.project``.

  The backend bucket is created on first use (idempotent), so any
  state-touching command can be the first one a team member runs.

  Args:
      program_fn: Pulumi inline program callable.
      config: InfraConfig instance.

  Returns:
      A pulumi.automation.Stack instance.
  """
  with loading("Connecting to Pulumi state backend (GCS)…"):
    ensure_gcs_backend(config.project)

    # Auto-install the Pulumi CLI if not already present.
    try:
      pulumi_cmd = auto.PulumiCommand(root=PULUMI_ROOT)
    except Exception:  # noqa: BLE001
      click.echo("Pulumi CLI not found. Installing...")
      pulumi_cmd = auto.PulumiCommand.install(root=PULUMI_ROOT)

    # Each (project, cluster) pair gets its own stack, so multiple clusters
    # within the same GCP project are fully independent.
    stack_name = f"{config.project}-{config.cluster_name}"

    project_settings = auto.ProjectSettings(
      name=RESOURCE_NAME_PREFIX,
      runtime="python",
      backend=auto.ProjectBackend(url=state_backend_url(config.project)),
    )

    stack = auto.create_or_select_stack(
      stack_name=stack_name,
      project_name=RESOURCE_NAME_PREFIX,
      program=program_fn,
      opts=auto.LocalWorkspaceOptions(
        project_settings=project_settings,
        env_vars={"PULUMI_CONFIG_PASSPHRASE": ""},
        pulumi_command=pulumi_cmd,
      ),
    )

    # Set GCP provider configuration on the stack
    stack.set_config("gcp:project", auto.ConfigValue(value=config.project))
    stack.set_config("gcp:zone", auto.ConfigValue(value=config.zone))

  return stack


def get_current_force_destroy(stack) -> bool:
  """Read ``force_destroy`` from stack outputs, defaulting to True.

  Stacks created before this setting was exported do not have the output,
  in which case they used ``force_destroy=True`` implicitly, so the same
  default is returned for compatibility.
  """
  outputs = stack.outputs()
  if "force_destroy" in outputs:
    return bool(outputs["force_destroy"].value)
  return True


def get_current_zone(stack) -> str | None:
  """Read the GCP zone from stack outputs, or None if the output is missing.

  Used to discover where an existing cluster lives when joining via
  ``kinetic init`` — the user does not know (and should not have to know)
  the zone for clusters provisioned by a teammate.
  """
  outputs = stack.outputs()
  if "zone" in outputs and outputs["zone"].value:
    return str(outputs["zone"].value)
  return None


def get_current_node_pools(stack) -> list[NodePoolConfig]:
  """Read the current node pool list from Pulumi stack exports.

  Handles both the new ``accelerators`` key (list) and the legacy
  ``accelerator`` key (single dict or None).

  Args:
      stack: A ``pulumi.automation.Stack`` whose outputs have been
          populated (e.g. after ``stack.refresh()``).

  Returns:
      A list of :class:`NodePoolConfig` objects.
  """
  outputs = stack.outputs()

  # New format: list of accelerator dicts.
  if "accelerators" in outputs:
    accel_list = outputs["accelerators"].value
    if not accel_list:
      return []
    return [_export_to_node_pool(entry) for entry in accel_list]

  # Legacy format: single dict or None.
  if "accelerator" in outputs:
    accel = outputs["accelerator"].value
    if accel is None:
      return []
    return [_export_to_node_pool(accel)]

  return []


def _export_to_node_pool(entry: dict) -> NodePoolConfig:
  """Convert a stack export dict back to a NodePoolConfig."""
  pool_name = entry["node_pool"]
  accelerator: accelerators.GpuConfig | accelerators.TpuConfig
  if entry["type"] == "GPU":
    accelerator = accelerators.make_gpu(entry["name"], entry["count"])
  elif entry["type"] == "TPU":
    accelerator = accelerators.make_tpu(entry["name"], entry["chips"])
  else:
    raise ValueError(f"Unknown accelerator type in node pool export: {entry}")
  return NodePoolConfig(
    name=pool_name,
    accelerator=accelerator,
    min_nodes=entry.get("min_nodes", 0),
  )
