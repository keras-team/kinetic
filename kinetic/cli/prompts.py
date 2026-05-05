"""Interactive prompts for the kinetic CLI."""

import json
import os
import subprocess

import click

from kinetic.cli.output import success, warning
from kinetic.core.accelerators import GPUS, TPUS, parse_accelerator


def resolve_project(allow_create=True):
  """Resolve GCP project ID from env or prompt.

  Validates that the project exists. If *allow_create* is True and it
  doesn't, offers to create it and link a billing account.
  """
  project = os.environ.get("KINETIC_PROJECT")
  if not project:
    prompt_msg = (
      "Enter your GCP project ID (or a new ID to create one)"
      if allow_create
      else "Enter your GCP project ID"
    )
    project = click.prompt(prompt_msg, type=str)

  if _project_exists(project):
    return project

  if not allow_create:
    raise click.ClickException(f"Project '{project}' was not found.")

  if not click.confirm(
    f"\nProject '{project}' was not found. Would you like to create it?"
  ):
    raise click.Abort()

  _create_project(project)
  _link_billing_account(project)
  return project


def _project_exists(project_id):
  """Check if a GCP project exists and is accessible."""
  result = subprocess.run(
    ["gcloud", "projects", "describe", "--", project_id],
    capture_output=True,
  )
  return result.returncode == 0


def _create_project(project_id):
  """Create a new GCP project."""
  click.echo(f"Creating project '{project_id}'...")
  result = subprocess.run(
    ["gcloud", "projects", "create", "--", project_id],
    capture_output=True,
    text=True,
  )
  if result.returncode != 0:
    raise click.ClickException(
      f"Failed to create project '{project_id}': {result.stderr.strip()}"
    )
  success(f"Project '{project_id}' created")


def _link_billing_account(project_id):
  """List billing accounts and link one to the project."""
  result = subprocess.run(
    [
      "gcloud",
      "billing",
      "accounts",
      "list",
      "--filter=open=true",
      "--format=json",
    ],
    capture_output=True,
    text=True,
  )
  if result.returncode != 0:
    warning("Could not list billing accounts. Link one manually later.")
    return

  accounts = json.loads(result.stdout) if result.stdout.strip() else []
  if not accounts:
    warning(
      "No billing accounts found. "
      "Link one at: https://console.cloud.google.com/billing"
    )
    return

  if len(accounts) == 1:
    account_id = accounts[0]["name"].removeprefix("billingAccounts/")
    display = accounts[0].get("displayName", account_id)
    if not click.confirm(
      f"\nLink billing account '{display}' ({account_id}) "
      f"to project '{project_id}'?"
    ):
      warning("Skipped billing account linking.")
      return
  else:
    click.echo("\nAvailable billing accounts:")
    for i, acct in enumerate(accounts, 1):
      acct_id = acct["name"].removeprefix("billingAccounts/")
      display = acct.get("displayName", acct_id)
      click.echo(f"  {i}) {display} ({acct_id})")

    choices = [str(i) for i in range(1, len(accounts) + 1)]
    idx = click.prompt(
      "\nSelect billing account",
      type=click.Choice(choices),
    )
    account_id = accounts[int(idx) - 1]["name"].removeprefix("billingAccounts/")

  result = subprocess.run(
    [
      "gcloud",
      "billing",
      "projects",
      "link",
      "--",
      project_id,
      f"--billing-account={account_id}",
    ],
    capture_output=True,
    text=True,
  )
  if result.returncode != 0:
    warning(
      f"Failed to link billing account: {result.stderr.strip()}\n"
      "Link one manually at: "
      "https://console.cloud.google.com/billing"
    )
    return
  success("Billing account linked")


def prompt_accelerator():
  """Interactive accelerator selection menu.

  Returns:
      GpuConfig, TpuConfig, or None (for CPU).
  """
  accel_type = click.prompt(
    "\nWhat type of accelerator node pool?",
    type=click.Choice(["cpu", "gpu", "tpu"], case_sensitive=False),
    default="cpu",
  )

  if accel_type == "cpu":
    return None
  elif accel_type == "gpu":
    return _prompt_gpu()
  elif accel_type == "tpu":
    return _prompt_tpu()
  else:
    raise click.ClickException(f"Unknown accelerator type: {accel_type}")


def _prompt_gpu():
  """Prompt for GPU type selection."""
  click.echo()
  click.echo("Available GPU types:")
  gpu_names = list(GPUS.keys())
  for i, name in enumerate(gpu_names, 1):
    spec = GPUS[name]
    click.echo(f"  {i}) {name:<12} ({spec.gke_label})")

  choice = click.prompt(
    "\nSelect GPU type",
    type=click.Choice(gpu_names, case_sensitive=False),
  )
  return parse_accelerator(choice)


def _prompt_tpu():
  """Prompt for TPU type and topology selection."""
  click.echo()
  tpu_names = list(TPUS.keys())
  click.echo("Available TPU types:")
  for i, name in enumerate(tpu_names, 1):
    spec = TPUS[name]
    topos = [ts.topology for ts in spec.topologies.values()]
    click.echo(f"  {i}) {name:<12} (topologies: {', '.join(topos)})")

  tpu = click.prompt(
    "\nSelect TPU type",
    type=click.Choice(tpu_names, case_sensitive=False),
  )

  spec = TPUS[tpu]
  topology_items = list(spec.topologies.items())
  click.echo()
  click.echo(f"Available topologies for {tpu}:")
  for i, (_chips, ts) in enumerate(topology_items, 1):
    click.echo(
      f"  {i}) {ts.topology:<6} "
      f"(machine: {ts.machine_type}, nodes: {ts.num_nodes})"
    )

  topo_strs = [ts.topology for _, ts in topology_items]
  default_topo = topo_strs[min(1, len(topo_strs) - 1)]
  topology = click.prompt(
    f"\nSelect topology for {tpu}",
    type=click.Choice(topo_strs),
    default=default_topo,
  )

  return parse_accelerator(f"{tpu}-{topology}")
