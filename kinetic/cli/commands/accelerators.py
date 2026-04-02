"""kinetic accelerators command -- list supported accelerator types."""

import click
from rich.table import Table

from kinetic.cli.output import banner, console, warning
from kinetic.core.accelerators import (
  GPUS,
  PREFERRED_GPUS,
  PREFERRED_TPUS,
  TPU_ALIASES,
  TPUS,
  GpuConfig,
  TpuConfig,
)


def _get_provisioned_names(node_pools):
  """Return the set of accelerator names that have active node pools."""
  names = set()
  for np in node_pools:
    a = np.accelerator
    if isinstance(a, (GpuConfig, TpuConfig)):
      names.add(a.name)
  return names


@click.command("accelerators")
@click.option(
  "--live",
  is_flag=True,
  help="Check cluster for provisioned (hot) accelerators.",
)
def accelerators(live):
  """List supported accelerator types and their configurations."""
  banner("kinetic Accelerators")

  provisioned = set()
  if live:
    provisioned = _load_provisioned()

  _print_gpu_table(provisioned, show_status=live)
  _print_tpu_table(provisioned, show_status=live)

  if live and provisioned:
    console.print("provisioned = has active node pool on cluster")
  console.print()


def _load_provisioned():
  """Load provisioned accelerator names from the cluster."""
  from kinetic.cli.infra.state import load_state

  try:
    state = load_state(None, None, None, allow_missing=True)
    if state.stack is not None and state.node_pools:
      return _get_provisioned_names(state.node_pools)
  except Exception:
    warning("Could not load cluster state.")
  return set()


def _print_gpu_table(provisioned, show_status):
  gpu_table = Table(title="GPUs")
  if show_status:
    gpu_table.add_column("Status", width=12)
  gpu_table.add_column("Name", style="bold")
  gpu_table.add_column("Counts")
  gpu_table.add_column("Machine Types")

  for name in PREFERRED_GPUS:
    if name not in GPUS:
      continue
    spec = GPUS[name]
    counts = sorted(spec.counts)
    row = []
    if show_status:
      row.append("provisioned" if name in provisioned else "")
    row.append(name)
    row.append(", ".join(str(c) for c in counts))
    row.append(", ".join(spec.counts[c] for c in counts))
    gpu_table.add_row(*row)

  console.print()
  console.print(gpu_table)


def _print_tpu_table(provisioned, show_status):
  reverse_aliases = {v: k for k, v in TPU_ALIASES.items()}

  tpu_table = Table(title="TPUs")
  if show_status:
    tpu_table.add_column("Status", width=12)
  tpu_table.add_column("Name", style="bold")
  tpu_table.add_column("Chips")
  tpu_table.add_column("Topologies")

  for name in PREFERRED_TPUS:
    if name not in TPUS:
      continue
    spec = TPUS[name]
    chips = sorted(spec.topologies)
    display = name
    alias = reverse_aliases.get(name)
    if alias:
      display = f"{name} ({alias})"
    row = []
    if show_status:
      row.append("provisioned" if name in provisioned else "")
    row.append(display)
    row.append(", ".join(str(c) for c in chips))
    row.append(", ".join(spec.topologies[c].topology for c in chips))
    tpu_table.add_row(*row)

  console.print()
  console.print(tpu_table)
