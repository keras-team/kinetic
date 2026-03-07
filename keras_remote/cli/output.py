"""Rich console output helpers for the keras-remote CLI."""

from collections import deque

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from keras_remote.core.accelerators import GpuConfig, TpuConfig

console = Console()


class LiveOutputPanel:
  """Context manager that displays streaming output in a Rich Live panel.

  Shows the last `max_lines` in a bordered box. Supports error state
  (yellow border) and optional transient mode (clears on success).

  In non-interactive terminals, falls back to plain console output.
  """

  def __init__(
    self, title, *, max_lines=7, target_console=None, transient=False
  ):
    self._title = title
    self._lines = deque(maxlen=max_lines)
    self._has_error = False
    self._transient = transient
    self._console = target_console or console
    self._live = None

  def __enter__(self):
    if self._console.is_terminal:
      self._live = Live(
        self._make_panel(),
        console=self._console,
        refresh_per_second=4,
      )
      self._live.__enter__()
    else:
      self._console.rule(self._title, style="blue")
    return self

  def __exit__(self, *args):
    if self._live:
      if self._transient and not self._has_error:
        self._live.update(Text(""))
      self._live.__exit__(*args)
    else:
      style = "yellow" if self._has_error else "blue"
      self._console.rule(style=style)
    return False

  def on_output(self, line):
    """Append a line and refresh the display."""
    stripped = line.rstrip("\n")
    if self._live:
      self._lines.append(stripped)
      self._live.update(self._make_panel())
    else:
      self._console.print(stripped)

  def mark_error(self):
    """Turn the panel border yellow to indicate an error."""
    self._has_error = True
    if self._live:
      self._live.update(self._make_panel())

  def _make_panel(self):
    content = "\n".join(self._lines) if self._lines else "Waiting..."
    style = "yellow" if self._has_error else "blue"
    return Panel(content, title=self._title, border_style=style)


def banner(text):
  """Display a styled banner."""
  console.print(Panel(f"  {text}", style="bold blue"))


def success(msg):
  """Display a success message."""
  console.print(f"[green]{msg}[/green]")


def warning(msg):
  """Display a warning message."""
  console.print(f"[yellow]{msg}[/yellow]")


def error(msg):
  """Display an error message."""
  console.print(f"[red]{msg}[/red]")


_INFRA_LABELS = {
  "project": "Project",
  "zone": "Zone",
  "cluster_name": "Cluster Name",
  "cluster_endpoint": "Cluster Endpoint",
  "ar_registry": "Artifact Registry",
}

_GPU_LABELS = {
  "name": "GPU Type",
  "count": "GPU Count",
  "machine_type": "Machine Type",
  "node_pool": "Node Pool",
  "node_count": "Node Count",
}

_TPU_LABELS = {
  "name": "TPU Type",
  "chips": "TPU Chips",
  "topology": "Topology",
  "machine_type": "Machine Type",
  "node_pool": "Node Pool",
  "node_count": "Node Count",
}


def infrastructure_state(outputs):
  """Display infrastructure state from Pulumi stack outputs.

  Args:
      outputs: dict of key -> pulumi.automation.OutputValue from stack.outputs().
  """
  table = Table(title="Infrastructure State")
  table.add_column("Resource", style="bold")
  table.add_column("Value", style="green")

  for key, label in _INFRA_LABELS.items():
    if key in outputs:
      table.add_row(label, str(outputs[key].value))

  # New format: "accelerators" (list of dicts)
  if "accelerators" in outputs:
    accel_list = outputs["accelerators"].value
    if not accel_list:
      table.add_row("Accelerators", "CPU only (no accelerator pools)")
    else:
      table.add_row("", "")
      table.add_row(f"Accelerator Pools ({len(accel_list)})", "")
      for i, accel in enumerate(accel_list, 1):
        _render_accelerator(table, accel, index=i)

  # Legacy format: "accelerator" (single dict or None)
  elif "accelerator" in outputs:
    if outputs["accelerator"].value is None:
      table.add_row("Accelerator", "CPU only")
    else:
      accel = outputs["accelerator"].value
      accel_type = accel.get("type", "Unknown")
      table.add_row("", "")
      table.add_row("Accelerator", accel_type)
      labels = _GPU_LABELS if accel_type == "GPU" else _TPU_LABELS
      for key, label in labels.items():
        if key in accel:
          table.add_row(f"  {label}", str(accel[key]))

  else:
    table.add_row(
      "Accelerator",
      "[dim]Unknown (run 'keras-remote up' to refresh)[/dim]",
    )

  console.print()
  console.print(table)
  console.print()


def _render_accelerator(table, accel, index=None):
  """Render a single accelerator pool entry in the status table."""
  accel_type = accel.get("type", "Unknown")
  pool_name = accel.get("node_pool", "")
  prefix = f"  Pool {index}" if index else "  Pool"
  table.add_row(f"{prefix}: {accel_type}", pool_name)
  labels = _GPU_LABELS if accel_type == "GPU" else _TPU_LABELS
  for key, label in labels.items():
    if key in accel and key != "node_pool":
      table.add_row(f"    {label}", str(accel[key]))


def config_summary(config):
  """Display a configuration summary table."""
  table = Table(title="Configuration Summary")
  table.add_column("Setting", style="bold")
  table.add_column("Value", style="green")

  table.add_row("Project", config.project)
  table.add_row("Zone", config.zone)
  table.add_row("Cluster Name", config.cluster_name)

  if not config.node_pools:
    table.add_row("Accelerators", "CPU only")
  else:
    accel_strs = []
    for np in config.node_pools:
      accel = np.accelerator
      if isinstance(accel, GpuConfig):
        accel_strs.append(f"GPU ({accel.gke_label})")
      elif isinstance(accel, TpuConfig):
        accel_strs.append(f"TPU ({accel.name}, {accel.topology})")
    table.add_row("Accelerators", ", ".join(accel_strs))

  console.print()
  console.print(table)
