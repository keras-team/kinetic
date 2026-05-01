"""Rich console output helpers for the kinetic CLI."""

import random
import time

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from kinetic.core.accelerators import GpuConfig, TpuConfig

console = Console()

_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

_SUBTITLE_MESSAGES = (
  # Fun phrases and helpful tips, interleaved.
  "Painting the pods",
  "Tip: Pass Data('./dataset/') as a function arg to upload data",
  "Winding all the butterflies",
  "Tip: Use volumes={'/mnt': Data('./data/')} to mount data on the pod",
  "Warming the compute engine",
  "Tip: Data is content-hashed — identical data is uploaded only once",
  "Reticulating splines",
  "Tip: Data() accepts GCS URIs too, e.g. Data('gs://bucket/path/')",
  "Charging the flux capacitor",
  "Tip: Data objects nested in lists/dicts are recursively discovered",
  "Aligning the cloud crystals",
  "Tip: Container images are content-hashed — unchanged deps skip rebuilds",
  "Feeding the hamsters",
  "Tip: Add a requirements.txt or pyproject.toml to auto-install deps on the remote pod",
  "Consulting the oracle",
  "Tip: Use --cluster to manage multiple clusters in the same project",
  "Calibrating the widgets",
  "Tip: Run 'kinetic pool add --accelerator v5p-8' to add a TPU pool",
  "Herding the containers",
  "Tip: Run 'kinetic pool list' to see all accelerators on your cluster",
  "Polishing the tensors",
  "Tip: Pass --yes to 'pool add/remove' to skip the confirmation prompt",
  "Summoning the cluster spirits",
  "Tip: Use cluster= in @run() to pick a cluster (or env KINETIC_CLUSTER)",
  "Untangling the neural pathways",
  "Tip: Set zone= in @run() to pick a GCP zone (or env KINETIC_ZONE)",
  "Brewing the cloud juice",
  "Tip: Use capture_env_vars=['PREFIX_*'] in @run() to forward env vars to the worker",
  "Wrangling the cloud gremlins",
  "Tip: Multi-host TPUs (e.g. v6e-4x4) auto-select the Pathways backend",
  "Compiling the butterfly wings",
  "Tip: Your working directory is auto-zipped and sent to the pod",
  "Tuning the hyperparameters of the universe",
  "Tip: Remote exceptions are re-raised locally with original traceback",
  "Spinning up the hamster wheels",
  "Tip: Run 'kinetic config show' to check your current settings",
  "Negotiating with the load balancer",
  "Tip: Use container_image= in @run() to bring your own Docker image",
  "Teaching the pods to dance",
  "Tip: Use namespace= in @run() to pick a K8s namespace",
  "Downloading more RAM",
  "Tip: Set KINETIC_PROJECT or --project to pick a specific GCP project",
)


class LiveOutputPanel:
  """Context manager that displays streaming output in a Rich Live panel.

  Shows the last `max_lines` in a bordered box. Supports error state
  (yellow border) and optional transient mode (clears on success).

  In non-interactive terminals, falls back to plain console output.
  """

  def __init__(
    self,
    title,
    *,
    max_lines=7,
    target_console=None,
    transient=False,
    show_subtitle=True,
  ):
    self._title = title
    self._max_lines = max_lines
    self._lines = []
    self._has_error = False
    self._transient = transient
    self._show_subtitle = show_subtitle
    self._console = target_console or console
    self._live = None
    self._start_time = None
    self._phrase_order = None

  def __enter__(self):
    self._start_time = time.monotonic()
    self._phrase_order = list(range(len(_SUBTITLE_MESSAGES)))
    random.shuffle(self._phrase_order)
    if self._console.is_terminal:
      self._live = Live(
        self,
        console=self._console,
        refresh_per_second=5,
      )
      self._live.__enter__()
    else:
      self._console.rule(self._title, style="blue")
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    if exc_type is not None:
      self._has_error = True
    if self._live:
      if self._transient and not self._has_error:
        self._live.update(Text(""))
      self._live.__exit__(exc_type, exc_val, exc_tb)
    else:
      style = "yellow" if self._has_error else "blue"
      self._console.rule(style=style)
    return False

  def __rich__(self):
    return self._make_panel()

  def on_output(self, line):
    """Append a line and refresh the display."""
    stripped = line.rstrip("\n")
    if self._live:
      self._lines.append(stripped)
    else:
      self._console.print(stripped)

  def mark_error(self):
    """Turn the panel border yellow to indicate an error."""
    self._has_error = True
    if self._live:
      self._live.refresh()

  def _make_subtitle(self):
    if self._start_time is None or self._phrase_order is None:
      return None
    elapsed = time.monotonic() - self._start_time
    spinner_idx = int(elapsed * 4) % len(_SPINNER_FRAMES)
    spinner = _SPINNER_FRAMES[spinner_idx]
    msg_idx = int(elapsed / 4) % len(_SUBTITLE_MESSAGES)
    message = _SUBTITLE_MESSAGES[self._phrase_order[msg_idx]]
    suffix = "" if message.startswith("Tip:") else "..."
    return f"[italic]{spinner} {message}{suffix}[/italic]"

  def _make_panel(self):
    if self._lines:
      visible = (
        self._lines if self._has_error else self._lines[-self._max_lines :]
      )
      content = "\n".join(visible)
    else:
      content = "Waiting..."
    style = "yellow" if self._has_error else "blue"
    return Panel(
      content,
      title=self._title,
      subtitle=self._make_subtitle()
      if self._show_subtitle and not self._has_error
      else None,
      border_style=style,
    )


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
      "[dim]Unknown (run 'kinetic up' to refresh)[/dim]",
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

  table.add_row(
    "Force Destroy Buckets",
    "yes"
    if config.force_destroy
    else "no (empty buckets manually before down)",
  )

  console.print()
  console.print(table)
