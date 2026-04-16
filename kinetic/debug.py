"""Debug connection utilities for Kinetic remote jobs.

Provides helpers for setting up debugpy-based remote debugging sessions,
including port-forwarding orchestration and VS Code attach configuration.
"""

import contextlib
import subprocess
import tempfile
import time

from absl import logging

from kinetic.job_status import JobStatus
from kinetic.utils import storage

# 5678 is the default port that debugpy listens on,
# and it's the port VS Code's Python debugger extension
# auto-fills when generating an "attach" launch
# configuration. Using it means most users won't need
# to change any port settings — VS Code just works out
# of the box.
DEBUGPY_PORT = 5678

_TERMINAL_STATUSES = frozenset(
  {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.NOT_FOUND}
)

# Grace period (seconds) to verify kubectl port-forward started successfully.
_PORT_FORWARD_STARTUP_SECONDS = 2


def start_port_forward(pod_name, namespace, local_port, remote_port):
  """Start kubectl port-forward as a background subprocess.

  After launching, waits briefly to verify the process didn't exit
  immediately (e.g. due to a port conflict). Stderr is captured to a
  temp file for diagnostics.

  Args:
      pod_name: Name of the pod to forward to.
      namespace: Kubernetes namespace.
      local_port: Local port to listen on.
      remote_port: Remote port on the pod.

  Returns:
      subprocess.Popen handle for the port-forward process.

  Raises:
      RuntimeError: If the port-forward process exits immediately,
          typically due to a port conflict.
  """
  cmd = [
    "kubectl",
    "port-forward",
    f"pod/{pod_name}",
    f"{local_port}:{remote_port}",
    "-n",
    namespace,
  ]
  logging.info(
    "Starting port-forward: localhost:%d -> %s:%d",
    local_port,
    pod_name,
    remote_port,
  )

  # Capture stderr to a temp file so failures are diagnosable.
  stderr_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
    mode="w+", prefix="kinetic-pf-", suffix=".log", delete=False
  )
  proc = subprocess.Popen(
    cmd,
    stdout=subprocess.DEVNULL,
    stderr=stderr_file,
  )

  # Give kubectl a moment to bind the port; if it exits immediately
  # the port is likely in use or the pod is unreachable.
  time.sleep(_PORT_FORWARD_STARTUP_SECONDS)
  exit_code = proc.poll()
  if exit_code is not None:
    stderr_file.seek(0)
    stderr_output = stderr_file.read().strip()
    stderr_file.close()
    msg = (
      f"kubectl port-forward exited immediately (code {exit_code}). "
      f"Port {local_port} may already be in use.\n"
    )
    if stderr_output:
      msg += f"stderr: {stderr_output}\n"
    msg += (
      "Try a different local port with: "
      "handle.debug_attach(local_port=<port>) or "
      "kinetic jobs debug <job_id> --port <port>"
    )
    raise RuntimeError(msg)

  # Keep the file handle on the process for later diagnostics.
  proc._stderr_file = stderr_file
  return proc


def print_attach_instructions(local_port, working_dir=None):
  """Print VS Code launch.json snippet for attaching to the remote debugger.

  Args:
      local_port: Local port where debugpy is forwarded.
      working_dir: Local working directory for path mappings. If None,
          uses a placeholder.
  """
  local_root = working_dir or "${workspaceFolder}"
  # Use print() rather than logging.info() — these are user-facing
  # instructions that must appear exactly once on stdout.  The logging
  # subsystem can duplicate messages when multiple handlers are
  # registered (common in notebooks and IDE integrations).
  lines = [
    "",
    "=" * 50,
    "  Connect your debugger (VS Code)",
    "=" * 50,
    "",
    "Add the following configuration to .vscode/launch.json",
    "in your workspace root. If the file does not exist, you",
    "can create it manually or via VS Code: open the Run and",
    "Debug panel (Ctrl+Shift+D / Cmd+Shift+D), click",
    '"create a launch.json file", then replace its contents.',
    "",
    "  {",
    '    "name": "Kinetic Debug",',
    '    "type": "debugpy",',
    '    "request": "attach",',
    f'    "connect": {{"host": "localhost", "port": {local_port}}},',
    '    "pathMappings": [',
    "      {",
    f'        "localRoot": "{local_root}",',
    '        "remoteRoot": "/tmp/workspace"',
    "      }",
    "    ]",
    "  }",
    "",
    "Set your breakpoints, then start debugging with F5 or",
    "via the menu: Run > Start Debugging.",
    "",
    "The debugger will pause inside the Kinetic runner before",
    "your function is called. Press Step Into (F11) to enter",
    "your function, or Step Over (F10) to run it directly.",
    "=" * 50,
    "",
  ]
  print("\n".join(lines))  # noqa: T201


def wait_for_debug_server(handle, timeout=300, poll_interval=5):
  """Poll GCS sentinel until the debugpy server confirms readiness.

  Logs progress as the job transitions through states so the user
  sees feedback during the wait.

  Args:
      handle: A JobHandle instance.
      timeout: Maximum seconds to wait.
      poll_interval: Seconds between log polls.

  Raises:
      TimeoutError: If the signal is not found within timeout.
      RuntimeError: If the job reaches a terminal state before the signal.
  """
  deadline = time.monotonic() + timeout
  last_status = None
  while time.monotonic() < deadline:
    status = handle.status()

    # Log status transitions so the user sees progress.
    if status != last_status:
      if status == JobStatus.PENDING:
        logging.info("Waiting for pod to be scheduled...")
      elif status == JobStatus.RUNNING:
        logging.info("Pod is running, waiting for debugpy server readiness...")
      last_status = status

    if status in _TERMINAL_STATUSES:
      raise RuntimeError(
        f"Job {handle.job_id} reached terminal state ({status.value}) "
        "before debugpy server was ready."
      )

    if status == JobStatus.RUNNING:
      try:
        if storage.blob_exists(
          handle.bucket_name,
          f"{handle.job_id}/.debug_ready",
          project=handle.project,
        ):
          logging.info("debugpy server is ready.")
          return
      except Exception:
        pass

    time.sleep(poll_interval)
  raise TimeoutError(
    f"Timed out after {timeout}s waiting for debugpy server to start "
    f"on job {handle.job_id}."
  )


def cleanup_port_forward(proc):
  """Terminate a port-forward subprocess and close its stderr file.

  Args:
      proc: The subprocess.Popen returned by start_port_forward().
  """
  proc.terminate()
  try:
    proc.wait(timeout=5)
  except subprocess.TimeoutExpired:
    proc.kill()
  stderr_file = getattr(proc, "_stderr_file", None)
  if stderr_file is not None:
    with contextlib.suppress(Exception):
      stderr_file.close()
