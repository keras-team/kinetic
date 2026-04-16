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

DEBUGPY_PORT = 5678
DEBUGPY_READY_SIGNAL = "[DEBUGPY] Ready"

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
  instructions = (
    "\n"
    "=" * 50 + "\n"
    "  Connect your debugger (VS Code)\n"
    "=" * 50 + "\n"
    "\n"
    "Add to your launch.json:\n"
    "\n"
    "  {\n"
    '    "name": "Kinetic Debug",\n'
    '    "type": "debugpy",\n'
    '    "request": "attach",\n'
    f'    "connect": {{"host": "localhost", "port": {local_port}}},\n'
    '    "pathMappings": [\n'
    "      {\n"
    f'        "localRoot": "{local_root}",\n'
    '        "remoteRoot": "/tmp/workspace"\n'
    "      }\n"
    "    ]\n"
    "  }\n"
    "\n"
    f"Then press F5 in VS Code to attach to localhost:{local_port}.\n"
    "\n"
    "The debugger will break inside the Kinetic runner, just before\n"
    "your function is called. Press Step Into (F11) to enter your\n"
    "function, or Step Over (F10) to execute it without stepping.\n"
    "=" * 50 + "\n"
  )
  logging.info(instructions)


def wait_for_debug_server(handle, timeout=300, poll_interval=5):
  """Poll job logs until the debugpy ready signal appears.

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
        logging.info("Pod is running, waiting for debugpy server...")
      last_status = status

    if status in _TERMINAL_STATUSES:
      raise RuntimeError(
        f"Job {handle.job_id} reached terminal state ({status.value}) "
        "before debugpy server was ready."
      )
    try:
      logs = handle.tail(n=50)
      if logs and DEBUGPY_READY_SIGNAL in logs:
        logging.info("debugpy server is ready.")
        return
    except Exception:
      pass  # Pod may not be ready yet
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
