"""Debug connection utilities for Kinetic remote jobs.

Provides helpers for setting up debugpy-based remote debugging sessions,
including port-forwarding orchestration and VS Code attach configuration.
"""

import subprocess
import time

from absl import logging

from kinetic.job_status import JobStatus

DEBUGPY_PORT = 5678
DEBUGPY_READY_SIGNAL = "[DEBUGPY] Ready"

_TERMINAL_STATUSES = frozenset(
  {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.NOT_FOUND}
)


def start_port_forward(pod_name, namespace, local_port, remote_port):
  """Start kubectl port-forward as a background subprocess.

  Args:
      pod_name: Name of the pod to forward to.
      namespace: Kubernetes namespace.
      local_port: Local port to listen on.
      remote_port: Remote port on the pod.

  Returns:
      subprocess.Popen handle for the port-forward process.
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
  return subprocess.Popen(
    cmd,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
  )


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
    "=" * 50 + "\n"
  )
  logging.info(instructions)


def wait_for_debug_server(handle, timeout=300, poll_interval=5):
  """Poll job logs until the debugpy ready signal appears.

  Args:
      handle: A JobHandle instance.
      timeout: Maximum seconds to wait.
      poll_interval: Seconds between log polls.

  Raises:
      TimeoutError: If the signal is not found within timeout.
      RuntimeError: If the job reaches a terminal state before the signal.
  """
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    status = handle.status()
    if status in _TERMINAL_STATUSES:
      raise RuntimeError(
        f"Job {handle.job_id} reached terminal state ({status.value}) "
        "before debugpy server was ready."
      )
    try:
      logs = handle.tail(n=50)
      if logs and DEBUGPY_READY_SIGNAL in logs:
        return
    except Exception:
      pass  # Pod may not be ready yet
    time.sleep(poll_interval)
  raise TimeoutError(
    f"Timed out after {timeout}s waiting for debugpy server to start "
    f"on job {handle.job_id}."
  )
