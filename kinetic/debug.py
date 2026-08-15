"""Debug connection utilities for Kinetic remote jobs.

Provides helpers for setting up debugpy-based remote debugging sessions,
including port-forwarding orchestration and VS Code attach configuration.
"""

import contextlib
import json
import os
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

# Environment variable that overrides the debugger attach window.
DEBUG_WAIT_TIMEOUT_ENV = "KINETIC_DEBUG_WAIT_TIMEOUT"

# Default debugger attach timeout (seconds).  Covers pod scheduling +
# debugpy install + time for the user to attach.
DEFAULT_DEBUG_WAIT_TIMEOUT = 600

_TERMINAL_STATUSES = frozenset(
  {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.NOT_FOUND}
)

# Grace period (seconds) to verify kubectl port-forward started successfully.
_PORT_FORWARD_STARTUP_SECONDS = 2


def resolve_debug_wait_timeout():
  """Return the debugger attach window in seconds.

  Single source of truth for both sides of a debug session: the client
  uses the value for its own wait in ``wait_for_debug_server``, and the
  backends put it in the pod's ``KINETIC_DEBUG_WAIT_TIMEOUT`` env var so
  remote_runner.py waits for exactly as long.

  Returns:
      The value of ``KINETIC_DEBUG_WAIT_TIMEOUT`` when it is a positive
      whole number of seconds, ``DEFAULT_DEBUG_WAIT_TIMEOUT`` otherwise.
  """
  raw = os.environ.get(DEBUG_WAIT_TIMEOUT_ENV)
  if not raw:
    return DEFAULT_DEBUG_WAIT_TIMEOUT
  try:
    seconds = int(raw)
  except ValueError:
    seconds = 0
  if seconds > 0:
    return seconds
  logging.warning(
    "Ignoring invalid %s=%r (expected a positive whole number of seconds); "
    "using %ds.",
    DEBUG_WAIT_TIMEOUT_ENV,
    raw,
    DEFAULT_DEBUG_WAIT_TIMEOUT,
  )
  return DEFAULT_DEBUG_WAIT_TIMEOUT


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
  # delete=False because we need the file to survive until
  # cleanup_port_forward() can read any mid-session errors
  # (e.g. cluster disconnect, pod eviction). The file is
  # explicitly unlinked there after logging its contents.
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

  The pod reconstructs the client's own working-directory path (the
  runner symlinks it to the extracted workspace), so source files carry
  identical paths on both sides and the mapping is the identity.  When
  the working directory is unknown the mapping is left out entirely,
  which is what debugpy already does with unmapped paths.

  Args:
      local_port: Local port where debugpy is forwarded.
      working_dir: Local working directory for path mappings. If None,
          no pathMappings entry is emitted.
  """
  config_lines = [
    '    "name": "Kinetic Debug",',
    '    "type": "debugpy",',
    '    "request": "attach",',
    f'    "connect": {{"host": "localhost", "port": {local_port}}}',
  ]
  if working_dir:
    # os.fspath because debug_attach() is public API and a caller can
    # hand it a pathlib.Path, which json.dumps cannot serialize.  This
    # runs after start_port_forward(), so raising here would leak the
    # kubectl subprocess.  json.dumps then keeps a path that holds a
    # backslash or a quote valid in the printed snippet.
    root = json.dumps(os.fspath(working_dir))
    config_lines[-1] += ","
    config_lines.extend(
      [
        '    "pathMappings": [',
        "      {",
        f'        "localRoot": {root},',
        f'        "remoteRoot": {root}',
        "      }",
        "    ]",
      ]
    )

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
    *config_lines,
    "  }",
    "",
  ]
  if working_dir:
    lines.extend(
      [
        "Kinetic mirrors your local paths on the pod, so localRoot",
        "and remoteRoot are the same directory. Edit both if you",
        "open the sources from somewhere else.",
        "",
      ]
    )
  else:
    lines.extend(
      [
        "The remote paths match your local ones, so no pathMappings",
        "entry is needed. If your sources are somewhere else, add a",
        'mapping with "localRoot" set to that directory and',
        '"remoteRoot" set to the directory you submitted from.',
        "",
      ]
    )
  lines.extend(
    [
      "Set your breakpoints, then start debugging with F5 or",
      "via the menu: Run > Start Debugging.",
      "",
      "The debugger will pause inside the Kinetic runner before",
      "your function is called. Press Step Into (F11) to enter",
      "your function, or Step Over (F10) to run it directly.",
      "=" * 50,
      "",
    ]
  )
  print("\n".join(lines))  # noqa: T201


def wait_for_debug_server(handle, timeout=None, poll_interval=5):
  """Poll GCS sentinel until the debugpy server confirms readiness.

  Logs progress as the job transitions through states so the user
  sees feedback during the wait.

  Args:
      handle: A JobHandle instance.
      timeout: Maximum seconds to wait.  Defaults to the window from
          ``resolve_debug_wait_timeout()``, so raising
          ``KINETIC_DEBUG_WAIT_TIMEOUT`` extends this wait and the pod's
          attach wait together rather than only one of the two.  The two
          waits are consecutive: this one covers pod scheduling and the
          debugpy install, and ends when the pod publishes its readiness
          sentinel; the pod's own window for the debugger to attach
          starts there.
      poll_interval: Seconds between log polls.

  Raises:
      TimeoutError: If the signal is not found within timeout.
      RuntimeError: If the job reaches a terminal state before the signal.
  """
  if timeout is None:
    timeout = resolve_debug_wait_timeout()
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
  """Terminate a port-forward subprocess, log any stderr, and clean up.

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
      stderr_file.seek(0)
      stderr_output = stderr_file.read().strip()
      if stderr_output:
        logging.warning("port-forward stderr: %s", stderr_output)
      stderr_file.close()
      os.unlink(stderr_file.name)
