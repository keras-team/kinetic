"""Live log streaming from Kubernetes pods.

Provides utilities to stream pod logs to stdout in real-time using a
background daemon thread. Used by both GKE and Pathways backends during
job execution.
"""

import threading

import urllib3
from absl import logging
from kubernetes.client.rest import ApiException
from rich.console import Console

from kinetic.cli.output import LiveOutputPanel

_MAX_DISPLAY_LINES = 25


def _stream_pod_logs(core_v1, pod_name, namespace):
  """Stream pod logs to stdout. Designed to run in a daemon thread.

  Uses the Kubernetes follow API to tail logs in real-time. The stream
  ends naturally when the container exits.

  In interactive terminals, logs are displayed in a Rich Live panel.
  In non-interactive contexts (piped output, CI), logs are streamed
  as plain lines with Rule delimiters.

  Args:
      core_v1: Kubernetes CoreV1Api client.
      pod_name: Name of the pod to stream logs from.
      namespace: Kubernetes namespace.
  """
  resp = None
  try:
    resp = core_v1.read_namespaced_pod_log(
      name=pod_name,
      namespace=namespace,
      follow=True,
      _preload_content=False,
    )
    title = f"Remote logs \u2022 {pod_name}"
    with LiveOutputPanel(
      title,
      max_lines=_MAX_DISPLAY_LINES,
      target_console=Console(),
      show_subtitle=False,
    ) as panel:
      buffer = ""
      for chunk in resp.stream(decode_content=True):
        buffer += chunk.decode("utf-8", errors="replace")
        while "\n" in buffer:
          line, buffer = buffer.split("\n", 1)
          if "\r" in line:
            line = line.rsplit("\r", 1)[-1]
          panel.on_output(line)
      # Flush remaining partial line
      if buffer.strip():
        line = buffer
        if "\r" in line:
          line = line.rsplit("\r", 1)[-1]
        panel.on_output(line)
  except ApiException:
    pass  # Pod deleted or not found
  except urllib3.exceptions.ProtocolError:
    pass  # Connection broken mid-stream (pod terminated)
  except Exception:
    logging.warning(
      "Log streaming from %s failed unexpectedly", pod_name, exc_info=True
    )
  finally:
    if resp is not None:
      resp.release_conn()


class LogStreamer:
  """Context manager that owns the log-streaming thread lifecycle.

  Usage::

      with LogStreamer(core_v1, namespace) as streamer:
          while polling:
              ...
              if pod_is_running:
                  streamer.start(pod_name)  # idempotent
  """

  def __init__(self, core_v1, namespace):
    self._core_v1 = core_v1
    self._namespace = namespace
    self._thread = None

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    if self._thread is not None:
      self._thread.join(timeout=5)
    return False

  def start(self, pod_name):
    """Start streaming if not already active (idempotent)."""
    if self._thread is not None:
      return
    logging.info("Streaming logs from %s...", pod_name)
    self._thread = threading.Thread(
      target=_stream_pod_logs,
      args=(self._core_v1, pod_name, self._namespace),
      daemon=True,
    )
    self._thread.start()
