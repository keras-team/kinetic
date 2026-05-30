"""Live log streaming from Kubernetes pods.

Provides utilities to stream pod logs to stdout in real-time using a
background daemon thread. Used by both GKE and Pathways backends during
job execution.

The stream survives transient interruptions like laptop sleep, wifi
drops, and VPN reconnects. After a disconnect the worker reconnects
with capped exponential backoff, resumes from the last seen log
timestamp, and dedupes the second-granular overlap. With
``resume=True`` (the default) a small cursor file under
``~/.kinetic/streams/`` lets a fresh ``kinetic jobs logs --follow``
invocation pick up where the previous one left off.
"""

import codecs
import contextlib
import hashlib
import http.client
import random
import re
import socket
import threading

import urllib3
from absl import logging
from kubernetes import client as k8s_client
from kubernetes import config as k8s_config
from kubernetes.client.rest import ApiException
from rich.console import Console

from kinetic.backend.log_cursor import (
  LogCursor,
  cursor_path_for,
  default_cursor_dir,
)
from kinetic.cli.output import LiveOutputPanel
from kinetic.credentials import invalidate_credential_cache

_MAX_DISPLAY_LINES = 25
_CONNECT_TIMEOUT_S = 10
_READ_TIMEOUT_S = 60
_MIN_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 30.0
_BACKOFF_JITTER = 0.25

# kubernetes pod log API with timestamps=True returns lines shaped like
#   "2024-01-01T12:00:00.123456789Z user log content"
# Match the timestamp prefix. Everything after the single separating space
# is the original log content.
_TIMESTAMP_PATTERN = re.compile(
  r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\s(.*)$"
)

# urllib3 / stdlib errors that mean "connection died, try again". We list
# the specific subclasses we want to retry on rather than the catch-all
# ``urllib3.exceptions.HTTPError`` base: SSL errors, malformed URLs, and
# unknown URL schemes are configuration bugs that retrying would just hide.
_TRANSIENT_ERRORS = (
  urllib3.exceptions.ProtocolError,
  urllib3.exceptions.ReadTimeoutError,
  urllib3.exceptions.ConnectTimeoutError,
  urllib3.exceptions.NewConnectionError,
  urllib3.exceptions.NameResolutionError,
  urllib3.exceptions.MaxRetryError,
  socket.timeout,
  http.client.IncompleteRead,
  ConnectionError,
  TimeoutError,
)


def _parse_timestamped_line(line: str) -> tuple[str | None, str]:
  m = _TIMESTAMP_PATTERN.match(line)
  if m is None:
    return None, line
  return m.group(1), m.group(2)


def _truncate_to_second(ts: str) -> str:
  """Trim fractional seconds, since k8s ``sinceTime`` is second-granular."""
  if "." in ts:
    return ts.split(".", 1)[0] + "Z"
  return ts


def _backoff_seconds(attempt: int) -> float:
  base = min(_MIN_BACKOFF_S * (2 ** max(0, attempt - 1)), _MAX_BACKOFF_S)
  return base + random.uniform(0, base * _BACKOFF_JITTER)


def _refresh_k8s_client() -> object | None:
  """Re-bootstrap kubeconfig and return a fresh CoreV1Api, or None on failure.

  Called after a 401/403 from the log API. Usually means the cached
  auth token expired during a long sleep and the exec plugin needs to
  run again.
  """
  invalidate_credential_cache()
  try:
    k8s_config.load_kube_config()
    return k8s_client.CoreV1Api()
  except Exception:
    logging.warning("Failed to refresh kubernetes client", exc_info=True)
    return None


def _is_pod_terminal(core_v1, pod_name: str, namespace: str) -> bool:
  """Check whether the pod has finished, meaning the stream is really over."""
  try:
    pod = core_v1.read_namespaced_pod(name=pod_name, namespace=namespace)
  except ApiException as e:
    return e.status == 404
  except Exception:
    return False
  try:
    return pod.status.phase in ("Succeeded", "Failed")
  except AttributeError:
    return False


def _process_line(line: str, panel, cursor: LogCursor, dedup: bool) -> None:
  if dedup:
    ts, content = _parse_timestamped_line(line)
  else:
    ts, content = None, line
  # Strip carriage-return prefixes (progress bars) from the content only, so
  # the timestamp prefix stays intact for dedup hashing.
  if "\r" in content:
    content = content.rsplit("\r", 1)[-1]
  if ts is not None:
    line_hash = hashlib.sha1(f"{ts}\t{content}".encode("utf-8")).hexdigest()
    if cursor.is_duplicate(line_hash):
      return
    cursor.record(ts, line_hash)
  panel.on_output(content)


def _consume_stream(
  resp,
  panel,
  cursor: LogCursor,
  stop_event: threading.Event,
  dedup: bool,
) -> bool:
  """Drain one open log stream into the panel, raising on transport error.

  Returns True if at least one log line was processed. The caller uses this
  signal to decide whether the stream made real progress, which resets the
  reconnect backoff escalation.
  """
  panel.set_subtitle(None)  # we're connected, clear any reconnect notice
  # Use an incremental decoder so multi-byte UTF-8 sequences that straddle
  # chunk boundaries are buffered instead of getting replaced with U+FFFD.
  decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
  buffer = ""
  saw_line = False
  for chunk in resp.stream(decode_content=True):
    if stop_event.is_set():
      return saw_line
    buffer += decoder.decode(chunk)
    while "\n" in buffer:
      line, buffer = buffer.split("\n", 1)
      _process_line(line, panel, cursor, dedup)
      saw_line = True
  if buffer:
    _process_line(buffer, panel, cursor, dedup)
    saw_line = True
  return saw_line


def _open_log_stream(
  core_v1, pod_name: str, namespace: str, since_time: str | None, dedup: bool
):
  return core_v1.read_namespaced_pod_log(
    name=pod_name,
    namespace=namespace,
    follow=True,
    timestamps=dedup,
    since_time=since_time,
    _preload_content=False,
    _request_timeout=(_CONNECT_TIMEOUT_S, _READ_TIMEOUT_S),
  )


def _stream_pod_logs(
  core_v1,
  pod_name: str,
  namespace: str,
  *,
  cursor: LogCursor | None = None,
  stop_event: threading.Event | None = None,
  resume: bool = True,
):
  """Stream pod logs to stdout, reconnecting through transient outages.

  Designed to run in a daemon thread. In interactive terminals, logs are
  displayed in a Rich Live panel. In non-interactive contexts like
  piped output or CI, they stream as plain lines with rule delimiters.

  Args:
      core_v1: Kubernetes CoreV1Api client. Replaced internally on
          401/403 if credentials need refreshing.
      pod_name: Name of the pod to stream logs from.
      namespace: Kubernetes namespace.
      cursor: Optional ``LogCursor`` for resume + dedup. If ``None`` and
          ``resume`` is true, an in-memory-only cursor is used.
      stop_event: ``threading.Event`` the owning ``LogStreamer`` sets to
          request shutdown. A no-op event is created if ``None``.
      resume: When true, request timestamped log lines, dedupe on
          reconnect, and use the cursor's ``since_time`` to skip
          already-seen output.
  """
  if stop_event is None:
    stop_event = threading.Event()
  if cursor is None:
    cursor = LogCursor(path=None)
  if resume:
    cursor.load()

  pod_gone = False
  title = f"Remote logs • {pod_name}"
  with LiveOutputPanel(
    title,
    max_lines=_MAX_DISPLAY_LINES,
    target_console=Console(),
    show_subtitle=False,
  ) as panel:
    # Counts back-to-back failures with no progress. Reset only when the
    # stream actually delivers a line, so repeated mid-stream drops still
    # escalate the backoff schedule.
    consecutive_failures = 0
    while not stop_event.is_set():
      resp = None
      since = (
        _truncate_to_second(cursor.since_time)
        if resume and cursor.since_time
        else None
      )
      try:
        resp = _open_log_stream(core_v1, pod_name, namespace, since, resume)
      except ApiException as e:
        if e.status == 404:
          pod_gone = True
          break
        if e.status == 410:
          # since_time too old, reset and retry without it
          cursor.clear_timestamp()
        elif e.status in (401, 403):
          refreshed = _refresh_k8s_client()
          if refreshed is not None:
            core_v1 = refreshed
        elif not 500 <= e.status < 600:
          logging.warning(
            "Pod log API returned %s for %s, giving up", e.status, pod_name
          )
          break
        consecutive_failures += 1
      except _TRANSIENT_ERRORS:
        consecutive_failures += 1
      except Exception:
        logging.warning(
          "Unexpected error opening log stream for %s", pod_name, exc_info=True
        )
        consecutive_failures += 1

      if resp is not None:
        saw_progress = False
        consume_failed = False
        try:
          saw_progress = _consume_stream(
            resp, panel, cursor, stop_event, resume
          )
        except (*_TRANSIENT_ERRORS, ApiException):
          consume_failed = True
        except Exception:
          logging.warning(
            "Log stream from %s ended unexpectedly", pod_name, exc_info=True
          )
          consume_failed = True
        finally:
          with contextlib.suppress(Exception):
            resp.release_conn()

        if saw_progress:
          consecutive_failures = 0
        if not consume_failed and _is_pod_terminal(
          core_v1, pod_name, namespace
        ):
          break
        if consume_failed or not saw_progress:
          consecutive_failures += 1

      if stop_event.is_set():
        break

      wait = _backoff_seconds(consecutive_failures)
      panel.set_subtitle(
        f"disconnected • reconnecting in {wait:.0f}s"
        f" (attempt {consecutive_failures + 1})"
      )
      if stop_event.wait(wait):
        break

  # Cursor cleanup: the pod is done with for good (terminal or gone), so
  # drop the on-disk cursor instead of leaving it for the next session.
  # A stop_event exit keeps the cursor so the user can resume.
  if pod_gone or (
    not stop_event.is_set() and _is_pod_terminal(core_v1, pod_name, namespace)
  ):
    cursor.delete()
  else:
    cursor.flush()


class LogStreamer:
  """Context manager that owns the log-streaming thread lifecycle.

  Usage::

      with LogStreamer(core_v1, namespace, job_id=jid) as streamer:
          while polling:
              ...
              if pod_is_running:
                  streamer.start(pod_name)  # idempotent

  The streamer maintains a small per-pod cursor file under
  ``~/.kinetic/streams/`` so log following can resume across process
  restarts. Pass ``resume=False`` to disable that persistence. The
  worker still reconnects on transient drops, it just won't remember
  state between shells.
  """

  def __init__(
    self,
    core_v1,
    namespace: str,
    *,
    job_id: str,
    resume: bool = True,
    cursor_dir=None,
  ):
    self._core_v1 = core_v1
    self._namespace = namespace
    self._job_id = job_id
    self._resume = resume
    self._cursor_dir = (
      cursor_dir
      if cursor_dir is not None
      else (default_cursor_dir() if resume else None)
    )
    self._thread = None
    self._stop_event = threading.Event()
    self._cursor: LogCursor | None = None

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    del exc_type, exc_val, exc_tb
    self._stop_event.set()
    if self._thread is not None:
      self._thread.join(timeout=5)
    return False

  def start(self, pod_name):
    """Start streaming if not already active (idempotent)."""
    if self._thread is not None:
      return
    logging.info("Streaming logs from %s...", pod_name)
    self._cursor = LogCursor(
      path=cursor_path_for(self._cursor_dir, self._job_id, pod_name)
      if self._resume
      else None,
    )
    self._thread = threading.Thread(
      target=_stream_pod_logs,
      args=(self._core_v1, pod_name, self._namespace),
      kwargs={
        "cursor": self._cursor,
        "stop_event": self._stop_event,
        "resume": self._resume,
      },
      daemon=True,
    )
    self._thread.start()
