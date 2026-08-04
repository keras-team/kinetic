"""Smoke harness for resumable log streaming.

Drives kinetic.backend.log_streaming._stream_pod_logs against an
in-process fake CoreV1Api so you can watch the Rich panel and verify
reconnect, dedup, and cross-process resume without a real cluster.

Scenarios::

    python smoke_log_streaming.py --scenario clean
    python smoke_log_streaming.py --scenario flap
    python smoke_log_streaming.py --scenario partial   # run twice to see resume
    python smoke_log_streaming.py --scenario partial --no-resume

Add ``--clean-cursor`` to wipe any persisted state before starting.
"""

import argparse
import shutil
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import urllib3.exceptions

from kinetic.backend.log_cursor import LogCursor
from kinetic.backend.log_streaming import _stream_pod_logs

_BASE = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _fmt_ts(seconds: float) -> str:
  t = _BASE + timedelta(seconds=seconds)
  return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond * 1000:09d}Z"


def _parse_since(since_time: str) -> float:
  ts = datetime.strptime(since_time, "%Y-%m-%dT%H:%M:%SZ").replace(
    tzinfo=timezone.utc
  )
  return (ts - _BASE).total_seconds()


class FakeResponse:
  def __init__(
    self, start_idx, rate, drop_after, terminate_at, born_at, timestamps
  ):
    self._idx = start_idx
    self._rate = rate
    self._drop_after = drop_after
    self._terminate_at = terminate_at
    self._born_at = born_at
    self._timestamps = timestamps
    self._opened_at = time.monotonic()

  def stream(self, decode_content=True):
    del decode_content
    while True:
      now = time.monotonic()
      if self._terminate_at and (now - self._born_at) >= self._terminate_at:
        return
      if self._drop_after and (now - self._opened_at) >= self._drop_after:
        raise urllib3.exceptions.ProtocolError("simulated disconnect")
      if self._timestamps:
        ts = _fmt_ts(self._idx / self._rate)
        line = f"{ts} line {self._idx:05d}\n"
      else:
        line = f"line {self._idx:05d}\n"
      yield line.encode("utf-8")
      time.sleep(1.0 / self._rate)
      self._idx += 1

  def release_conn(self):
    pass


class _FakeApiClient:
  """Translates the raw ``call_api`` log request back into kwargs.

  Production calls ``api_client.call_api(...)`` with a ``sinceTime`` query
  param because the generated client lacks a ``since_time`` argument.
  """

  def __init__(self, core):
    self._core = core

  def select_header_accept(self, accepts):
    del accepts
    return "text/plain"

  def call_api(
    self, resource_path, method, path_params, query_params, header_params, **_
  ):
    del resource_path, method, path_params, header_params
    q = dict(query_params)
    return self._core.read_namespaced_pod_log(
      name=None,
      namespace=None,
      since_time=q.get("sinceTime"),
      timestamps=q.get("timestamps", False),
    )


class FakeCoreV1Api:
  def __init__(self, rate, disconnects, terminate_at):
    self._rate = rate
    self._drops = deque(disconnects)
    self._terminate_at = terminate_at
    self._born_at = time.monotonic()
    self.api_client = _FakeApiClient(self)

  def read_namespaced_pod_log(
    self, name, namespace, since_time=None, timestamps=False, **_
  ):
    del name, namespace
    start_idx = 0
    if since_time:
      offset = _parse_since(since_time)
      # Fake k8s behavior: server returns everything >= since_time (whole
      # seconds), so the client will see ``rate`` duplicates per second of
      # overlap that the dedup ring needs to filter.
      start_idx = max(0, int(offset * self._rate))
    drop_after = self._drops.popleft() if self._drops else None
    return FakeResponse(
      start_idx=start_idx,
      rate=self._rate,
      drop_after=drop_after,
      terminate_at=self._terminate_at,
      born_at=self._born_at,
      timestamps=timestamps,
    )

  def read_namespaced_pod(self, name, namespace):
    del name, namespace
    elapsed = time.monotonic() - self._born_at
    terminal = self._terminate_at is not None and elapsed >= self._terminate_at
    phase = "Succeeded" if terminal else "Running"
    return SimpleNamespace(status=SimpleNamespace(phase=phase))


SCENARIOS = {
  # name -> (disconnects: drop_after seconds per opened stream, terminate_at)
  "clean": ([], 20.0),
  "flap": ([4.0, 4.0, 4.0, 4.0], 25.0),
  "partial": ([], None),  # no terminal, no drops — for cross-process resume
}


def _build_cursor(args):
  if args.no_resume:
    return LogCursor(path=None)
  cursor_dir = Path(args.cursor_dir)
  if args.clean_cursor:
    shutil.rmtree(cursor_dir / args.job_id, ignore_errors=True)
  path = cursor_dir / args.job_id / f"{args.pod}.json"
  return LogCursor(path=path, write_interval_s=0.5)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--scenario", choices=list(SCENARIOS), default="clean")
  p.add_argument("--job-id", default="smoke-job")
  p.add_argument("--pod", default="smoke-pod-0")
  p.add_argument("--rate", type=float, default=5.0, help="lines/sec")
  p.add_argument(
    "--duration",
    type=float,
    default=10.0,
    help="for non-terminating scenarios, stop after N seconds",
  )
  p.add_argument("--no-resume", action="store_true")
  p.add_argument(
    "--cursor-dir", default=str(Path.home() / ".kinetic" / "streams")
  )
  p.add_argument("--clean-cursor", action="store_true")
  args = p.parse_args()

  disconnects, terminate_at = SCENARIOS[args.scenario]
  fake = FakeCoreV1Api(
    rate=args.rate, disconnects=disconnects, terminate_at=terminate_at
  )
  cursor = _build_cursor(args)
  stop_event = threading.Event()

  if terminate_at is None:

    def _stopper():
      time.sleep(args.duration)
      stop_event.set()

    threading.Thread(target=_stopper, daemon=True).start()

  try:
    _stream_pod_logs(
      fake,
      args.pod,
      "default",
      cursor=cursor,
      stop_event=stop_event,
      resume=not args.no_resume,
    )
  except KeyboardInterrupt:
    stop_event.set()


if __name__ == "__main__":
  main()
