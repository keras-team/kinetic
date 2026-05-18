"""Per-pod log streaming cursor, persisted to disk for cross-shell resume.

Records the last-seen pod-log timestamp and a small ring of recent line
hashes. On reconnect (or in a fresh process), the streamer uses
``since_time`` to skip past already-seen output and the hash ring to
dedupe the inevitable second-granular overlap.
"""

import json
import os
import shutil
import time
from collections import deque
from pathlib import Path

_DEFAULT_WRITE_INTERVAL_S = 1.0
_RING_SIZE = 200


def default_cursor_dir() -> Path:
  return Path.home() / ".kinetic" / "streams"


def cursor_path_for(
  cursor_dir: Path | None, job_id: str, pod_name: str
) -> Path | None:
  if cursor_dir is None:
    return None
  return cursor_dir / _safe_name(job_id) / f"{_safe_name(pod_name)}.json"


def clear_job_cursors(cursor_dir: Path | None, job_id: str) -> None:
  """Remove every per-pod cursor for a job once it reaches a terminal state."""
  if cursor_dir is None:
    return
  job_dir = cursor_dir / _safe_name(job_id)
  if job_dir.exists():
    shutil.rmtree(job_dir, ignore_errors=True)


def _safe_name(name: str) -> str:
  # k8s names are already restrictive, but be defensive against unusual job IDs.
  return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


class LogCursor:
  """In-memory cursor with optional disk persistence.

  ``path=None`` disables persistence entirely (the cursor still tracks
  dedup state in memory for the current process).
  """

  def __init__(
    self,
    path: Path | None,
    *,
    write_interval_s: float = _DEFAULT_WRITE_INTERVAL_S,
    ring_size: int = _RING_SIZE,
  ):
    self._path = path
    self._interval = write_interval_s
    self._last_write = time.monotonic()
    self._last_ts: str | None = None
    self._recent_hashes: deque[str] = deque(maxlen=ring_size)
    self._dirty = False

  @property
  def since_time(self) -> str | None:
    return self._last_ts

  def load(self) -> None:
    """Best-effort read from disk. Missing or corrupt files reset to empty."""
    if self._path is None or not self._path.exists():
      return
    try:
      data = json.loads(self._path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
      return
    if not isinstance(data, dict):
      return
    ts = data.get("last_ts")
    if isinstance(ts, str):
      self._last_ts = ts
    hashes = data.get("recent_hashes")
    if isinstance(hashes, list):
      for h in hashes:
        if isinstance(h, str):
          self._recent_hashes.append(h)

  def is_duplicate(self, line_hash: str) -> bool:
    return line_hash in self._recent_hashes

  def record(self, timestamp: str, line_hash: str) -> None:
    self._last_ts = timestamp
    self._recent_hashes.append(line_hash)
    self._dirty = True
    if self._path is None:
      return
    now = time.monotonic()
    if now - self._last_write >= self._interval:
      self._flush()
      self._last_write = now

  def flush(self) -> None:
    """Force a write to disk if any state has changed since the last flush."""
    if self._dirty:
      self._flush()

  def _flush(self) -> None:
    if self._path is None:
      return
    try:
      self._path.parent.mkdir(parents=True, exist_ok=True)
      tmp = self._path.with_suffix(self._path.suffix + ".tmp")
      tmp.write_text(
        json.dumps(
          {
            "last_ts": self._last_ts,
            "recent_hashes": list(self._recent_hashes),
          }
        )
      )
      os.replace(tmp, self._path)
      self._dirty = False
    except OSError:
      # Best-effort: dropping a cursor write only costs us extra dedup on
      # the next resume.
      pass
