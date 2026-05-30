"""Tests for kinetic.backend.log_streaming — live pod log streaming."""

import json
import tempfile
import threading
import time
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import urllib3
from absl.testing import absltest
from kubernetes.client.rest import ApiException

from kinetic.backend.log_cursor import (
  LogCursor,
  cursor_path_for,
)
from kinetic.backend.log_streaming import (
  LogStreamer,
  _backoff_seconds,
  _parse_timestamped_line,
  _stream_pod_logs,
  _truncate_to_second,
)


def _make_resp(chunks):
  resp = MagicMock()
  resp.stream.return_value = iter(chunks)
  return resp


def _stop_after(stop_event, *items):
  """Yield each item, then set ``stop_event`` so the worker exits cleanly."""
  for item in items:
    yield item
  stop_event.set()


class TestStreamPodLogs(absltest.TestCase):
  """Tests for the top-level _stream_pod_logs orchestrator."""

  def _run_once(self, mock_core, **kwargs):
    """Run _stream_pod_logs with a stop_event that fires after one iteration."""
    stop_event = kwargs.pop("stop_event", None) or threading.Event()
    # Default: stop the loop as soon as the stream consumer returns.
    # Tests that want multi-iteration behavior pass their own stop_event.
    with mock.patch(
      "kinetic.backend.log_streaming.Console"
    ) as mock_console_cls:
      mock_console_cls.return_value.is_terminal = False
      _stream_pod_logs(
        mock_core,
        "pod-1",
        "default",
        stop_event=stop_event,
        **kwargs,
      )
    return stop_event

  def test_calls_log_api_with_timestamps_and_request_timeout(self):
    mock_core = MagicMock()
    mock_core.read_namespaced_pod_log.return_value = _make_resp([])
    mock_core.read_namespaced_pod.return_value.status.phase = "Succeeded"

    self._run_once(mock_core)

    mock_core.read_namespaced_pod_log.assert_called_once()
    kwargs = mock_core.read_namespaced_pod_log.call_args.kwargs
    self.assertEqual(kwargs["name"], "pod-1")
    self.assertEqual(kwargs["namespace"], "default")
    self.assertTrue(kwargs["follow"])
    self.assertTrue(kwargs["timestamps"])
    self.assertIsNone(kwargs["since_time"])
    self.assertEqual(kwargs["_preload_content"], False)
    self.assertEqual(kwargs["_request_timeout"], (10, 60))

  def test_resume_passes_since_time_truncated_to_second(self):
    mock_core = MagicMock()
    mock_core.read_namespaced_pod_log.return_value = _make_resp([])
    mock_core.read_namespaced_pod.return_value.status.phase = "Succeeded"

    cursor = LogCursor(path=None)
    cursor._last_ts = "2024-01-01T12:00:00.123456789Z"

    self._run_once(mock_core, cursor=cursor)

    self.assertEqual(
      mock_core.read_namespaced_pod_log.call_args.kwargs["since_time"],
      "2024-01-01T12:00:00Z",
    )

  def test_resume_false_omits_timestamps_and_since_time(self):
    mock_core = MagicMock()
    mock_core.read_namespaced_pod_log.return_value = _make_resp([])
    mock_core.read_namespaced_pod.return_value.status.phase = "Succeeded"

    cursor = LogCursor(path=None)
    cursor._last_ts = "2024-01-01T12:00:00.0Z"

    self._run_once(mock_core, cursor=cursor, resume=False)

    kwargs = mock_core.read_namespaced_pod_log.call_args.kwargs
    self.assertFalse(kwargs["timestamps"])
    self.assertIsNone(kwargs["since_time"])

  def test_dedupes_lines_already_in_cursor_ring(self):
    mock_core = MagicMock()
    mock_core.read_namespaced_pod_log.return_value = _make_resp(
      [
        b"2024-01-01T12:00:00.000Z dup line\n",
        b"2024-01-01T12:00:01.000Z fresh line\n",
      ]
    )
    mock_core.read_namespaced_pod.return_value.status.phase = "Succeeded"

    cursor = LogCursor(path=None)
    # Pre-seed the ring with the hash that line 1 will compute to.
    import hashlib

    dup_hash = hashlib.sha1(b"2024-01-01T12:00:00.000Z\tdup line").hexdigest()
    cursor._recent_hashes.append(dup_hash)

    with mock.patch(
      "kinetic.backend.log_streaming.LiveOutputPanel"
    ) as mock_panel_cls:
      mock_panel = MagicMock()
      mock_panel_cls.return_value.__enter__.return_value = mock_panel
      mock_panel_cls.return_value.__exit__.return_value = False
      _stream_pod_logs(
        mock_core,
        "pod-1",
        "default",
        cursor=cursor,
        stop_event=threading.Event(),
      )

    emitted = [c.args[0] for c in mock_panel.on_output.call_args_list]
    self.assertEqual(emitted, ["fresh line"])
    self.assertEqual(cursor.since_time, "2024-01-01T12:00:01.000Z")

  def test_handles_partial_lines(self):
    mock_core = MagicMock()
    mock_core.read_namespaced_pod_log.return_value = _make_resp(
      [b"2024-01-01T12:00:00.000Z hel", b"lo\n2024-01-01T12:00:01.000Z world\n"]
    )
    mock_core.read_namespaced_pod.return_value.status.phase = "Succeeded"

    with mock.patch(
      "kinetic.backend.log_streaming.LiveOutputPanel"
    ) as mock_panel_cls:
      mock_panel = MagicMock()
      mock_panel_cls.return_value.__enter__.return_value = mock_panel
      mock_panel_cls.return_value.__exit__.return_value = False
      _stream_pod_logs(
        mock_core, "pod-1", "default", stop_event=threading.Event()
      )

    emitted = [c.args[0] for c in mock_panel.on_output.call_args_list]
    self.assertEqual(emitted, ["hello", "world"])

  def test_handles_carriage_returns(self):
    mock_core = MagicMock()
    mock_core.read_namespaced_pod_log.return_value = _make_resp(
      [b"2024-01-01T12:00:00.000Z 1/10\r2/10\r", b"3/10\n"]
    )
    mock_core.read_namespaced_pod.return_value.status.phase = "Succeeded"

    cursor = LogCursor(path=None)
    with mock.patch(
      "kinetic.backend.log_streaming.LiveOutputPanel"
    ) as mock_panel_cls:
      mock_panel = MagicMock()
      mock_panel_cls.return_value.__enter__.return_value = mock_panel
      mock_panel_cls.return_value.__exit__.return_value = False
      _stream_pod_logs(
        mock_core,
        "pod-1",
        "default",
        cursor=cursor,
        stop_event=threading.Event(),
      )

    emitted = [c.args[0] for c in mock_panel.on_output.call_args_list]
    self.assertEqual(emitted, ["3/10"])
    # Regression: \r stripping must not lose the timestamp prefix, so dedup
    # still records the cursor.
    self.assertEqual(cursor.since_time, "2024-01-01T12:00:00.000Z")

  def test_handles_multibyte_utf8_split_across_chunks(self):
    mock_core = MagicMock()
    # "héllo" → "h" + 0xc3 0xa9 + "llo" — break the 2-byte é across chunks.
    encoded = "2024-01-01T12:00:00.000Z héllo\n".encode("utf-8")
    split = encoded.index(b"\xa9")  # split right after the first é byte
    mock_core.read_namespaced_pod_log.return_value = _make_resp(
      [encoded[:split], encoded[split:]]
    )
    mock_core.read_namespaced_pod.return_value.status.phase = "Succeeded"

    with mock.patch(
      "kinetic.backend.log_streaming.LiveOutputPanel"
    ) as mock_panel_cls:
      mock_panel = MagicMock()
      mock_panel_cls.return_value.__enter__.return_value = mock_panel
      mock_panel_cls.return_value.__exit__.return_value = False
      _stream_pod_logs(
        mock_core, "pod-1", "default", stop_event=threading.Event()
      )

    emitted = [c.args[0] for c in mock_panel.on_output.call_args_list]
    self.assertEqual(emitted, ["héllo"])

  def test_releases_conn_on_api_exception(self):
    mock_core = MagicMock()
    resp = MagicMock()
    resp.stream.side_effect = ApiException(status=500, reason="boom")
    mock_core.read_namespaced_pod_log.return_value = resp
    # Pod still Running so we'd otherwise retry. Stop after one iter.
    mock_core.read_namespaced_pod.return_value.status.phase = "Running"

    stop_event = threading.Event()

    # Force the backoff sleep to immediately set stop_event and return True.
    with (
      mock.patch("kinetic.backend.log_streaming.Console") as mock_console_cls,
      mock.patch.object(stop_event, "wait", return_value=True),
    ):
      mock_console_cls.return_value.is_terminal = False
      _stream_pod_logs(mock_core, "pod-1", "default", stop_event=stop_event)

    resp.release_conn.assert_called_once()

  def test_404_on_open_breaks_immediately(self):
    mock_core = MagicMock()
    mock_core.read_namespaced_pod_log.side_effect = ApiException(
      status=404, reason="Not Found"
    )
    # Should not raise, should not retry, returns quickly.
    self._run_once(mock_core)
    mock_core.read_namespaced_pod_log.assert_called_once()

  def test_auth_failure_triggers_client_refresh(self):
    bad_core = MagicMock()
    bad_core.read_namespaced_pod_log.side_effect = ApiException(
      status=401, reason="Unauthorized"
    )
    bad_core.read_namespaced_pod.return_value.status.phase = "Running"

    fresh_core = MagicMock()
    fresh_core.read_namespaced_pod_log.return_value = _make_resp([])
    fresh_core.read_namespaced_pod.return_value.status.phase = "Succeeded"

    stop_event = threading.Event()
    with (
      mock.patch("kinetic.backend.log_streaming.Console") as mock_console_cls,
      mock.patch(
        "kinetic.backend.log_streaming._refresh_k8s_client",
        return_value=fresh_core,
      ) as refresh,
      # Backoff exits immediately to keep the test fast.
      mock.patch.object(stop_event, "wait", return_value=False),
    ):
      mock_console_cls.return_value.is_terminal = False
      _stream_pod_logs(bad_core, "pod-1", "default", stop_event=stop_event)

    refresh.assert_called()
    fresh_core.read_namespaced_pod_log.assert_called()

  def test_backoff_escalates_on_repeated_open_failures(self):
    """Repeated transient failures with no progress must escalate backoff."""
    mock_core = MagicMock()
    mock_core.read_namespaced_pod_log.side_effect = (
      urllib3.exceptions.ProtocolError("nope") for _ in range(50)
    )
    mock_core.read_namespaced_pod.return_value.status.phase = "Running"

    backoff_calls = []
    stop_event = threading.Event()

    def fake_backoff(attempt):
      backoff_calls.append(attempt)
      if len(backoff_calls) >= 4:
        stop_event.set()
      return 0.0

    with (
      mock.patch("kinetic.backend.log_streaming.Console") as mock_console_cls,
      mock.patch(
        "kinetic.backend.log_streaming._backoff_seconds",
        side_effect=fake_backoff,
      ),
    ):
      mock_console_cls.return_value.is_terminal = False
      _stream_pod_logs(mock_core, "pod-1", "default", stop_event=stop_event)

    self.assertEqual(backoff_calls, [1, 2, 3, 4])

  def test_backoff_resets_after_a_line_is_consumed(self):
    """A stream that delivers at least one line clears the failure counter."""
    mock_core = MagicMock()
    good_resp = _make_resp([b"2024-01-01T12:00:00.000Z progress!\n"])

    def open_side_effect(*args, **kwargs):
      if mock_core.read_namespaced_pod_log.call_count == 1:
        return good_resp
      raise urllib3.exceptions.ProtocolError("nope")

    mock_core.read_namespaced_pod_log.side_effect = open_side_effect
    mock_core.read_namespaced_pod.return_value.status.phase = "Running"

    backoff_calls = []
    stop_event = threading.Event()

    def fake_backoff(attempt):
      backoff_calls.append(attempt)
      if len(backoff_calls) >= 2:
        stop_event.set()
      return 0.0

    with (
      mock.patch(
        "kinetic.backend.log_streaming.LiveOutputPanel"
      ) as mock_panel_cls,
      mock.patch(
        "kinetic.backend.log_streaming._backoff_seconds",
        side_effect=fake_backoff,
      ),
    ):
      mock_panel = MagicMock()
      mock_panel_cls.return_value.__enter__.return_value = mock_panel
      mock_panel_cls.return_value.__exit__.return_value = False
      _stream_pod_logs(mock_core, "pod-1", "default", stop_event=stop_event)

    # After the successful first iteration the counter resets to 0, so the
    # next reconnect waits the minimum backoff. A subsequent failure escalates.
    self.assertEqual(backoff_calls, [0, 1])

  def test_cursor_deleted_on_terminal_pod(self):
    mock_core = MagicMock()
    mock_core.read_namespaced_pod_log.return_value = _make_resp([])
    mock_core.read_namespaced_pod.return_value.status.phase = "Succeeded"

    with tempfile.TemporaryDirectory() as tmp:
      cursor_path = Path(tmp) / "job" / "pod-1.json"
      cursor = LogCursor(path=cursor_path)
      cursor.record("2024-01-01T12:00:00.000Z", "h")
      cursor.flush()
      self.assertTrue(cursor_path.exists())

      with mock.patch(
        "kinetic.backend.log_streaming.Console"
      ) as mock_console_cls:
        mock_console_cls.return_value.is_terminal = False
        _stream_pod_logs(
          mock_core,
          "pod-1",
          "default",
          cursor=cursor,
          stop_event=threading.Event(),
        )

      self.assertFalse(
        cursor_path.exists(),
        "cursor file should be removed once the pod is terminal",
      )

  def test_cursor_kept_when_stop_event_set(self):
    """A user-initiated stop should preserve the cursor for next time."""
    mock_core = MagicMock()
    mock_core.read_namespaced_pod_log.side_effect = (
      urllib3.exceptions.ProtocolError("nope") for _ in range(50)
    )
    mock_core.read_namespaced_pod.return_value.status.phase = "Running"

    with tempfile.TemporaryDirectory() as tmp:
      cursor_path = Path(tmp) / "job" / "pod-1.json"
      cursor = LogCursor(path=cursor_path)
      cursor.record("2024-01-01T12:00:00.000Z", "h")
      cursor.flush()

      stop_event = threading.Event()
      stop_event.set()  # ask for shutdown before the loop even runs

      with mock.patch(
        "kinetic.backend.log_streaming.Console"
      ) as mock_console_cls:
        mock_console_cls.return_value.is_terminal = False
        _stream_pod_logs(
          mock_core,
          "pod-1",
          "default",
          cursor=cursor,
          stop_event=stop_event,
        )

      self.assertTrue(cursor_path.exists())

  def test_reconnects_after_transient_protocol_error(self):
    mock_core = MagicMock()
    bad_resp = MagicMock()
    bad_resp.stream.side_effect = urllib3.exceptions.ProtocolError(
      "Connection broken"
    )
    good_resp = _make_resp([b"2024-01-01T12:00:05.000Z after reconnect\n"])
    mock_core.read_namespaced_pod_log.side_effect = [bad_resp, good_resp]
    mock_core.read_namespaced_pod.return_value.status.phase = "Succeeded"

    stop_event = threading.Event()
    with (
      mock.patch(
        "kinetic.backend.log_streaming.LiveOutputPanel"
      ) as mock_panel_cls,
      mock.patch.object(stop_event, "wait", return_value=False),
    ):
      mock_panel = MagicMock()
      mock_panel_cls.return_value.__enter__.return_value = mock_panel
      mock_panel_cls.return_value.__exit__.return_value = False
      _stream_pod_logs(mock_core, "pod-1", "default", stop_event=stop_event)

    self.assertEqual(mock_core.read_namespaced_pod_log.call_count, 2)
    emitted = [c.args[0] for c in mock_panel.on_output.call_args_list]
    self.assertIn("after reconnect", emitted)


class TestLogStreamer(absltest.TestCase):
  def test_start_launches_daemon_thread(self):
    mock_core = MagicMock()
    with (
      mock.patch(
        "kinetic.backend.log_streaming._stream_pod_logs"
      ) as mock_stream,
      LogStreamer(mock_core, "default", job_id="job-1") as streamer,
    ):
      streamer.start("pod-1")
      self.assertIsNotNone(streamer._thread)
      self.assertTrue(streamer._thread.daemon)

    mock_stream.assert_called_once()

  def test_start_is_idempotent(self):
    mock_core = MagicMock()
    with (
      mock.patch(
        "kinetic.backend.log_streaming._stream_pod_logs"
      ) as mock_stream,
      LogStreamer(mock_core, "default", job_id="job-1") as streamer,
    ):
      streamer.start("pod-1")
      streamer.start("pod-1")
      streamer.start("pod-2")
    self.assertEqual(mock_stream.call_count, 1)

  def test_exit_sets_stop_event_and_joins(self):
    mock_core = MagicMock()
    observed = {}

    def fake_stream(*args, **kwargs):
      observed["stop_event"] = kwargs["stop_event"]
      # Block until stop_event is set, mimicking the real backoff path.
      kwargs["stop_event"].wait(timeout=3)

    with (
      mock.patch(
        "kinetic.backend.log_streaming._stream_pod_logs",
        side_effect=fake_stream,
      ),
      LogStreamer(mock_core, "default", job_id="job-1") as streamer,
    ):
      streamer.start("pod-1")
      # __exit__ runs at end of with, should set stop_event and join quickly.
      time.sleep(0.05)  # let thread enter wait()
      start = time.monotonic()
    elapsed = time.monotonic() - start

    self.assertTrue(observed["stop_event"].is_set())
    self.assertLess(
      elapsed, 2.0, "exit should not block until backoff times out"
    )

  def test_resume_false_disables_cursor_path(self):
    mock_core = MagicMock()
    captured = {}

    def fake_stream(*args, **kwargs):
      captured["cursor"] = kwargs["cursor"]

    with (
      mock.patch(
        "kinetic.backend.log_streaming._stream_pod_logs",
        side_effect=fake_stream,
      ),
      LogStreamer(
        mock_core, "default", job_id="job-1", resume=False
      ) as streamer,
    ):
      streamer.start("pod-1")

    self.assertIsNone(captured["cursor"]._path)

  def test_resume_true_uses_per_pod_cursor_path(self):
    mock_core = MagicMock()
    captured = {}

    def fake_stream(*args, **kwargs):
      captured["cursor"] = kwargs["cursor"]

    with tempfile.TemporaryDirectory() as tmp:
      cursor_dir = Path(tmp)
      with (
        mock.patch(
          "kinetic.backend.log_streaming._stream_pod_logs",
          side_effect=fake_stream,
        ),
        LogStreamer(
          mock_core,
          "default",
          job_id="job-abc",
          resume=True,
          cursor_dir=cursor_dir,
        ) as streamer,
      ):
        streamer.start("pod-xyz")
      self.assertEqual(
        captured["cursor"]._path,
        cursor_dir / "job-abc" / "pod-xyz.json",
      )


class TestParsing(absltest.TestCase):
  def test_parse_timestamp_with_fractional_seconds(self):
    ts, content = _parse_timestamped_line(
      "2024-01-01T12:00:00.123456789Z user line"
    )
    self.assertEqual(ts, "2024-01-01T12:00:00.123456789Z")
    self.assertEqual(content, "user line")

  def test_parse_timestamp_without_fractional_seconds(self):
    ts, content = _parse_timestamped_line("2024-01-01T12:00:00Z line")
    self.assertEqual(ts, "2024-01-01T12:00:00Z")
    self.assertEqual(content, "line")

  def test_parse_untimestamped_line_returns_none(self):
    ts, content = _parse_timestamped_line("not a timestamp")
    self.assertIsNone(ts)
    self.assertEqual(content, "not a timestamp")

  def test_truncate_to_second_strips_fractional(self):
    self.assertEqual(
      _truncate_to_second("2024-01-01T12:00:00.123456Z"),
      "2024-01-01T12:00:00Z",
    )
    self.assertEqual(
      _truncate_to_second("2024-01-01T12:00:00Z"), "2024-01-01T12:00:00Z"
    )

  def test_backoff_increases_and_caps(self):
    # Bounds (without jitter): 1, 2, 4, 8, 16, 30, 30, ...
    self.assertLess(_backoff_seconds(1), 2)
    self.assertLess(_backoff_seconds(2), 3)
    self.assertLessEqual(_backoff_seconds(20), 40)  # 30 + 25% jitter


class TestLogCursor(absltest.TestCase):
  def test_roundtrip_persists_last_ts_and_hashes(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "job-1" / "pod-1.json"
      cursor = LogCursor(path=path, write_interval_s=0)
      cursor.record("2024-01-01T12:00:00.000Z", "hash-a")
      cursor.record("2024-01-01T12:00:01.000Z", "hash-b")
      cursor.flush()

      reloaded = LogCursor(path=path)
      reloaded.load()
      self.assertEqual(reloaded.since_time, "2024-01-01T12:00:01.000Z")
      self.assertTrue(reloaded.is_duplicate("hash-a"))
      self.assertTrue(reloaded.is_duplicate("hash-b"))
      self.assertFalse(reloaded.is_duplicate("hash-c"))

  def test_missing_file_loads_as_empty(self):
    with tempfile.TemporaryDirectory() as tmp:
      cursor = LogCursor(path=Path(tmp) / "missing.json")
      cursor.load()
      self.assertIsNone(cursor.since_time)

  def test_corrupt_file_loads_as_empty(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "corrupt.json"
      path.write_text("this is not json {{{")
      cursor = LogCursor(path=path)
      cursor.load()
      self.assertIsNone(cursor.since_time)

  def test_record_throttles_writes(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "throttled.json"
      cursor = LogCursor(path=path, write_interval_s=999)
      cursor.record("2024-01-01T12:00:00.000Z", "hash-a")
      # No write yet because the throttle window hasn't elapsed.
      self.assertFalse(path.exists())
      cursor.flush()  # explicit flush bypasses throttle
      self.assertTrue(path.exists())
      data = json.loads(path.read_text())
      self.assertEqual(data["last_ts"], "2024-01-01T12:00:00.000Z")

  def test_none_path_is_memory_only(self):
    cursor = LogCursor(path=None)
    cursor.record("2024-01-01T12:00:00.000Z", "hash-a")
    cursor.flush()  # no error
    self.assertEqual(cursor.since_time, "2024-01-01T12:00:00.000Z")

  def test_safe_name_sanitizes(self):
    p = cursor_path_for(Path("/tmp/streams"), "job/with:slashes", "pod-1")
    self.assertEqual(p, Path("/tmp/streams/job_with_slashes/pod-1.json"))

  def test_safe_name_strips_dots_to_prevent_traversal(self):
    p = cursor_path_for(Path("/tmp/streams"), "..", "pod-1")
    self.assertEqual(p, Path("/tmp/streams/__/pod-1.json"))

  def test_cursor_file_is_user_only_readable(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "secure.json"
      cursor = LogCursor(path=path, write_interval_s=0)
      cursor.record("2024-01-01T12:00:00.000Z", "h")
      cursor.flush()
      mode = path.stat().st_mode & 0o777
      self.assertEqual(mode, 0o600)

  def test_delete_removes_cursor_file(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "to-delete.json"
      cursor = LogCursor(path=path, write_interval_s=0)
      cursor.record("2024-01-01T12:00:00.000Z", "h")
      cursor.flush()
      self.assertTrue(path.exists())
      cursor.delete()
      self.assertFalse(path.exists())

  def test_delete_on_missing_file_is_safe(self):
    with tempfile.TemporaryDirectory() as tmp:
      cursor = LogCursor(path=Path(tmp) / "never-written.json")
      cursor.delete()  # should not raise

  def test_clear_timestamp_resets_and_marks_dirty(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "c.json"
      cursor = LogCursor(path=path, write_interval_s=0)
      cursor.record("2024-01-01T12:00:00.000Z", "h")
      cursor.flush()
      self.assertEqual(cursor.since_time, "2024-01-01T12:00:00.000Z")

      cursor.clear_timestamp()
      self.assertIsNone(cursor.since_time)
      # A subsequent flush should persist the cleared state.
      cursor.flush()
      data = json.loads(path.read_text())
      self.assertIsNone(data["last_ts"])


if __name__ == "__main__":
  absltest.main()
