"""Tests for kinetic.backend.log_streaming — live pod log streaming."""

from unittest import mock
from unittest.mock import MagicMock

from absl.testing import absltest
from kubernetes.client.rest import ApiException

from kinetic.backend.log_streaming import (
  LogStreamer,
  _stream_pod_logs,
)


class TestStreamPodLogs(absltest.TestCase):
  """Tests for the top-level _stream_pod_logs orchestrator."""

  def _make_mock_resp(self, chunks):
    mock_resp = MagicMock()
    mock_resp.stream.return_value = chunks
    return mock_resp

  def test_calls_log_api_correctly(self):
    mock_core = MagicMock()
    mock_core.read_namespaced_pod_log.return_value = self._make_mock_resp([])

    with mock.patch(
      "kinetic.backend.log_streaming.Console"
    ) as mock_console_cls:
      mock_console_cls.return_value.is_terminal = False
      _stream_pod_logs(mock_core, "pod-1", "default")

    mock_core.read_namespaced_pod_log.assert_called_once_with(
      name="pod-1",
      namespace="default",
      follow=True,
      _preload_content=False,
    )

  def test_handles_partial_lines(self):
    mock_core = MagicMock()
    # "hello\nwor" then "ld\n" — "world" is split across chunks
    mock_core.read_namespaced_pod_log.return_value = self._make_mock_resp(
      [b"hello\nwor", b"ld\n"]
    )

    with mock.patch(
      "kinetic.backend.log_streaming.LiveOutputPanel"
    ) as mock_panel_cls:
      mock_panel = MagicMock()
      mock_panel_cls.return_value.__enter__ = MagicMock(return_value=mock_panel)
      mock_panel_cls.return_value.__exit__ = MagicMock(return_value=False)
      _stream_pod_logs(mock_core, "pod-1", "default")

    lines = [call[0][0] for call in mock_panel.on_output.call_args_list]
    self.assertEqual(lines, ["hello", "world"])

  def test_handles_carriage_returns(self):
    mock_core = MagicMock()
    # "1/10\r2/10\r3/10\n"
    mock_core.read_namespaced_pod_log.return_value = self._make_mock_resp(
      [b"1/10\r2/10\r", b"3/10\n"]
    )

    with mock.patch(
      "kinetic.backend.log_streaming.LiveOutputPanel"
    ) as mock_panel_cls:
      mock_panel = MagicMock()
      mock_panel_cls.return_value.__enter__ = MagicMock(return_value=mock_panel)
      mock_panel_cls.return_value.__exit__ = MagicMock(return_value=False)
      _stream_pod_logs(mock_core, "pod-1", "default")

    lines = [call[0][0] for call in mock_panel.on_output.call_args_list]
    self.assertEqual(lines, ["3/10"])

  def test_releases_conn_on_api_exception(self):
    mock_core = MagicMock()
    mock_resp = MagicMock()
    mock_resp.stream.side_effect = ApiException(status=404, reason="Not Found")
    mock_core.read_namespaced_pod_log.return_value = mock_resp

    with mock.patch(
      "kinetic.backend.log_streaming.Console"
    ) as mock_console_cls:
      mock_console_cls.return_value.is_terminal = False
      _stream_pod_logs(mock_core, "pod-1", "default")

    mock_resp.release_conn.assert_called_once()

  def test_handles_error_before_response(self):
    mock_core = MagicMock()
    mock_core.read_namespaced_pod_log.side_effect = ApiException(
      status=404, reason="Not Found"
    )

    # Should not raise even when resp is None
    _stream_pod_logs(mock_core, "pod-1", "default")

  def test_logs_warning_on_unexpected_error(self):
    mock_core = MagicMock()
    mock_resp = MagicMock()
    mock_resp.stream.side_effect = ValueError("something unexpected")
    mock_core.read_namespaced_pod_log.return_value = mock_resp

    with (
      mock.patch("kinetic.backend.log_streaming.Console") as mock_console_cls,
      mock.patch("kinetic.backend.log_streaming.logging") as mock_log,
    ):
      mock_console_cls.return_value.is_terminal = False
      _stream_pod_logs(mock_core, "pod-1", "default")

    mock_log.warning.assert_called_once()
    self.assertIn("pod-1", mock_log.warning.call_args[0][1])
    mock_resp.release_conn.assert_called_once()


class TestLogStreamer(absltest.TestCase):
  def test_start_launches_daemon_thread(self):
    mock_core = MagicMock()

    with (
      mock.patch(
        "kinetic.backend.log_streaming._stream_pod_logs"
      ) as mock_stream,
      LogStreamer(mock_core, "default") as streamer,
    ):
      streamer.start("pod-1")
      self.assertIsNotNone(streamer._thread)
      self.assertTrue(streamer._thread.daemon)

    mock_stream.assert_called_once_with(mock_core, "pod-1", "default")

  def test_start_is_idempotent(self):
    mock_core = MagicMock()

    with (
      mock.patch(
        "kinetic.backend.log_streaming._stream_pod_logs"
      ) as mock_stream,
      LogStreamer(mock_core, "default") as streamer,
    ):
      streamer.start("pod-1")
      streamer.start("pod-1")
      streamer.start("pod-2")  # different name, still no-op

    mock_stream.assert_called_once_with(mock_core, "pod-1", "default")

  def test_exit_joins_thread(self):
    mock_core = MagicMock()
    mock_thread = MagicMock()

    with (
      mock.patch(
        "kinetic.backend.log_streaming.threading.Thread",
        return_value=mock_thread,
      ),
      LogStreamer(mock_core, "ns") as streamer,
    ):
      streamer.start("pod-1")

    mock_thread.join.assert_called_once_with(timeout=5)

  def test_exit_without_start_is_noop(self):
    mock_core = MagicMock()
    # Should not raise
    with LogStreamer(mock_core, "default"):
      pass


if __name__ == "__main__":
  absltest.main()
