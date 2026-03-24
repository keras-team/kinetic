"""Tests for kinetic.cli.output — LiveOutputPanel."""

from unittest import mock

from absl.testing import absltest
from rich.console import Console
from rich.text import Text

from kinetic.cli.output import LiveOutputPanel


def _make_non_terminal_console():
  """Create a Console that reports is_terminal=False."""
  return Console(force_terminal=False, file=open("/dev/null", "w"))


class MakePanelTest(absltest.TestCase):
  """Tests for _make_panel rendering logic."""

  def test_shows_last_max_lines(self):
    panel = LiveOutputPanel("Title", max_lines=3)
    for i in range(10):
      panel._lines.append(f"line {i}")

    content = panel._make_panel().renderable

    self.assertNotIn("line 6", content)
    self.assertIn("line 7", content)
    self.assertIn("line 9", content)

  def test_error_shows_all_lines(self):
    panel = LiveOutputPanel("Title", max_lines=3)
    for i in range(10):
      panel._lines.append(f"line {i}")
    panel._has_error = True

    content = panel._make_panel().renderable

    self.assertIn("line 0", content)
    self.assertIn("line 9", content)

  def test_subtitle_suppressed_on_error(self):
    panel = LiveOutputPanel("Title")
    panel._has_error = True

    self.assertIsNone(panel._make_panel().subtitle)

  def test_subtitle_suppressed_when_show_subtitle_false(self):
    panel = LiveOutputPanel("Title", show_subtitle=False)
    panel._start_time = 0
    panel._phrase_order = list(range(10))

    self.assertIsNone(panel._make_panel().subtitle)


class TransientBehaviorTest(absltest.TestCase):
  """Tests for transient panel clearing/persistence on exit."""

  def test_transient_clears_on_success(self):
    panel = LiveOutputPanel("Test", transient=True)
    panel._live = mock.MagicMock()
    panel.on_output("some output")

    panel.__exit__(None, None, None)

    update_calls = [
      c
      for c in panel._live.update.call_args_list
      if isinstance(c.args[0], Text)
    ]
    self.assertLen(update_calls, 1)

  def test_transient_persists_on_mark_error(self):
    panel = LiveOutputPanel("Test", transient=True)
    panel._live = mock.MagicMock()
    panel.on_output("some output")
    panel.mark_error()

    panel.__exit__(None, None, None)

    update_calls = [
      c
      for c in panel._live.update.call_args_list
      if isinstance(c.args[0], Text)
    ]
    self.assertEmpty(update_calls)

  def test_transient_persists_on_exception(self):
    console = _make_non_terminal_console()
    panel = LiveOutputPanel("Test", transient=True, target_console=console)

    with self.assertRaises(RuntimeError), panel:
      raise RuntimeError("fail")

    self.assertTrue(panel._has_error)

  def test_exception_sets_has_error_without_mark_error(self):
    console = _make_non_terminal_console()
    panel = LiveOutputPanel("Test", target_console=console)

    with self.assertRaises(TypeError), panel:
      raise TypeError("bad")

    self.assertTrue(panel._has_error)


if __name__ == "__main__":
  absltest.main()
