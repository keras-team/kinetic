"""Tests for kinetic.debug — attach instructions and the attach window."""

import contextlib
import io
import json
import os
from unittest import mock

from absl.testing import absltest, parameterized

from kinetic.debug import (
  DEBUG_WAIT_TIMEOUT_ENV,
  DEFAULT_DEBUG_WAIT_TIMEOUT,
  print_attach_instructions,
  resolve_debug_wait_timeout,
)


def _capture(local_port=5678, working_dir=None):
  """Return the text print_attach_instructions() writes to stdout."""
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf):
    print_attach_instructions(local_port, working_dir)
  return buf.getvalue()


def _launch_config(output):
  """Parse the launch.json object out of the printed instructions.

  The snippet is printed for the user to paste into VS Code, so it has
  to be valid JSON. Parsing it here is the assertion: a malformed
  snippet fails the test instead of failing in the user's editor.
  """
  lines = output.splitlines()
  start = lines.index("  {")
  end = lines.index("  }", start)
  return json.loads("\n".join(lines[start : end + 1]))


class TestPrintAttachInstructions(absltest.TestCase):
  """The printed pathMappings must match what the runner actually does.

  Regression test for the snippet that hardcoded
  ``"remoteRoot": "/tmp/workspace"``. Since the runner extracts the
  workspace under a ``tempfile.mkdtemp(prefix="kinetic-run-")``
  directory and symlinks the client's own working_dir at it, that
  literal path is never where the sources are, and breakpoints set
  against it never bind.
  """

  def test_working_dir_gives_identity_mapping(self):
    working_dir = "/Users/dev/project"
    config = _launch_config(_capture(working_dir=working_dir))

    self.assertEqual(
      config["pathMappings"],
      [{"localRoot": working_dir, "remoteRoot": working_dir}],
    )

  def test_never_prints_stale_tmp_workspace_root(self):
    """The pod has no /tmp/workspace; it must not appear anywhere."""
    self.assertNotIn("/tmp/workspace", _capture(working_dir="/Users/dev/proj"))
    self.assertNotIn("/tmp/workspace", _capture())

  def test_no_working_dir_omits_path_mappings(self):
    """`kinetic jobs debug` has no working_dir; emit no mapping at all.

    An unmapped path is what debugpy already assumes, so leaving the
    entry out is better than printing a root that is a guess.
    """
    output = _capture()
    config = _launch_config(output)

    self.assertNotIn("pathMappings", config)
    # The old snippet fell back to this VS Code variable, which resolves
    # to whatever folder happens to be open — not the submit directory.
    self.assertNotIn("${workspaceFolder}", output)

  def test_windows_working_dir_stays_valid_json(self):
    """Backslashes must be escaped, not pasted raw into the snippet."""
    working_dir = r"C:\Users\dev\project"
    config = _launch_config(_capture(working_dir=working_dir))

    self.assertEqual(
      config["pathMappings"],
      [{"localRoot": working_dir, "remoteRoot": working_dir}],
    )

  def test_snippet_carries_the_forwarded_port(self):
    for working_dir in (None, "/Users/dev/project"):
      with self.subTest(working_dir=working_dir):
        config = _launch_config(_capture(4242, working_dir))

        self.assertEqual(config["connect"], {"host": "localhost", "port": 4242})
        self.assertEqual(config["type"], "debugpy")
        self.assertEqual(config["request"], "attach")


@contextlib.contextmanager
def _attach_window(value):
  """Set (or unset, when value is None) the attach-window env var."""
  with mock.patch.dict(os.environ, {}, clear=False):
    if value is None:
      os.environ.pop(DEBUG_WAIT_TIMEOUT_ENV, None)
    else:
      os.environ[DEBUG_WAIT_TIMEOUT_ENV] = value
    yield


class TestResolveDebugWaitTimeout(parameterized.TestCase):
  """KINETIC_DEBUG_WAIT_TIMEOUT is read on the client, not just the pod."""

  def test_default_is_ten_minutes(self):
    """Locks the value documented in docs/configuration.md."""
    self.assertEqual(DEFAULT_DEBUG_WAIT_TIMEOUT, 600)

  def test_unset_uses_default(self):
    with _attach_window(None):
      self.assertEqual(resolve_debug_wait_timeout(), 600)

  @parameterized.named_parameters(
    ("half_a_minute", "30", 30),
    ("thirty_minutes", "1800", 1800),
    ("surrounding_whitespace", " 900 ", 900),
  )
  def test_positive_value_is_honored(self, raw, expected):
    with _attach_window(raw):
      self.assertEqual(resolve_debug_wait_timeout(), expected)

  @parameterized.named_parameters(
    ("empty", ""),
    ("zero", "0"),
    ("negative", "-59"),
    ("not_a_number", "ten minutes"),
    ("fractional", "12.5"),
  )
  def test_invalid_value_falls_back_to_default(self, raw):
    """A bad value must not disable the wait or crash the submit."""
    with _attach_window(raw):
      self.assertEqual(resolve_debug_wait_timeout(), DEFAULT_DEBUG_WAIT_TIMEOUT)


if __name__ == "__main__":
  absltest.main()
