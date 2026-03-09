"""Tests for keras_remote.cli.commands.down — destroy infrastructure."""

from unittest import mock

from absl.testing import absltest
from click.testing import CliRunner

from keras_remote.cli.commands.down import down

# Shared CLI args that skip interactive prompts.
_CLI_ARGS = [
  "--project",
  "test-project",
  "--zone",
  "us-central2-b",
  "--yes",
]

# Patches applied to every test to bypass prerequisites and infrastructure.
_BASE_PATCHES = {
  "check_all": mock.patch("keras_remote.cli.commands.down.check_all"),
  "resolve_project": mock.patch(
    "keras_remote.cli.commands.down.resolve_project",
    return_value="test-project",
  ),
  "apply_destroy": mock.patch(
    "keras_remote.cli.commands.down.apply_destroy", return_value=True
  ),
}


def _start_patches(test_case):
  """Start all base patches and return a dict of mock objects."""
  mocks = {}
  for name, patcher in _BASE_PATCHES.items():
    mocks[name] = test_case.enterContext(patcher)
  return mocks


class DownCommandTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.mocks = _start_patches(self)

  def test_successful_destroy(self):
    """Successful destroy — exit code 0, 'Cleanup Complete' shown."""
    result = self.runner.invoke(down, _CLI_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Cleanup Complete", result.output)
    self.mocks["apply_destroy"].assert_called_once()

  def test_destroy_failure_still_shows_summary(self):
    """apply_destroy returns False — summary still displayed."""
    self.mocks["apply_destroy"].return_value = False

    result = self.runner.invoke(down, _CLI_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Cleanup Complete", result.output)
    self.assertIn("Check manually", result.output)

  def test_abort_on_no_confirmation(self):
    """User declines confirmation — apply_destroy not called."""
    args = ["--project", "test-project", "--zone", "us-central2-b"]

    self.runner.invoke(down, args, input="n\n")

    self.mocks["apply_destroy"].assert_not_called()

  def test_yes_flag_skips_confirmation(self):
    """--yes flag skips confirmation prompt."""
    result = self.runner.invoke(down, _CLI_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.mocks["apply_destroy"].assert_called_once()

  def test_config_passed_correctly(self):
    """InfraConfig args match CLI options."""
    args = [
      "--project",
      "my-proj",
      "--zone",
      "europe-west1-b",
      "--cluster",
      "my-cluster",
      "--yes",
    ]

    result = self.runner.invoke(down, args)

    self.assertEqual(result.exit_code, 0, result.output)
    config = self.mocks["apply_destroy"].call_args[0][0]
    self.assertEqual(config.project, "my-proj")
    self.assertEqual(config.zone, "europe-west1-b")
    self.assertEqual(config.cluster_name, "my-cluster")

  def test_resolve_project_allow_create_false(self):
    """When --project not given, resolve_project(allow_create=False) is called."""
    args = ["--zone", "us-central1-a", "--yes"]

    result = self.runner.invoke(down, args)

    self.assertEqual(result.exit_code, 0, result.output)
    self.mocks["resolve_project"].assert_called_once_with(allow_create=False)


if __name__ == "__main__":
  absltest.main()
