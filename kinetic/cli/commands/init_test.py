"""Tests for kinetic.cli.commands.init — onboarding orchestration."""

import json
import tempfile
from pathlib import Path
from unittest import mock

import click
from absl.testing import absltest
from click.testing import CliRunner

from kinetic.cli.commands.init import init
from kinetic.cli.infra.state import StackState


def _tmp(testcase):
  td = tempfile.TemporaryDirectory()
  testcase.addCleanup(td.cleanup)
  return Path(td.name)


def _patch_prereqs(testcase, *, all_ok=True):
  """Stub the four prereq checks. When all_ok=False, gcloud fails."""
  for name in (
    "check_gcloud",
    "check_kubectl",
    "check_gke_auth_plugin",
    "check_gcloud_auth",
  ):
    target = f"kinetic.cli.commands.init.{name}"
    side = None if all_ok else click.ClickException("missing")
    testcase.enterContext(mock.patch(target, side_effect=side))


def _patch_resolve_project(testcase, value="test-proj"):
  testcase.enterContext(
    mock.patch("kinetic.cli.commands.init.resolve_project", return_value=value)
  )


def _patch_list_clusters(testcase, clusters):
  testcase.enterContext(
    mock.patch(
      "kinetic.cli.commands.init.list_clusters", return_value=clusters
    )
  )


def _patch_kubectl(testcase):
  testcase.enterContext(
    mock.patch("kinetic.cli.commands.init.configure_kubectl")
  )


def _patch_load_state_zone(testcase, zone):
  testcase.enterContext(
    mock.patch(
      "kinetic.cli.commands.init.load_state",
      return_value=StackState(
        project="test-proj",
        zone=zone,
        cluster_name="c",
      ),
    )
  )


def _patch_up(testcase):
  """Stub the `up` command that `init` invokes in the Create path.

  Returns the MagicMock standing in for `up` so callers can assert it was
  (or was not) invoked.
  """
  up_mock = mock.MagicMock()
  # ctx.invoke(up, ...) expects `up` to behave like a Click command.
  # The simplest faithful stub: replace the symbol imported inside init.py.
  testcase.enterContext(mock.patch("kinetic.cli.commands.init.up", up_mock))
  return up_mock


def _isolate_profiles(testcase):
  path = str(_tmp(testcase) / "profiles.json")
  testcase.enterContext(
    mock.patch.dict("os.environ", {"KINETIC_PROFILES_FILE": path})
  )
  return Path(path)


class InitCreatePathTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    _patch_prereqs(self)
    _patch_resolve_project(self)
    _patch_list_clusters(self, [])
    self.up_mock = _patch_up(self)
    self.profiles_path = _isolate_profiles(self)

  def test_no_clusters_routes_to_create(self):
    result = self.runner.invoke(init, [])
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.up_mock.assert_called_once()

  def test_name_flag_forwarded_to_up(self):
    result = self.runner.invoke(init, ["--name", "my-prof"])
    self.assertEqual(result.exit_code, 0, msg=result.output)
    _, kwargs = self.up_mock.call_args
    self.assertEqual(kwargs.get("profile_name"), "my-prof")


class InitJoinPathTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    _patch_prereqs(self)
    _patch_resolve_project(self)
    _patch_kubectl(self)
    _patch_load_state_zone(self, zone="us-west4-a")
    self.up_mock = _patch_up(self)
    self.profiles_path = _isolate_profiles(self)

  def test_single_cluster_defaults_to_join(self):
    _patch_list_clusters(self, ["dev-tpu"])
    # Accept the default "join" at the prompt.
    result = self.runner.invoke(init, [], input="\n")
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.up_mock.assert_not_called()
    data = json.loads(self.profiles_path.read_text())
    self.assertEqual(data["current"], "dev-tpu")
    self.assertEqual(data["profiles"]["dev-tpu"]["cluster"], "dev-tpu")
    self.assertEqual(data["profiles"]["dev-tpu"]["zone"], "us-west4-a")

  def test_multiple_clusters_user_picks(self):
    _patch_list_clusters(self, ["alpha", "beta", "gamma"])
    # Accept prompt to join (default), then pick 'beta'.
    result = self.runner.invoke(init, [], input="\nbeta\n")
    self.assertEqual(result.exit_code, 0, msg=result.output)
    data = json.loads(self.profiles_path.read_text())
    self.assertEqual(data["current"], "beta")

  def test_choose_create_despite_clusters_present(self):
    _patch_list_clusters(self, ["dev-tpu"])
    result = self.runner.invoke(init, [], input="create\n")
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.up_mock.assert_called_once()
    # Profile is NOT saved by init — `up` is responsible in this branch.
    # The test's `up_mock` doesn't actually write, so profiles.json stays empty.

  def test_name_flag_overrides_cluster_name_in_join(self):
    _patch_list_clusters(self, ["dev-tpu"])
    result = self.runner.invoke(init, ["--name", "mine"], input="\n")
    self.assertEqual(result.exit_code, 0, msg=result.output)
    data = json.loads(self.profiles_path.read_text())
    self.assertIn("mine", data["profiles"])
    self.assertEqual(data["current"], "mine")


class InitPrereqFailureTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    _patch_prereqs(self, all_ok=False)
    _patch_list_clusters(self, [])
    self.up_mock = _patch_up(self)
    self.profiles_path = _isolate_profiles(self)

  def test_missing_prereq_exits_with_doctor_hint(self):
    result = self.runner.invoke(init, [])
    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("kinetic doctor", result.output)
    # No invocation of `up` when prereqs fail.
    self.up_mock.assert_not_called()


if __name__ == "__main__":
  absltest.main()
