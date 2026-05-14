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
    mock.patch("kinetic.cli.commands.init.list_clusters", return_value=clusters)
  )


def _patch_kubectl(testcase):
  testcase.enterContext(
    mock.patch("kinetic.cli.commands.init.configure_kubectl")
  )


def _patch_load_state_zone(testcase, zone):
  """Make load_state() return a StackState whose stack outputs include `zone`.

  init's _infer_zone reads the zone from stack outputs (not from
  state.zone), so the mocked stack must expose `outputs()["zone"].value`.
  """
  stack = mock.MagicMock()
  stack.outputs.return_value = {"zone": mock.MagicMock(value=zone)}
  testcase.enterContext(
    mock.patch(
      "kinetic.cli.commands.init.load_state",
      return_value=StackState(
        project="test-proj",
        zone=zone,
        cluster_name="c",
        stack=stack,
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
  """Create-path behaviors not already covered by the forwarding tests."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    _patch_prereqs(self)
    _patch_resolve_project(self)
    _patch_list_clusters(self, [])
    self.up_mock = _patch_up(self)
    self.profiles_path = _isolate_profiles(self)

  def test_name_flag_forwarded_to_up(self):
    result = self.runner.invoke(init, ["--yes", "--profile-name", "my-prof"])
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
    result = self.runner.invoke(init, ["--profile-name", "mine"], input="\n")
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


class InitCreatePathForwardingTest(absltest.TestCase):
  """Regression for Codex P1: ctx.invoke(up, ...) bypasses Click's envvar
  resolution, so init must forward project/zone/cluster/namespace explicitly.
  """

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    _patch_prereqs(self)
    _patch_resolve_project(self)
    _patch_list_clusters(self, [])
    self.up_mock = _patch_up(self)
    self.profiles_path = _isolate_profiles(self)

  def test_env_vars_flow_to_up(self):
    env = {
      "KINETIC_PROFILES_FILE": str(self.profiles_path),
      "KINETIC_ZONE": "us-east1-b",
      "KINETIC_CLUSTER": "team-x",
      "KINETIC_NAMESPACE": "team-ns",
    }
    with mock.patch.dict("os.environ", env, clear=True):
      result = self.runner.invoke(init, ["--yes"])
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.up_mock.assert_called_once()
    _, kwargs = self.up_mock.call_args
    self.assertEqual(kwargs.get("zone"), "us-east1-b")
    self.assertEqual(kwargs.get("cluster_name"), "team-x")
    self.assertEqual(kwargs.get("namespace"), "team-ns")

  def test_flags_flow_to_up(self):
    result = self.runner.invoke(
      init,
      [
        "--yes",
        "--zone",
        "europe-west4-a",
        "--cluster",
        "explicit",
        "--namespace",
        "flag-ns",
      ],
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    _, kwargs = self.up_mock.call_args
    self.assertEqual(kwargs.get("zone"), "europe-west4-a")
    self.assertEqual(kwargs.get("cluster_name"), "explicit")
    self.assertEqual(kwargs.get("namespace"), "flag-ns")


class InitJoinZoneInferenceTest(absltest.TestCase):
  """Regression for Codex P1: _infer_zone must read the actual zone from
  stack outputs, not echo back the default we passed in.
  """

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    _patch_prereqs(self)
    _patch_resolve_project(self)
    self.kubectl_mock = mock.MagicMock()
    self.enterContext(
      mock.patch(
        "kinetic.cli.commands.init.configure_kubectl", self.kubectl_mock
      )
    )
    _patch_up(self)
    self.profiles_path = _isolate_profiles(self)

  def test_zone_comes_from_stack_outputs(self):
    _patch_list_clusters(self, ["dev-tpu"])
    _patch_load_state_zone(self, zone="asia-northeast1-a")
    result = self.runner.invoke(init, [], input="\n")
    self.assertEqual(result.exit_code, 0, msg=result.output)
    # kubectl must be configured against the real zone, not DEFAULT_ZONE.
    self.kubectl_mock.assert_called_once_with(
      "dev-tpu", "asia-northeast1-a", "test-proj"
    )
    data = json.loads(self.profiles_path.read_text())
    self.assertEqual(data["profiles"]["dev-tpu"]["zone"], "asia-northeast1-a")

  def test_missing_zone_output_raises_actionable_error(self):
    _patch_list_clusters(self, ["dev-tpu"])
    stack = mock.MagicMock()
    stack.outputs.return_value = {}  # 'zone' missing
    self.enterContext(
      mock.patch(
        "kinetic.cli.commands.init.load_state",
        return_value=StackState(
          project="test-proj", zone="placeholder", cluster_name="c", stack=stack
        ),
      )
    )
    result = self.runner.invoke(init, [], input="\n")
    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("zone", result.output.lower())


class InitActivatesNewlySavedProfileTest(absltest.TestCase):
  """Regression for Codex P2: upsert_profile alone does not activate a
  newly saved profile when another profile is already current.
  """

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    _patch_prereqs(self)
    _patch_resolve_project(self)
    _patch_kubectl(self)
    _patch_load_state_zone(self, zone="us-west4-a")
    _patch_up(self)
    self.profiles_path = _isolate_profiles(self)
    # Pre-seed an existing active profile so upsert_profile's auto-activate
    # heuristic (which only fires when 'current' is None) does NOT apply.
    self.profiles_path.parent.mkdir(parents=True, exist_ok=True)
    self.profiles_path.write_text(
      json.dumps(
        {
          "current": "old",
          "profiles": {
            "old": {
              "project": "p",
              "zone": "z",
              "cluster": "c",
              "namespace": "n",
            }
          },
        }
      )
    )

  def test_join_path_activates_new_profile(self):
    _patch_list_clusters(self, ["dev-tpu"])
    result = self.runner.invoke(init, [], input="\n")
    self.assertEqual(result.exit_code, 0, msg=result.output)
    data = json.loads(self.profiles_path.read_text())
    self.assertEqual(data["current"], "dev-tpu")


class InitChoicePromptExplainerTest(absltest.TestCase):
  """The prompt itself must explain the consequences of each choice so the
  user can decide without leaving the terminal.
  """

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    _patch_prereqs(self)
    _patch_resolve_project(self)
    _patch_kubectl(self)
    _patch_load_state_zone(self, zone="us-west4-a")
    self.up_mock = _patch_up(self)
    self.profiles_path = _isolate_profiles(self)

  def test_no_clusters_path_shows_create_explainer(self):
    _patch_list_clusters(self, [])
    result = self.runner.invoke(init, ["--yes"])
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("No existing Kinetic clusters", result.output)
    self.assertIn("Creating a new cluster will:", result.output)
    self.assertIn("GKE cluster", result.output)
    self.assertIn("incur", result.output)
    self.assertIn("kinetic down", result.output)

  def test_no_clusters_prompt_can_be_declined(self):
    _patch_list_clusters(self, [])
    # No --yes, decline the confirmation.
    result = self.runner.invoke(init, [], input="n\n")
    self.assertNotEqual(result.exit_code, 0)
    self.up_mock.assert_not_called()

  def test_clusters_present_prompt_shows_both_choices(self):
    _patch_list_clusters(self, ["dev-tpu", "team-x"])
    # Two prompts on this path: join/create (default join), then cluster
    # selection (default dev-tpu). Accept both defaults.
    result = self.runner.invoke(init, [], input="\n\n")
    self.assertEqual(result.exit_code, 0, msg=result.output)
    # Both option names + their consequences are surfaced.
    self.assertIn("dev-tpu", result.output)
    self.assertIn("team-x", result.output)
    self.assertIn("join", result.output)
    self.assertIn("create", result.output)
    self.assertIn("Configure kubectl", result.output)
    self.assertIn("Provision a NEW GKE cluster", result.output)


if __name__ == "__main__":
  absltest.main()
