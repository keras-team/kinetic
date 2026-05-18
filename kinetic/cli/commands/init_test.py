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


def _patch_run_diagnostics(testcase, returns=True):
  """Stub run_diagnostics so init's troubleshoot path doesn't shell out
  to gcloud/kubectl/the GCP SDKs. Returns the mock for assertions."""
  m = mock.MagicMock(return_value=returns)
  testcase.enterContext(
    mock.patch("kinetic.cli.commands.init.run_diagnostics", m)
  )
  return m


def _isolate_profiles(testcase):
  path = str(_tmp(testcase) / "profiles.json")
  testcase.enterContext(
    mock.patch.dict("os.environ", {"KINETIC_PROFILES_FILE": path})
  )
  return Path(path)


def _seed_profiles(
  profiles_path, clusters, *, project="test-proj", zone="us-west4-a"
):
  """Write a profiles.json containing one profile per cluster name.

  Each profile's ``cluster`` field equals its name; the profile name is
  the same as the cluster so the troubleshoot picker shows them. Used by
  tests that exercise the join-first model — only clusters with a saved
  profile are picker-selectable.
  """
  profiles_path.parent.mkdir(parents=True, exist_ok=True)
  profiles_path.write_text(
    json.dumps(
      {
        "current": None,
        "profiles": {
          c: {
            "project": project,
            "zone": zone,
            "cluster": c,
            "namespace": "default",
          }
          for c in clusters
        },
      }
    )
  )


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
  """When prereqs are missing, init should offer to run troubleshoot
  rather than the old 'go run kinetic doctor' hint (which referenced a
  command that no longer exists)."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    _patch_prereqs(self, all_ok=False)
    _patch_list_clusters(self, [])
    self.up_mock = _patch_up(self)
    self.diag_mock = _patch_run_diagnostics(self)
    self.profiles_path = _isolate_profiles(self)

  def test_missing_prereq_offers_troubleshoot(self):
    # Accept the "Run troubleshoot now?" confirm, then leave the
    # cluster-name prompt blank (env-only checks).
    result = self.runner.invoke(init, [], input="y\n\n")
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.diag_mock.assert_called_once()
    self.up_mock.assert_not_called()

  def test_missing_prereq_declined_exits_cleanly(self):
    # Decline the troubleshoot confirm — should exit non-zero with no
    # diagnostics or `up` invocation.
    result = self.runner.invoke(init, [], input="n\n")
    self.assertNotEqual(result.exit_code, 0)
    self.diag_mock.assert_not_called()
    self.up_mock.assert_not_called()

  def test_missing_prereq_with_yes_runs_troubleshoot_automatically(self):
    # --yes opts into troubleshoot without prompting for confirmation.
    result = self.runner.invoke(init, ["--yes"], input="\n")
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.diag_mock.assert_called_once()

  def test_missing_prereq_path_honors_zone_override(self):
    """`--zone` provided to init must reach diagnostics even when the
    troubleshoot path is entered via the prereq-failure branch."""
    result = self.runner.invoke(
      init, ["--yes", "--zone", "asia-east1-a"], input="\n"
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    _, kwargs = self.diag_mock.call_args
    self.assertEqual(kwargs.get("zone"), "asia-east1-a")


class InitCreatePathForwardingTest(absltest.TestCase):
  """`ctx.invoke(up, ...)` bypasses Click's envvar resolution, so `init` must
  forward project/zone/cluster/namespace explicitly.
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
          project="test-proj",
          zone="placeholder",
          cluster_name="c",
          stack=stack,
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
    self.diag_mock = _patch_run_diagnostics(self)
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

  def test_no_clusters_path_shows_troubleshoot_option(self):
    """When no clusters exist, the prompt must still offer troubleshoot
    so the user can investigate (vs. only being able to create)."""
    _patch_list_clusters(self, [])
    # Pick 'troubleshoot' at the prompt, then leave the cluster-name
    # prompt blank to run env-only checks.
    result = self.runner.invoke(init, [], input="troubleshoot\n\n")
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("troubleshoot", result.output)
    self.diag_mock.assert_called_once()
    self.up_mock.assert_not_called()


class InitTroubleshootPathTest(absltest.TestCase):
  """Tests for the troubleshoot path added to init when doctor was rolled in."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    _patch_prereqs(self)
    _patch_resolve_project(self)
    _patch_load_state_zone(self, zone="us-west4-a")
    self.up_mock = _patch_up(self)
    self.diag_mock = _patch_run_diagnostics(self)
    self.profiles_path = _isolate_profiles(self)

  def test_troubleshoot_with_single_joined_cluster_picks_it(self):
    _patch_list_clusters(self, ["dev-tpu"])
    _seed_profiles(self.profiles_path, ["dev-tpu"])
    # Path choice: 'troubleshoot'; cluster picker default = the only cluster.
    result = self.runner.invoke(init, [], input="troubleshoot\n\n")
    self.assertEqual(result.exit_code, 0, msg=result.output)
    _, kwargs = self.diag_mock.call_args
    self.assertEqual(kwargs.get("cluster_name"), "dev-tpu")
    self.assertEqual(kwargs.get("project"), "test-proj")

  def test_troubleshoot_with_multiple_joined_clusters_user_picks(self):
    _patch_list_clusters(self, ["alpha", "beta", "gamma"])
    _seed_profiles(self.profiles_path, ["alpha", "beta", "gamma"])
    # Path choice: 'troubleshoot'; pick 'beta' from the picker.
    result = self.runner.invoke(init, [], input="troubleshoot\nbeta\n")
    self.assertEqual(result.exit_code, 0, msg=result.output)
    _, kwargs = self.diag_mock.call_args
    self.assertEqual(kwargs.get("cluster_name"), "beta")

  def test_troubleshoot_picker_only_shows_joined_clusters(self):
    """Clusters in the GCS bucket but not in saved profiles are not
    selectable — the join-first model requires explicit join."""
    # GCS bucket has three clusters; the user has only joined 'alpha'.
    _patch_list_clusters(self, ["alpha", "beta", "gamma"])
    _seed_profiles(self.profiles_path, ["alpha"])
    # Picker only offers 'alpha'; default-accept it.
    result = self.runner.invoke(init, [], input="troubleshoot\n\n")
    self.assertEqual(result.exit_code, 0, msg=result.output)
    _, kwargs = self.diag_mock.call_args
    self.assertEqual(kwargs.get("cluster_name"), "alpha")
    # 'beta'/'gamma' should not have been offered.
    self.assertNotIn(
      "beta",
      result.output.split("Select cluster")[0].split("Joined clusters:")[1]
      if "Joined clusters:" in result.output
      else "",
    )

  def test_troubleshoot_with_no_joined_clusters_offers_env_only(self):
    """No profiles → tell the user to join first, but allow env-only
    checks as a fallback after a confirmation prompt."""
    _patch_list_clusters(self, [])
    # Accept the "Run environment-only checks?" confirmation.
    result = self.runner.invoke(init, [], input="troubleshoot\ny\n")
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("No joined clusters", result.output)
    self.assertIn("kinetic init", result.output)
    _, kwargs = self.diag_mock.call_args
    self.assertIsNone(kwargs.get("cluster_name"))

  def test_troubleshoot_with_no_joined_clusters_declined_exits(self):
    """Declining env-only checks exits cleanly with a join-first message."""
    _patch_list_clusters(self, [])
    result = self.runner.invoke(init, [], input="troubleshoot\nn\n")
    self.assertNotEqual(result.exit_code, 0)
    self.diag_mock.assert_not_called()

  def test_troubleshoot_does_not_save_profile(self):
    _patch_list_clusters(self, ["dev-tpu"])
    _seed_profiles(self.profiles_path, ["dev-tpu"])
    pre_contents = self.profiles_path.read_text()
    result = self.runner.invoke(init, [], input="troubleshoot\n\n")
    self.assertEqual(result.exit_code, 0, msg=result.output)
    # Troubleshoot is read-only — profile store unchanged from seed.
    self.assertEqual(self.profiles_path.read_text(), pre_contents)

  def test_troubleshoot_returns_nonzero_when_diagnostics_fail(self):
    _patch_list_clusters(self, ["dev-tpu"])
    _seed_profiles(self.profiles_path, ["dev-tpu"])
    self.diag_mock.return_value = False
    result = self.runner.invoke(init, [], input="troubleshoot\n\n")
    self.assertNotEqual(result.exit_code, 0)

  def test_cluster_override_must_be_joined(self):
    """--cluster pointing at an unjoined cluster errors with a join hint
    rather than silently targeting an unknown cluster."""
    _patch_list_clusters(self, ["alpha", "beta"])
    _seed_profiles(self.profiles_path, ["alpha"])
    result = self.runner.invoke(
      init, ["--cluster", "secret-cluster"], input="troubleshoot\n"
    )
    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("not in your saved profiles", result.output)
    self.assertIn("kinetic init", result.output)
    self.diag_mock.assert_not_called()

  def test_cluster_override_when_joined_is_used(self):
    """--cluster naming a joined cluster skips the picker."""
    _patch_list_clusters(self, ["alpha", "beta"])
    _seed_profiles(self.profiles_path, ["alpha", "beta"])
    result = self.runner.invoke(
      init, ["--cluster", "beta"], input="troubleshoot\n"
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    _, kwargs = self.diag_mock.call_args
    self.assertEqual(kwargs.get("cluster_name"), "beta")

  def test_troubleshoot_pulls_zone_from_saved_profile(self):
    """The zone in run_diagnostics comes from the joined cluster's profile."""
    _patch_list_clusters(self, ["dev-tpu"])
    _seed_profiles(self.profiles_path, ["dev-tpu"], zone="europe-west4-c")
    result = self.runner.invoke(init, [], input="troubleshoot\n\n")
    self.assertEqual(result.exit_code, 0, msg=result.output)
    _, kwargs = self.diag_mock.call_args
    self.assertEqual(kwargs.get("zone"), "europe-west4-c")

  def test_troubleshoot_zone_flag_overrides_saved_profile(self):
    """An explicit --zone wins over the saved-profile zone lookup."""
    _patch_list_clusters(self, ["dev-tpu"])
    _seed_profiles(self.profiles_path, ["dev-tpu"], zone="europe-west4-c")
    result = self.runner.invoke(
      init, ["--zone", "asia-east1-a"], input="troubleshoot\n\n"
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    _, kwargs = self.diag_mock.call_args
    self.assertEqual(kwargs.get("zone"), "asia-east1-a")

  def test_troubleshoot_prints_target_header(self):
    """The header tells the user exactly what's being diagnosed."""
    _patch_list_clusters(self, ["dev-tpu"])
    _seed_profiles(self.profiles_path, ["dev-tpu"], zone="us-west4-a")
    result = self.runner.invoke(init, [], input="troubleshoot\n\n")
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("Troubleshooting target", result.output)
    self.assertIn("test-proj", result.output)
    self.assertIn("dev-tpu", result.output)
    self.assertIn("us-west4-a", result.output)


if __name__ == "__main__":
  absltest.main()
