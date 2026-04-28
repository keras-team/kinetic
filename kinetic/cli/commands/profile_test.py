"""Tests for kinetic.cli.commands.profile and profile->default_map wiring."""

import json
import tempfile
from pathlib import Path
from unittest import mock

from absl.testing import absltest
from click.testing import CliRunner

from kinetic.cli.commands.profile import profile as profile_cmd
from kinetic.cli.main import cli


def _tmp(testcase):
  td = tempfile.TemporaryDirectory()
  testcase.addCleanup(td.cleanup)
  return Path(td.name)


def _make_runner_env(tmp_path):
  """Build an env dict that isolates profiles storage to tmp_path.

  KINETIC_PROFILES_FILE is honored by kinetic.cli.profiles._profiles_path.
  """
  return {"KINETIC_PROFILES_FILE": str(tmp_path / "profiles.json")}


class ProfileCreateTest(absltest.TestCase):
  def test_create_with_flags(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    result = runner.invoke(
      profile_cmd,
      [
        "create",
        "dev",
        "--project",
        "p1",
        "--zone",
        "us-east1-b",
        "--cluster",
        "c1",
        "--namespace",
        "ns1",
      ],
      env=env,
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    data = json.loads((tmp / "profiles.json").read_text())
    self.assertEqual(data["current"], "dev")
    self.assertEqual(data["profiles"]["dev"]["project"], "p1")

  def test_create_refuses_overwrite_without_force(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    args = [
      "create",
      "dev",
      "--project",
      "p1",
      "--zone",
      "z",
      "--cluster",
      "c",
      "--namespace",
      "n",
    ]
    runner.invoke(profile_cmd, args, env=env)
    result = runner.invoke(profile_cmd, args, env=env)
    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("already exists", result.output)

  def test_create_prompts_for_missing_fields(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    # Order: project, zone (default accepted), cluster (default accepted),
    # namespace (default accepted), state-backend (default 'local').
    result = runner.invoke(
      profile_cmd,
      ["create", "dev"],
      input="my-proj\n\n\n\n\n",
      env=env,
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    data = json.loads((tmp / "profiles.json").read_text())
    self.assertEqual(data["profiles"]["dev"]["project"], "my-proj")


class ProfileCreateStateBackendTest(absltest.TestCase):
  def test_default_local_choice_stores_local_explicitly(self):
    """Picking 'local' interactively persists 'local' (not None) so the
    profile keeps overriding any global state-backend setting."""
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    # Order: project, zone (default), cluster (default), namespace (default),
    # state-backend choice (default 'local').
    result = runner.invoke(
      profile_cmd,
      ["create", "dev"],
      input="my-proj\n\n\n\n\n",
      env=env,
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    data = json.loads((tmp / "profiles.json").read_text())
    self.assertEqual(data["profiles"]["dev"]["state_backend"], "local")

  def test_gcs_choice_stores_sentinel(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    result = runner.invoke(
      profile_cmd,
      ["create", "dev"],
      input="my-proj\n\n\n\ngcs\n",
      env=env,
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    data = json.loads((tmp / "profiles.json").read_text())
    self.assertEqual(data["profiles"]["dev"]["state_backend"], "gcs")

  def test_custom_choice_with_url(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    result = runner.invoke(
      profile_cmd,
      ["create", "dev"],
      input="my-proj\n\n\n\ncustom\ngs://my-team-bucket\n",
      env=env,
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    data = json.loads((tmp / "profiles.json").read_text())
    self.assertEqual(
      data["profiles"]["dev"]["state_backend"], "gs://my-team-bucket"
    )

  def test_flag_overrides_prompt(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    result = runner.invoke(
      profile_cmd,
      [
        "create",
        "dev",
        "--project",
        "p",
        "--zone",
        "z",
        "--cluster",
        "c",
        "--namespace",
        "n",
        "--state-backend",
        "gs://flag-bucket",
      ],
      env=env,
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    data = json.loads((tmp / "profiles.json").read_text())
    self.assertEqual(
      data["profiles"]["dev"]["state_backend"], "gs://flag-bucket"
    )

  def test_local_flag_stores_local_explicitly(self):
    """`--state-backend local` is an explicit opt-out — must persist as
    'local' so it overrides any global `kinetic config set` value."""
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    result = runner.invoke(
      profile_cmd,
      [
        "create",
        "dev",
        "--project",
        "p",
        "--zone",
        "z",
        "--cluster",
        "c",
        "--namespace",
        "n",
        "--state-backend",
        "local",
      ],
      env=env,
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    data = json.loads((tmp / "profiles.json").read_text())
    self.assertEqual(data["profiles"]["dev"]["state_backend"], "local")

  def test_no_prompt_no_flag_stays_unset(self):
    """All other fields supplied non-interactively, no --state-backend
    flag → state_backend stays unset (not 'local'), so global settings
    can still apply."""
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    result = runner.invoke(
      profile_cmd,
      [
        "create",
        "dev",
        "--project",
        "p",
        "--zone",
        "z",
        "--cluster",
        "c",
        "--namespace",
        "n",
      ],
      env=env,
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    data = json.loads((tmp / "profiles.json").read_text())
    self.assertNotIn("state_backend", data["profiles"]["dev"])


class ProfileLsTest(absltest.TestCase):
  def test_ls_empty(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    result = runner.invoke(profile_cmd, ["ls"], env=env)
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("No profiles", result.output)

  def test_ls_marks_active(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    for name in ["alpha", "beta"]:
      runner.invoke(
        profile_cmd,
        [
          "create",
          name,
          "--project",
          "p",
          "--zone",
          "z",
          "--cluster",
          "c",
          "--namespace",
          "n",
        ],
        env=env,
      )
    result = runner.invoke(profile_cmd, ["ls"], env=env)
    self.assertEqual(result.exit_code, 0, msg=result.output)
    # First-created profile becomes current.
    self.assertIn("alpha", result.output)
    self.assertIn("beta", result.output)
    self.assertIn("Active profile: alpha", result.output)


class ProfileUseTest(absltest.TestCase):
  def test_use_switches_active(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    for name in ["alpha", "beta"]:
      runner.invoke(
        profile_cmd,
        [
          "create",
          name,
          "--project",
          "p",
          "--zone",
          "z",
          "--cluster",
          "c",
          "--namespace",
          "n",
        ],
        env=env,
      )
    result = runner.invoke(profile_cmd, ["use", "beta"], env=env)
    self.assertEqual(result.exit_code, 0, msg=result.output)
    data = json.loads((tmp / "profiles.json").read_text())
    self.assertEqual(data["current"], "beta")

  def test_use_missing_profile_errors(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    result = runner.invoke(profile_cmd, ["use", "nope"], env=env)
    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("does not exist", result.output)


class ProfileShowTest(absltest.TestCase):
  def test_show_active(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    runner.invoke(
      profile_cmd,
      [
        "create",
        "dev",
        "--project",
        "proj-1",
        "--zone",
        "z",
        "--cluster",
        "c",
        "--namespace",
        "n",
      ],
      env=env,
    )
    # `profile show` reads the active selection from ctx.obj, which is
    # populated by the root cli group — invoke through that path.
    with mock.patch.dict("os.environ", env, clear=True):
      result = runner.invoke(cli, ["profile", "show"])
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("proj-1", result.output)

  def test_show_no_active_errors(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    with mock.patch.dict("os.environ", env, clear=True):
      result = runner.invoke(cli, ["profile", "show"])
    self.assertNotEqual(result.exit_code, 0)


class ProfileRmTest(absltest.TestCase):
  def test_rm_with_yes(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    runner.invoke(
      profile_cmd,
      [
        "create",
        "dev",
        "--project",
        "p",
        "--zone",
        "z",
        "--cluster",
        "c",
        "--namespace",
        "n",
      ],
      env=env,
    )
    result = runner.invoke(profile_cmd, ["rm", "dev", "--yes"], env=env)
    self.assertEqual(result.exit_code, 0, msg=result.output)
    data = json.loads((tmp / "profiles.json").read_text())
    self.assertEqual(data["profiles"], {})

  def test_rm_missing_errors(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    result = runner.invoke(profile_cmd, ["rm", "nope", "--yes"], env=env)
    self.assertNotEqual(result.exit_code, 0)


class DefaultMapIntegrationTest(absltest.TestCase):
  """Verify the root group injects profile fields as Click defaults.

  We target `kinetic config show` because it reads resolved project/zone
  without making any cloud calls.
  """

  def test_active_profile_shows_as_source(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    # Create + activate a profile.
    runner.invoke(
      profile_cmd,
      [
        "create",
        "mine",
        "--project",
        "super-proj",
        "--zone",
        "us-west4-b",
        "--cluster",
        "my-cluster",
        "--namespace",
        "my-ns",
      ],
      env=env,
    )
    # Invoke `config show` via the root cli with no env KINETIC_* set.
    # mock.patch.dict with clear=True ensures no KINETIC_* overrides
    # leak in from the parent process.
    with mock.patch.dict("os.environ", env, clear=True):
      result = runner.invoke(cli, ["config", "show"])
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("super-proj", result.output)
    self.assertIn("my-cluster", result.output)
    self.assertIn("my-ns", result.output)
    self.assertIn("profile", result.output)

  def test_env_var_overrides_profile(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    runner.invoke(
      profile_cmd,
      [
        "create",
        "mine",
        "--project",
        "from-profile",
        "--zone",
        "us-west4-b",
        "--cluster",
        "c",
        "--namespace",
        "n",
      ],
      env=env,
    )
    override_env = dict(env)
    override_env["KINETIC_PROJECT"] = "from-env"
    with mock.patch.dict("os.environ", override_env, clear=True):
      result = runner.invoke(cli, ["config", "show"])
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("from-env", result.output)

  def test_explicit_profile_flag(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    for name, proj in [("a", "proj-a"), ("b", "proj-b")]:
      runner.invoke(
        profile_cmd,
        [
          "create",
          name,
          "--project",
          proj,
          "--zone",
          "z",
          "--cluster",
          "c",
          "--namespace",
          "n",
        ],
        env=env,
      )
    # 'a' is current (first created). Explicit --profile b should win.
    with mock.patch.dict("os.environ", env, clear=True):
      result = runner.invoke(cli, ["--profile", "b", "config", "show"])
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("proj-b", result.output)


class ProfileResolutionErrorsTest(absltest.TestCase):
  def test_missing_explicit_profile_errors(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    with mock.patch.dict("os.environ", env, clear=True):
      result = runner.invoke(cli, ["--profile", "nope", "config", "show"])
    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("does not exist", result.output)


class ProfileGroupRespectsSelectorTest(absltest.TestCase):
  """Regression: --profile / KINETIC_PROFILE must be honored by the profile
  command group itself, not just commands that consume it via default_map.
  """

  def _seed_two_profiles(self, tmp):
    env = _make_runner_env(tmp)
    runner = CliRunner()
    # 'a' is created first and becomes the stored 'current'.
    for name, proj in [("a", "proj-a"), ("b", "proj-b")]:
      runner.invoke(
        profile_cmd,
        [
          "create",
          name,
          "--project",
          proj,
          "--zone",
          "z",
          "--cluster",
          "c",
          "--namespace",
          "n",
        ],
        env=env,
      )
    return env

  def test_profile_show_respects_explicit_flag(self):
    tmp = _tmp(self)
    env = self._seed_two_profiles(tmp)
    runner = CliRunner()
    with mock.patch.dict("os.environ", env, clear=True):
      result = runner.invoke(cli, ["--profile", "b", "profile", "show"])
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("proj-b", result.output)
    self.assertNotIn("proj-a", result.output)

  def test_profile_show_respects_env_var(self):
    tmp = _tmp(self)
    env = self._seed_two_profiles(tmp)
    env["KINETIC_PROFILE"] = "b"
    runner = CliRunner()
    with mock.patch.dict("os.environ", env, clear=True):
      result = runner.invoke(cli, ["profile", "show"])
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("proj-b", result.output)
    self.assertNotIn("proj-a", result.output)

  def test_profile_ls_marks_overridden_active(self):
    tmp = _tmp(self)
    env = self._seed_two_profiles(tmp)
    runner = CliRunner()
    with mock.patch.dict("os.environ", env, clear=True):
      result = runner.invoke(cli, ["--profile", "b", "profile", "ls"])
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("Active profile: b", result.output)
    self.assertIn("override", result.output)
    self.assertIn("stored: a", result.output)

  def test_profile_show_falls_back_to_stored_current(self):
    tmp = _tmp(self)
    env = self._seed_two_profiles(tmp)
    runner = CliRunner()
    # No --profile, no KINETIC_PROFILE — use stored current 'a'.
    with mock.patch.dict("os.environ", env, clear=True):
      result = runner.invoke(cli, ["profile", "show"])
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("proj-a", result.output)


class ProfileCreateEnvVarsTest(absltest.TestCase):
  """Regression: `profile create` must consume KINETIC_* env vars rather
  than always prompting when a flag is omitted.
  """

  def test_env_vars_skip_prompts(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    env.update(
      {
        "KINETIC_PROJECT": "env-proj",
        "KINETIC_ZONE": "env-zone",
        "KINETIC_CLUSTER": "env-cluster",
        "KINETIC_NAMESPACE": "env-ns",
      }
    )
    # Passing no input on stdin — if the command prompts, the runner
    # will produce a non-zero exit from the empty-input stream.
    result = runner.invoke(profile_cmd, ["create", "dev"], env=env, input="")
    self.assertEqual(result.exit_code, 0, msg=result.output)
    data = json.loads((tmp / "profiles.json").read_text())
    self.assertEqual(data["profiles"]["dev"]["project"], "env-proj")
    self.assertEqual(data["profiles"]["dev"]["zone"], "env-zone")
    self.assertEqual(data["profiles"]["dev"]["cluster"], "env-cluster")
    self.assertEqual(data["profiles"]["dev"]["namespace"], "env-ns")
    self.assertNotIn("state_backend", data["profiles"]["dev"])

  def test_flag_overrides_env_var(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    env["KINETIC_PROJECT"] = "env-proj"
    result = runner.invoke(
      profile_cmd,
      [
        "create",
        "dev",
        "--project",
        "flag-proj",
        "--zone",
        "z",
        "--cluster",
        "c",
        "--namespace",
        "n",
      ],
      env=env,
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    data = json.loads((tmp / "profiles.json").read_text())
    self.assertEqual(data["profiles"]["dev"]["project"], "flag-proj")


if __name__ == "__main__":
  absltest.main()
