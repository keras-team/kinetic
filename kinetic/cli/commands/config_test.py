"""Tests for kinetic.cli.commands.config — show / set / unset."""

import json
import tempfile
from pathlib import Path
from unittest import mock

from absl.testing import absltest
from click.testing import CliRunner

from kinetic.cli import settings
from kinetic.cli.commands.config import config as config_cmd
from kinetic.cli.commands.profile import profile as profile_cmd
from kinetic.cli.main import cli


def _tmp(testcase):
  td = tempfile.TemporaryDirectory()
  testcase.addCleanup(td.cleanup)
  return Path(td.name)


def _make_runner_env(tmp_path):
  return {
    "KINETIC_PROFILES_FILE": str(tmp_path / "profiles.json"),
    "KINETIC_SETTINGS_FILE": str(tmp_path / "settings.json"),
  }


class ConfigSetTest(absltest.TestCase):
  def test_set_state_backend_persists(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    result = runner.invoke(config_cmd, ["set", "state-backend", "gcs"], env=env)
    self.assertEqual(result.exit_code, 0, msg=result.output)
    data = json.loads((tmp / "settings.json").read_text())
    self.assertEqual(data["state_backend"], "gcs")

  def test_set_state_backend_explicit_url(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    result = runner.invoke(
      config_cmd,
      ["set", "state-backend", "gs://my-bucket/prefix"],
      env=env,
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    data = json.loads((tmp / "settings.json").read_text())
    self.assertEqual(data["state_backend"], "gs://my-bucket/prefix")

  def test_set_invalid_value_rejected(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    result = runner.invoke(
      config_cmd, ["set", "state-backend", "garbage://nope"], env=env
    )
    self.assertNotEqual(result.exit_code, 0)
    self.assertFalse((tmp / "settings.json").exists())

  def test_set_unknown_key_rejected(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    result = runner.invoke(config_cmd, ["set", "bogus", "value"], env=env)
    self.assertNotEqual(result.exit_code, 0)


class ConfigUnsetTest(absltest.TestCase):
  def test_unset_removes_key(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    runner.invoke(config_cmd, ["set", "state-backend", "gcs"], env=env)
    result = runner.invoke(config_cmd, ["unset", "state-backend"], env=env)
    self.assertEqual(result.exit_code, 0, msg=result.output)
    data = json.loads((tmp / "settings.json").read_text())
    self.assertNotIn("state_backend", data)

  def test_unset_missing_key_succeeds(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    result = runner.invoke(config_cmd, ["unset", "state-backend"], env=env)
    self.assertEqual(result.exit_code, 0, msg=result.output)


class ConfigShowSourceTest(absltest.TestCase):
  """Source attribution for the State Backend row."""

  def test_default_when_nothing_set(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    with mock.patch.dict("os.environ", env, clear=True):
      result = runner.invoke(cli, ["config", "show"])
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("State Backend", result.output)
    self.assertIn("default", result.output)

  def test_settings_source_when_only_global_set(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    with mock.patch.dict("os.environ", env, clear=True):
      runner.invoke(cli, ["config", "set", "state-backend", "gcs"])
      result = runner.invoke(cli, ["config", "show"])
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("settings", result.output)

  def test_profile_source_beats_settings(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    # Persist a global setting first.
    runner.invoke(
      config_cmd, ["set", "state-backend", "gs://from-global"], env=env
    )
    # Create a profile with its own state_backend.
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
        "--state-backend",
        "gs://from-profile",
      ],
      env=env,
    )
    with mock.patch.dict("os.environ", env, clear=True):
      result = runner.invoke(cli, ["config", "show"])
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("from-profile", result.output)
    # Source column should attribute to profile, not settings.
    self.assertIn("profile", result.output)

  def test_env_source_beats_profile(self):
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
        "--state-backend",
        "gs://from-profile",
      ],
      env=env,
    )
    override_env = dict(env)
    override_env["KINETIC_STATE_BACKEND"] = "gs://from-env"
    with mock.patch.dict("os.environ", override_env, clear=True):
      result = runner.invoke(cli, ["config", "show"])
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("from-env", result.output)
    self.assertIn("KINETIC_STATE_BACKEND", result.output)

  def test_profile_local_overrides_global_gcs(self):
    """A profile that explicitly opts into 'local' must override a
    persisted global `state_backend = gcs` — the documented precedence."""
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    runner.invoke(config_cmd, ["set", "state-backend", "gcs"], env=env)
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
        "--state-backend",
        "local",
      ],
      env=env,
    )
    with mock.patch.dict("os.environ", env, clear=True):
      result = runner.invoke(cli, ["config", "show"])
    self.assertEqual(result.exit_code, 0, msg=result.output)
    # Profile says local → file:// must win over the global gcs setting.
    self.assertIn("file://", result.output)
    self.assertIn("profile", result.output)

  def test_global_setting_drives_default_map_for_no_profile(self):
    """A user without any profile but with `kinetic config set` should
    have the value injected into Click's default_map for command flags."""
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    runner.invoke(config_cmd, ["set", "state-backend", "gcs"], env=env)
    # Sanity: profiles.json was never written.
    self.assertFalse((tmp / "profiles.json").exists())
    with mock.patch.dict("os.environ", env, clear=True):
      result = runner.invoke(cli, ["config", "show"])
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("settings", result.output)
    # No active profile → no profile.state_backend → settings wins.
    self.assertNotIn("Active profile", result.output)


class ConfigShowStateDirTest(absltest.TestCase):
  """The 'Pulumi State Dir' row only appears when the resolved backend
  is file://."""

  def test_state_dir_shown_for_local(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    with mock.patch.dict("os.environ", env, clear=True):
      result = runner.invoke(cli, ["config", "show"])
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertIn("Pulumi State Dir", result.output)

  def test_state_dir_hidden_for_gcs(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    runner.invoke(
      config_cmd, ["set", "state-backend", "gs://my-bucket"], env=env
    )
    with mock.patch.dict("os.environ", env, clear=True):
      result = runner.invoke(cli, ["config", "show"])
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertNotIn("Pulumi State Dir", result.output)


class SettingsErrorPropagatesTest(absltest.TestCase):
  """A malformed settings.json should fail the CLI, not silently mask
  user intent."""

  def test_malformed_settings_file_errors(self):
    runner = CliRunner()
    tmp = _tmp(self)
    env = _make_runner_env(tmp)
    (tmp / "settings.json").write_text("not json")
    # Force the settings module to read from the patched path.
    with mock.patch.dict("os.environ", env, clear=True):
      result = runner.invoke(cli, ["config", "show"])
    self.assertNotEqual(result.exit_code, 0)


# Silence unused-import warnings for `settings` in environments where the
# import is only for module side effects.
_ = settings


if __name__ == "__main__":
  absltest.main()
