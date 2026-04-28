"""Tests for kinetic.cli.settings — persisted global settings."""

import json
import tempfile
from pathlib import Path
from unittest import mock

from absl.testing import absltest

from kinetic.cli import settings


def _tmp(testcase):
  td = tempfile.TemporaryDirectory()
  testcase.addCleanup(td.cleanup)
  return Path(td.name)


def _patched_path(tmp):
  return mock.patch.object(settings, "SETTINGS_FILE", tmp / "settings.json")


class LoadTest(absltest.TestCase):
  def test_missing_file_returns_empty(self):
    with mock.patch.object(
      settings, "SETTINGS_FILE", Path("/nonexistent/x.json")
    ):
      self.assertEqual(settings.load(), {})

  def test_round_trip(self):
    tmp = _tmp(self)
    with _patched_path(tmp):
      settings.set_("state_backend", "gcs")
      self.assertEqual(settings.get("state_backend"), "gcs")

  def test_get_missing_returns_none(self):
    tmp = _tmp(self)
    with _patched_path(tmp):
      self.assertIsNone(settings.get("state_backend"))

  def test_malformed_json_raises(self):
    tmp = _tmp(self) / "settings.json"
    tmp.write_text("not json")
    with (
      mock.patch.object(settings, "SETTINGS_FILE", tmp),
      self.assertRaises(settings.SettingsError),
    ):
      settings.load()

  def test_non_object_top_level_raises(self):
    tmp = _tmp(self) / "settings.json"
    tmp.write_text(json.dumps([1, 2, 3]))
    with (
      mock.patch.object(settings, "SETTINGS_FILE", tmp),
      self.assertRaises(settings.SettingsError),
    ):
      settings.load()

  def test_get_non_string_value_raises(self):
    tmp = _tmp(self) / "settings.json"
    tmp.write_text(json.dumps({"state_backend": 42}))
    with (
      mock.patch.object(settings, "SETTINGS_FILE", tmp),
      self.assertRaises(settings.SettingsError),
    ):
      settings.get("state_backend")


class SetTest(absltest.TestCase):
  def test_set_unknown_key_raises(self):
    tmp = _tmp(self)
    with _patched_path(tmp), self.assertRaises(settings.SettingsError):
      settings.set_("not_a_real_key", "x")

  def test_set_non_string_raises(self):
    tmp = _tmp(self)
    with _patched_path(tmp), self.assertRaises(settings.SettingsError):
      settings.set_("state_backend", 42)  # type: ignore[arg-type]

  def test_set_overwrites_existing(self):
    tmp = _tmp(self)
    with _patched_path(tmp):
      settings.set_("state_backend", "gcs")
      settings.set_("state_backend", "gs://other")
      self.assertEqual(settings.get("state_backend"), "gs://other")


class UnsetTest(absltest.TestCase):
  def test_unset_present_key(self):
    tmp = _tmp(self)
    with _patched_path(tmp):
      settings.set_("state_backend", "gcs")
      settings.unset("state_backend")
      self.assertIsNone(settings.get("state_backend"))

  def test_unset_missing_key_is_noop(self):
    tmp = _tmp(self)
    with _patched_path(tmp):
      settings.unset("state_backend")  # no-op, no error


class EnvOverrideTest(absltest.TestCase):
  def test_settings_file_env_override(self):
    tmp = _tmp(self)
    custom = tmp / "custom.json"
    with mock.patch.dict("os.environ", {"KINETIC_SETTINGS_FILE": str(custom)}):
      settings.set_("state_backend", "gcs")
      self.assertTrue(custom.exists())
      self.assertEqual(settings.get("state_backend"), "gcs")


class AtomicWriteTest(absltest.TestCase):
  def test_creates_parent_dir(self):
    tmp = _tmp(self)
    nested = tmp / "sub" / "settings.json"
    with mock.patch.object(settings, "SETTINGS_FILE", nested):
      settings.set_("state_backend", "gcs")
    self.assertTrue(nested.exists())
    data = json.loads(nested.read_text())
    self.assertEqual(data["state_backend"], "gcs")

  def test_no_partial_file_on_error(self):
    tmp = _tmp(self)
    target = tmp / "settings.json"
    with (
      mock.patch.object(settings, "SETTINGS_FILE", target),
      mock.patch("os.replace", side_effect=OSError("disk full")),
      self.assertRaises(OSError),
    ):
      settings.set_("state_backend", "gcs")
    # No tmp file or target file should remain.
    leftovers = list(tmp.glob(".settings-*.json.tmp"))
    self.assertEqual(leftovers, [])
    self.assertFalse(target.exists())


if __name__ == "__main__":
  absltest.main()
