"""Tests for kinetic.cli.profiles — data model, I/O, resolution."""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

from absl.testing import absltest

from kinetic.cli import profiles


def _tmp(testcase):
  """Return a fresh tempdir Path that is cleaned up on test teardown."""
  td = tempfile.TemporaryDirectory()
  testcase.addCleanup(td.cleanup)
  return Path(td.name)


def _patched_path(tmp):
  return mock.patch.object(profiles, "PROFILES_FILE", tmp / "profiles.json")


class ValidateNameTest(absltest.TestCase):
  def test_valid_names(self):
    for name in ["dev", "dev-tpu", "team_shared", "a1", "a" * 64]:
      profiles.validate_name(name)

  def test_invalid_names(self):
    for name in ["", "-bad", "_bad", "has space", "has/slash", "a" * 65]:
      with self.assertRaises(profiles.ProfileError):
        profiles.validate_name(name)


class LoadStoreTest(absltest.TestCase):
  def test_missing_file_returns_empty(self):
    with mock.patch.object(
      profiles, "PROFILES_FILE", Path("/nonexistent/x.json")
    ):
      current, p = profiles.load_store()
    self.assertIsNone(current)
    self.assertEqual(p, {})

  def test_round_trip(self):
    tmp = _tmp(self)
    with _patched_path(tmp):
      p = profiles.Profile(
        name="dev",
        project="proj-1",
        zone="us-east1-b",
        cluster="my-cluster",
        namespace="ns",
      )
      profiles.upsert_profile(p)
      current, loaded = profiles.load_store()
    self.assertEqual(current, "dev")
    self.assertEqual(loaded["dev"].project, "proj-1")
    self.assertEqual(loaded["dev"].namespace, "ns")

  def test_first_upsert_becomes_current(self):
    tmp = _tmp(self)
    with _patched_path(tmp):
      profiles.upsert_profile(profiles.Profile("a", "p", "z", "c"))
      profiles.upsert_profile(profiles.Profile("b", "p2", "z2", "c2"))
      current, loaded = profiles.load_store()
    self.assertEqual(current, "a")
    self.assertEqual(set(loaded), {"a", "b"})

  def test_malformed_json_raises(self):
    tmp = _tmp(self) / "profiles.json"
    tmp.write_text("not json")
    with (
      mock.patch.object(profiles, "PROFILES_FILE", tmp),
      self.assertRaises(profiles.ProfileError),
    ):
      profiles.load_store()

  def test_stale_current_is_dropped(self):
    tmp = _tmp(self) / "profiles.json"
    tmp.write_text(
      json.dumps(
        {
          "current": "missing",
          "profiles": {
            "a": {
              "project": "p",
              "zone": "z",
              "cluster": "c",
              "namespace": "default",
            }
          },
        }
      )
    )
    with mock.patch.object(profiles, "PROFILES_FILE", tmp):
      current, _ = profiles.load_store()
    self.assertIsNone(current)

  def test_missing_required_field_raises(self):
    tmp = _tmp(self) / "profiles.json"
    tmp.write_text(
      json.dumps(
        {"current": None, "profiles": {"bad": {"project": "p", "zone": "z"}}}
      )
    )
    with (
      mock.patch.object(profiles, "PROFILES_FILE", tmp),
      self.assertRaises(profiles.ProfileError),
    ):
      profiles.load_store()


class SetCurrentTest(absltest.TestCase):
  def test_set_current_to_existing_profile(self):
    tmp = _tmp(self)
    with _patched_path(tmp):
      profiles.upsert_profile(profiles.Profile("a", "p", "z", "c"))
      profiles.upsert_profile(profiles.Profile("b", "p2", "z2", "c2"))
      profiles.set_current("b")
      current, _ = profiles.load_store()
    self.assertEqual(current, "b")

  def test_set_current_to_missing_raises(self):
    tmp = _tmp(self)
    with _patched_path(tmp), self.assertRaises(profiles.ProfileError):
      profiles.set_current("nope")


class RemoveProfileTest(absltest.TestCase):
  def test_remove_clears_current(self):
    tmp = _tmp(self)
    with _patched_path(tmp):
      profiles.upsert_profile(profiles.Profile("a", "p", "z", "c"))
      profiles.remove_profile("a")
      current, loaded = profiles.load_store()
    self.assertIsNone(current)
    self.assertEqual(loaded, {})

  def test_remove_missing_raises(self):
    tmp = _tmp(self)
    with _patched_path(tmp), self.assertRaises(profiles.ProfileError):
      profiles.remove_profile("nope")

  def test_remove_non_current_keeps_current(self):
    tmp = _tmp(self)
    with _patched_path(tmp):
      profiles.upsert_profile(profiles.Profile("a", "p", "z", "c"))
      profiles.upsert_profile(profiles.Profile("b", "p2", "z2", "c2"))
      profiles.remove_profile("b")
      current, loaded = profiles.load_store()
    self.assertEqual(current, "a")
    self.assertEqual(set(loaded), {"a"})


class ResolveActiveTest(absltest.TestCase):
  def test_explicit_name_wins(self):
    tmp = _tmp(self)
    with _patched_path(tmp), mock.patch.dict(os.environ, {}, clear=False):
      profiles.upsert_profile(profiles.Profile("a", "pa", "z", "c"))
      profiles.upsert_profile(profiles.Profile("b", "pb", "z", "c"))
      os.environ.pop("KINETIC_PROFILE", None)
      result = profiles.resolve_active(explicit_name="b")
    self.assertEqual(result.project, "pb")

  def test_env_var_beats_current(self):
    tmp = _tmp(self)
    with (
      _patched_path(tmp),
      mock.patch.dict(os.environ, {"KINETIC_PROFILE": "b"}),
    ):
      profiles.upsert_profile(profiles.Profile("a", "pa", "z", "c"))
      profiles.upsert_profile(profiles.Profile("b", "pb", "z", "c"))
      result = profiles.resolve_active()
    self.assertEqual(result.project, "pb")

  def test_falls_back_to_current(self):
    tmp = _tmp(self)
    with _patched_path(tmp), mock.patch.dict(os.environ, {}, clear=False):
      profiles.upsert_profile(profiles.Profile("a", "pa", "z", "c"))
      os.environ.pop("KINETIC_PROFILE", None)
      result = profiles.resolve_active()
    self.assertEqual(result.name, "a")

  def test_missing_explicit_raises(self):
    tmp = _tmp(self)
    with _patched_path(tmp), self.assertRaises(profiles.ProfileError):
      profiles.resolve_active(explicit_name="nope")

  def test_no_profiles_returns_none(self):
    tmp = _tmp(self)
    with _patched_path(tmp), mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop("KINETIC_PROFILE", None)
      result = profiles.resolve_active()
    self.assertIsNone(result)


class AtomicWriteTest(absltest.TestCase):
  def test_creates_parent_dir(self):
    tmp = _tmp(self)
    nested = tmp / "sub" / "profiles.json"
    with mock.patch.object(profiles, "PROFILES_FILE", nested):
      profiles.upsert_profile(profiles.Profile("a", "p", "z", "c"))
    self.assertTrue(nested.exists())
    data = json.loads(nested.read_text())
    self.assertEqual(data["current"], "a")


class StateBackendFieldTest(absltest.TestCase):
  def test_round_trip_gcs_sentinel(self):
    tmp = _tmp(self)
    with _patched_path(tmp):
      profiles.upsert_profile(
        profiles.Profile("dev", "p", "z", "c", state_backend="gcs")
      )
      _, loaded = profiles.load_store()
    self.assertEqual(loaded["dev"].state_backend, "gcs")

  def test_round_trip_explicit_url(self):
    tmp = _tmp(self)
    with _patched_path(tmp):
      profiles.upsert_profile(
        profiles.Profile(
          "dev", "p", "z", "c", state_backend="gs://team-a/state"
        )
      )
      _, loaded = profiles.load_store()
    self.assertEqual(loaded["dev"].state_backend, "gs://team-a/state")

  def test_old_profile_without_field_loads(self):
    """Existing profile JSON without the new field still loads cleanly."""
    tmp = _tmp(self) / "profiles.json"
    tmp.write_text(
      json.dumps(
        {
          "current": "old",
          "profiles": {
            "old": {
              "project": "p",
              "zone": "z",
              "cluster": "c",
              "namespace": "default",
            }
          },
        }
      )
    )
    with mock.patch.object(profiles, "PROFILES_FILE", tmp):
      _, loaded = profiles.load_store()
    self.assertIsNone(loaded["old"].state_backend)

  def test_serialization_omits_none(self):
    """Profiles with state_backend=None must not write the key."""
    tmp = _tmp(self)
    with _patched_path(tmp):
      profiles.upsert_profile(profiles.Profile("dev", "p", "z", "c"))
    data = json.loads((tmp / "profiles.json").read_text())
    self.assertNotIn("state_backend", data["profiles"]["dev"])

  def test_serialization_includes_set_value(self):
    tmp = _tmp(self)
    with _patched_path(tmp):
      profiles.upsert_profile(
        profiles.Profile("dev", "p", "z", "c", state_backend="gcs")
      )
    data = json.loads((tmp / "profiles.json").read_text())
    self.assertEqual(data["profiles"]["dev"]["state_backend"], "gcs")


if __name__ == "__main__":
  absltest.main()
