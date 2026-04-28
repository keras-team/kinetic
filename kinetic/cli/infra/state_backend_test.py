"""Tests for kinetic.cli.infra.state_backend."""

import os
import tempfile
from pathlib import Path
from unittest import mock

import click
from absl.testing import absltest
from google.api_core import exceptions as gax

from kinetic.cli import settings
from kinetic.cli.infra import state_backend


class NormalizeStateBackendUrlTest(absltest.TestCase):
  def test_none_resolves_to_local_default(self):
    url = state_backend.normalize_state_backend_url(None, "any-proj")
    self.assertTrue(url.startswith("file://"))

  def test_empty_string_resolves_to_local_default(self):
    url = state_backend.normalize_state_backend_url("", "any-proj")
    self.assertTrue(url.startswith("file://"))

  def test_local_keyword_resolves_to_local_default(self):
    url = state_backend.normalize_state_backend_url("local", "any-proj")
    self.assertTrue(url.startswith("file://"))

  def test_gcs_sentinel_uses_project(self):
    url = state_backend.normalize_state_backend_url("gcs", "my-proj")
    self.assertEqual(url, "gs://my-proj-kinetic-state")

  def test_gcs_sentinel_without_project_raises(self):
    with self.assertRaises(click.BadParameter):
      state_backend.normalize_state_backend_url("gcs", "")

  def test_explicit_gcs_url_passthrough(self):
    url = state_backend.normalize_state_backend_url(
      "gs://my-bucket", "any-proj"
    )
    self.assertEqual(url, "gs://my-bucket")

  def test_explicit_gcs_url_with_prefix_passthrough(self):
    url = state_backend.normalize_state_backend_url(
      "gs://my-bucket/team-a", "any-proj"
    )
    self.assertEqual(url, "gs://my-bucket/team-a")

  def test_invalid_gcs_url_raises(self):
    with self.assertRaises(click.BadParameter):
      state_backend.normalize_state_backend_url("gs://", "any-proj")

  def test_unsupported_scheme_raises(self):
    with self.assertRaises(click.BadParameter):
      state_backend.normalize_state_backend_url("s3://x", "any-proj")

  def test_typo_gcs_scheme_rejected(self):
    with self.assertRaises(click.BadParameter):
      state_backend.normalize_state_backend_url("gcs://my-bucket", "any-proj")

  def test_explicit_file_url_passthrough(self):
    url = state_backend.normalize_state_backend_url(
      "file:///tmp/kinetic-state", "any-proj"
    )
    self.assertEqual(url, "file:///tmp/kinetic-state")


class ResolveForShowTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.settings_path = Path(self.tmp.name) / "settings.json"
    self.enterContext(
      mock.patch.object(settings, "SETTINGS_FILE", self.settings_path)
    )

  def _clear_env(self):
    return mock.patch.dict(os.environ, {}, clear=True)

  def test_default_when_nothing_set(self):
    with self._clear_env():
      url, src = state_backend.resolve_state_backend_for_show(None, "p")
    self.assertTrue(url.startswith("file://"))
    self.assertEqual(src, "default")

  def test_settings_used_when_no_env_or_profile(self):
    with self._clear_env():
      settings.set_("state_backend", "gcs")
      url, src = state_backend.resolve_state_backend_for_show(None, "my-proj")
    self.assertEqual(url, "gs://my-proj-kinetic-state")
    self.assertEqual(src, "settings")

  def test_profile_beats_settings(self):
    profile = mock.Mock(state_backend="gs://from-profile")
    with self._clear_env():
      settings.set_("state_backend", "gcs")
      url, src = state_backend.resolve_state_backend_for_show(
        profile, "my-proj"
      )
    self.assertEqual(url, "gs://from-profile")
    self.assertEqual(src, "profile")

  def test_env_beats_profile(self):
    profile = mock.Mock(state_backend="gs://from-profile")
    with mock.patch.dict(
      os.environ, {"KINETIC_STATE_BACKEND": "gs://from-env"}, clear=True
    ):
      url, src = state_backend.resolve_state_backend_for_show(profile, "p")
    self.assertEqual(url, "gs://from-env")
    self.assertEqual(src, "KINETIC_STATE_BACKEND")

  def test_profile_with_none_state_backend_falls_through(self):
    profile = mock.Mock(state_backend=None)
    with self._clear_env():
      settings.set_("state_backend", "gcs")
      url, src = state_backend.resolve_state_backend_for_show(
        profile, "my-proj"
      )
    self.assertEqual(url, "gs://my-proj-kinetic-state")
    self.assertEqual(src, "settings")


class EnsureGcsBackendTest(absltest.TestCase):
  """ensure_gcs_backend is best-effort: it only raises on the one error
  the user must fix (Conflict — bucket name globally taken). Everything
  else is swallowed so collaborators with only object-level permissions
  reach Pulumi, which surfaces a clean object-level error if access is
  actually wrong."""

  def _patch_client(self, bucket_mock):
    client_mock = mock.MagicMock()
    client_mock.bucket.return_value = bucket_mock
    return (
      mock.patch.object(
        state_backend.storage, "Client", return_value=client_mock
      ),
      client_mock,
    )

  def test_creates_with_versioning_and_ubla(self):
    bucket = mock.MagicMock()
    patcher, client_mock = self._patch_client(bucket)
    with patcher:
      state_backend.ensure_gcs_backend("gs://my-bucket", project="my-proj")
    self.assertTrue(bucket.versioning_enabled)
    self.assertTrue(
      bucket.iam_configuration.uniform_bucket_level_access_enabled
    )
    client_mock.create_bucket.assert_called_once()

  def test_storage_client_pinned_to_project(self):
    bucket = mock.MagicMock()
    with mock.patch.object(state_backend.storage, "Client") as client_cls:
      client_cls.return_value.bucket.return_value = bucket
      state_backend.ensure_gcs_backend("gs://my-bucket", project="kinetic-proj")
    client_cls.assert_called_once_with(project="kinetic-proj")

  def test_storage_client_unpinned_when_no_project(self):
    bucket = mock.MagicMock()
    with mock.patch.object(state_backend.storage, "Client") as client_cls:
      client_cls.return_value.bucket.return_value = bucket
      state_backend.ensure_gcs_backend("gs://my-bucket")
    client_cls.assert_called_once_with()

  def test_existing_bucket_swallowed_for_collaborators(self):
    """Conflict on create means the bucket already exists. We must NOT
    raise — the second admin (with only objectAdmin) needs to proceed."""
    bucket = mock.MagicMock()
    patcher, client_mock = self._patch_client(bucket)
    client_mock.create_bucket.side_effect = gax.Conflict("exists")
    with patcher:
      state_backend.ensure_gcs_backend("gs://my-bucket", project="p")

  def test_forbidden_on_create_swallowed_for_collaborators(self):
    """Object-only collaborators lack storage.admin; they should still
    pass through to Pulumi, which works at the object level."""
    bucket = mock.MagicMock()
    patcher, client_mock = self._patch_client(bucket)
    client_mock.create_bucket.side_effect = gax.Forbidden("nope")
    with patcher:
      state_backend.ensure_gcs_backend("gs://my-bucket", project="p")

  def test_permission_denied_on_create_swallowed(self):
    bucket = mock.MagicMock()
    patcher, client_mock = self._patch_client(bucket)
    client_mock.create_bucket.side_effect = gax.PermissionDenied("nope")
    with patcher:
      state_backend.ensure_gcs_backend("gs://my-bucket", project="p")

  def test_extracts_bucket_name_from_url_with_prefix(self):
    bucket = mock.MagicMock()
    patcher, client_mock = self._patch_client(bucket)
    with patcher:
      state_backend.ensure_gcs_backend(
        "gs://my-bucket/some/prefix", project="p"
      )
    client_mock.bucket.assert_called_once_with("my-bucket")


class ParseGcsUrlTest(absltest.TestCase):
  def test_bucket_only(self):
    self.assertEqual(
      state_backend._parse_gcs_url("gs://my-bucket"), ("my-bucket", "")
    )

  def test_bucket_with_prefix(self):
    self.assertEqual(
      state_backend._parse_gcs_url("gs://my-bucket/foo/bar"),
      ("my-bucket", "foo/bar"),
    )

  def test_invalid_raises(self):
    with self.assertRaises(click.BadParameter):
      state_backend._parse_gcs_url("not a url")


if __name__ == "__main__":
  absltest.main()
