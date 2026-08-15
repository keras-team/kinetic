"""Tests for kinetic.cli.infra.state_backend.

Bucket lifecycle runs against the real fake-gcs-server emulator: the
derived bucket name is asserted by its actual existence, versioning by
reading it back, and Conflict by a genuine duplicate create. The one
thing the emulator does not persist is uniform bucket-level access, so
that flag is asserted on the request itself via a spy over the real
``create_bucket`` call.
"""

import uuid
from unittest import mock

from absl.testing import absltest
from google.api_core import exceptions as gax
from google.cloud import storage

from kinetic.cli.infra import state_backend
from kinetic.utils.fake_gcs_fixture import FakeGcsTestCase


class StateBackendUrlTest(absltest.TestCase):
  def test_derives_from_project(self):
    self.assertEqual(
      state_backend.state_backend_url("my-proj"),
      "gs://my-proj-kinetic-state",
    )


class EnsureGcsBackendTest(FakeGcsTestCase):
  """ensure_gcs_backend is best-effort. It tries to create the bucket once;
  Conflict / Forbidden / PermissionDenied are silently swallowed so that
  collaborators with only object-level perms reach Pulumi, which surfaces
  a clean object-level error if access is actually wrong."""

  def _project(self):
    """A unique project name so each test gets its own state bucket."""
    return f"proj-{uuid.uuid4().hex[:12]}"

  def test_creates_the_derived_bucket_with_versioning(self):
    project = self._project()
    expected_bucket = f"{project}-kinetic-state"

    state_backend.ensure_gcs_backend(project)

    bucket = storage.Client(project=self.PROJECT).get_bucket(expected_bucket)
    self.assertEqual(bucket.name, expected_bucket)
    self.assertTrue(bucket.versioning_enabled)

  def test_requests_uniform_bucket_level_access(self):
    # The emulator does not persist UBLA, so assert it on the request.
    # The real create_bucket reloads the bucket from the server's reply
    # (dropping the flag), so snapshot the requested config at call time
    # and then let the real call proceed.
    project = self._project()
    requested = {}
    real_create = storage.Client.create_bucket

    def spy(client, bucket, *args, **kwargs):
      requested["ubla"] = (
        bucket.iam_configuration.uniform_bucket_level_access_enabled
      )
      requested["versioning"] = bucket.versioning_enabled
      return real_create(client, bucket, *args, **kwargs)

    with mock.patch.object(storage.Client, "create_bucket", spy):
      state_backend.ensure_gcs_backend(project)

    self.assertEqual(requested, {"ubla": True, "versioning": True})

  def test_storage_client_pinned_to_project(self):
    project = self._project()
    with mock.patch.object(
      storage, "Client", wraps=storage.Client
    ) as client_cls:
      state_backend.ensure_gcs_backend(project)
    client_cls.assert_called_once_with(project=project)

  def test_existing_bucket_is_left_alone(self):
    """A real second create raises Conflict, which is swallowed."""
    project = self._project()
    state_backend.ensure_gcs_backend(project)

    state_backend.ensure_gcs_backend(project)  # no exception

    self.assertTrue(
      storage.Client(project=self.PROJECT)
      .bucket(f"{project}-kinetic-state")
      .exists()
    )

  def test_forbidden_swallowed_for_collaborators(self):
    # Fault injection: the emulator has no IAM, so a Forbidden create
    # can only be simulated.
    with mock.patch.object(
      storage.Client, "create_bucket", side_effect=gax.Forbidden("nope")
    ):
      state_backend.ensure_gcs_backend(self._project())  # no exception

  def test_permission_denied_swallowed(self):
    with mock.patch.object(
      storage.Client,
      "create_bucket",
      side_effect=gax.PermissionDenied("nope"),
    ):
      state_backend.ensure_gcs_backend(self._project())  # no exception


if __name__ == "__main__":
  absltest.main()
