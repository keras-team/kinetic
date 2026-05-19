"""Tests for kinetic.cli.infra.state — centralized state loading."""

from unittest import mock

import click
from absl.testing import absltest
from pulumi.automation import errors as pulumi_errors

from kinetic.cli.config import NodePoolConfig
from kinetic.cli.infra import state
from kinetic.core.accelerators import GpuConfig


def _make_mock_stack(pools=None):
  """Create a mock stack with configured outputs."""
  mock_stack = mock.MagicMock()
  mock_stack.up.return_value.summary.resource_changes = {"create": 1}
  return mock_stack


class LoadStateTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.enterContext(mock.patch.object(state, "check_all"))
    self.enterContext(
      mock.patch.object(state, "resolve_project", return_value="test-proj")
    )
    self.mock_create = self.enterContext(
      mock.patch.object(state, "create_program")
    )
    self.mock_get_stack = self.enterContext(
      mock.patch.object(state, "get_stack")
    )
    self.mock_get_pools = self.enterContext(
      mock.patch.object(state, "get_current_node_pools", return_value=[])
    )
    self.mock_get_force_destroy = self.enterContext(
      mock.patch.object(state, "get_current_force_destroy", return_value=True)
    )
    self.mock_stack = _make_mock_stack()
    self.mock_get_stack.return_value = self.mock_stack

  def test_returns_full_state(self):
    pool = NodePoolConfig(
      "gpu-l4-1234", GpuConfig("l4", 1, "nvidia-l4", "g2-standard-4")
    )
    self.mock_get_pools.return_value = [pool]

    result = state.load_state("proj", "us-central1-a", "cluster")

    self.assertEqual(result.project, "proj")
    self.assertEqual(result.zone, "us-central1-a")
    self.assertEqual(result.cluster_name, "cluster")
    self.assertLen(result.node_pools, 1)
    self.assertEqual(result.node_pools[0].name, "gpu-l4-1234")
    self.assertIsNotNone(result.stack)

  def test_resolves_defaults(self):
    result = state.load_state(None, None, None)

    self.assertEqual(result.project, "test-proj")
    self.assertEqual(result.zone, "us-central1-a")
    self.assertEqual(result.cluster_name, "kinetic-cluster")

  def test_missing_stack_raises_by_default(self):
    self.mock_get_stack.side_effect = pulumi_errors.CommandError("not found")

    with self.assertRaises(click.exceptions.ClickException) as cm:
      state.load_state("proj", "us-central1-a", "cluster")
    self.assertIn("No Pulumi stack found", str(cm.exception))

  def test_missing_stack_allowed(self):
    self.mock_get_stack.side_effect = pulumi_errors.CommandError("not found")

    result = state.load_state(
      "proj", "us-central1-a", "cluster", allow_missing=True
    )

    self.assertEqual(result.project, "proj")
    self.assertEmpty(result.node_pools)
    self.assertIsNone(result.stack)

  def test_refresh_failure_proceeds(self):
    self.mock_stack.refresh.side_effect = pulumi_errors.CommandError(
      "refresh failed"
    )

    result = state.load_state("proj", "us-central1-a", "cluster")

    self.assertIsNotNone(result.stack)
    self.mock_get_pools.assert_called_once()

  def test_loads_force_destroy_from_stack(self):
    self.mock_get_force_destroy.return_value = False

    result = state.load_state("proj", "us-central1-a", "cluster")

    self.assertFalse(result.force_destroy)

  def test_defaults_force_destroy_true_when_stack_missing(self):
    self.mock_get_stack.side_effect = pulumi_errors.CommandError("not found")

    result = state.load_state(
      "proj", "us-central1-a", "cluster", allow_missing=True
    )

    self.assertTrue(result.force_destroy)

  def test_skips_prerequisites_when_requested(self):
    mock_check = self.enterContext(mock.patch.object(state, "check_all"))

    state.load_state("proj", "zone", "cluster", check_prerequisites=False)

    mock_check.assert_not_called()


class FormatChangesTest(absltest.TestCase):
  def test_mixed_changes(self):
    result = state._format_changes({"create": 3, "update": 1, "same": 5})
    self.assertEqual(result, "3 to create, 1 to update, 5 unchanged")

  def test_no_changes(self):
    result = state._format_changes({"same": 4})
    self.assertEqual(result, "4 unchanged")

  def test_empty(self):
    result = state._format_changes({})
    self.assertEqual(result, "no changes")

  def test_deletes(self):
    result = state._format_changes({"delete": 2, "same": 1})
    self.assertEqual(result, "2 to delete, 1 unchanged")


class ApplyUpdateTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.mock_create = self.enterContext(
      mock.patch.object(state, "create_program")
    )
    self.mock_get_stack = self.enterContext(
      mock.patch.object(state, "get_stack")
    )
    self.mock_stack = _make_mock_stack()
    self.mock_get_stack.return_value = self.mock_stack

  def test_success_returns_true(self):
    config = mock.MagicMock()

    result = state.apply_update(config)

    self.assertTrue(result)
    self.mock_stack.up.assert_called_once()

  def test_failure_returns_false(self):
    self.mock_stack.up.side_effect = pulumi_errors.CommandError("failed")
    config = mock.MagicMock()

    result = state.apply_update(config)

    self.assertFalse(result)

  def test_passes_config_to_create_program(self):
    config = mock.MagicMock()

    state.apply_update(config)

    self.mock_create.assert_called_once_with(config)


class ApplyPreviewTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.mock_create = self.enterContext(
      mock.patch.object(state, "create_program")
    )
    self.mock_get_stack = self.enterContext(
      mock.patch.object(state, "get_stack")
    )
    self.mock_stack = mock.MagicMock()
    self.mock_stack.preview.return_value.change_summary = {"create": 1}
    self.mock_get_stack.return_value = self.mock_stack

  def test_success_returns_true(self):
    config = mock.MagicMock()

    result = state.apply_preview(config)

    self.assertTrue(result)
    self.mock_stack.preview.assert_called_once()

  def test_failure_returns_false(self):
    self.mock_stack.preview.side_effect = pulumi_errors.CommandError("failed")
    config = mock.MagicMock()

    result = state.apply_preview(config)

    self.assertFalse(result)

  def test_passes_config_to_create_program(self):
    config = mock.MagicMock()

    state.apply_preview(config)

    self.mock_create.assert_called_once_with(config)


class ApplyDestroyTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.mock_create = self.enterContext(
      mock.patch.object(state, "create_program")
    )
    self.mock_get_stack = self.enterContext(
      mock.patch.object(state, "get_stack")
    )
    self.mock_stack = mock.MagicMock()
    self.mock_stack.destroy.return_value.summary.resource_changes = {
      "delete": 1
    }
    self.mock_get_stack.return_value = self.mock_stack

  def test_success_returns_true(self):
    config = mock.MagicMock()

    result = state.apply_destroy(config)

    self.assertTrue(result)
    self.mock_stack.destroy.assert_called_once()

  def test_failure_returns_false(self):
    self.mock_stack.destroy.side_effect = pulumi_errors.CommandError("failed")
    config = mock.MagicMock()

    result = state.apply_destroy(config)

    self.assertFalse(result)

  def test_passes_config_to_create_program(self):
    config = mock.MagicMock()

    state.apply_destroy(config)

    self.mock_create.assert_called_once_with(config)

  def test_removes_stack_after_successful_destroy(self):
    """Regression for Codex P2: stack file must be removed from the backend
    so list_clusters() doesn't keep offering destroyed clusters as join targets.
    """
    config = mock.MagicMock()
    result = state.apply_destroy(config)
    self.assertTrue(result)
    self.mock_stack.workspace.remove_stack.assert_called_once_with(
      self.mock_stack.name
    )

  def test_does_not_remove_stack_when_destroy_fails(self):
    self.mock_stack.destroy.side_effect = pulumi_errors.CommandError("failed")
    config = mock.MagicMock()
    result = state.apply_destroy(config)
    self.assertFalse(result)
    self.mock_stack.workspace.remove_stack.assert_not_called()

  def test_remove_stack_failure_does_not_fail_destroy(self):
    """If destroy succeeded, a backend-side cleanup glitch shouldn't surface
    as a failure — the user's resources are gone, that's what matters.
    """
    self.mock_stack.workspace.remove_stack.side_effect = (
      pulumi_errors.CommandError("backend glitch")
    )
    config = mock.MagicMock()
    result = state.apply_destroy(config)
    self.assertTrue(result)


class ListClustersTest(absltest.TestCase):
  """Regression for Codex P2: list_clusters must work for collaborators
  with only object-level (not bucket-level) IAM on the state bucket.
  """

  def setUp(self):
    super().setUp()
    self.mock_client_cls = self.enterContext(
      mock.patch("kinetic.cli.infra.state.storage.Client")
    )
    self.mock_client = self.mock_client_cls.return_value

  def _set_blob_names(self, names):
    blobs = []
    for n in names:
      # MagicMock's `name` constructor kwarg is special, so set the
      # attribute after construction.
      blob = mock.MagicMock()
      blob.name = n
      blobs.append(blob)
    self.mock_client.list_blobs.return_value = blobs

  def test_does_not_call_bucket_exists(self):
    """The bucket.exists() precheck required storage.buckets.get IAM that
    object-only collaborators don't have. Confirm we skip it entirely.
    """
    self._set_blob_names(
      [
        ".pulumi/stacks/kinetic/my-proj-dev-tpu.json",
        ".pulumi/stacks/kinetic/my-proj-team-x.json",
      ]
    )
    clusters = state.list_clusters("my-proj")
    self.assertEqual(clusters, ["dev-tpu", "team-x"])
    self.mock_client.bucket.assert_not_called()

  def test_uses_list_blobs_with_kinetic_prefix(self):
    self._set_blob_names([])
    state.list_clusters("my-proj")
    self.mock_client.list_blobs.assert_called_once_with(
      "my-proj-kinetic-state", prefix=".pulumi/stacks/kinetic/"
    )

  def test_returns_empty_when_list_blobs_raises(self):
    """Missing bucket, Forbidden, etc. all collapse to []."""
    self.mock_client.list_blobs.side_effect = Exception("any cloud error")
    self.assertEqual(state.list_clusters("my-proj"), [])

  def test_skips_pulumi_backup_files(self):
    self._set_blob_names(
      [
        ".pulumi/stacks/kinetic/my-proj-good.json",
        ".pulumi/stacks/kinetic/my-proj-stale.bak.json",
      ]
    )
    self.assertEqual(state.list_clusters("my-proj"), ["good"])

  def test_filters_by_project_prefix(self):
    """A bucket may contain stacks from other projects; list only ours."""
    self._set_blob_names(
      [
        ".pulumi/stacks/kinetic/my-proj-mine.json",
        ".pulumi/stacks/kinetic/other-proj-theirs.json",
      ]
    )
    self.assertEqual(state.list_clusters("my-proj"), ["mine"])


if __name__ == "__main__":
  absltest.main()
