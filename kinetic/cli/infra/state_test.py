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


if __name__ == "__main__":
  absltest.main()
