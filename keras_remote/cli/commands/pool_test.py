"""Tests for keras_remote.cli.commands.pool — add, remove, list."""

from unittest import mock

from absl.testing import absltest
from click.testing import CliRunner
from pulumi.automation import errors as pulumi_errors

from keras_remote.cli.commands.pool import pool
from keras_remote.cli.config import NodePoolConfig
from keras_remote.core.accelerators import GpuConfig, TpuConfig

# Patches applied to every test to bypass prerequisites and infrastructure.
_BASE_PATCHES = {
  "check_all": mock.patch("keras_remote.cli.commands.pool.check_all"),
  "create_program": mock.patch("keras_remote.cli.commands.pool.create_program"),
  "get_stack": mock.patch("keras_remote.cli.commands.pool.get_stack"),
  "get_current_node_pools": mock.patch(
    "keras_remote.cli.commands.pool.get_current_node_pools"
  ),
  "generate_pool_name": mock.patch(
    "keras_remote.cli.commands.pool.generate_pool_name",
    return_value="gpu-l4-abcd",
  ),
}


def _start_patches(test_case):
  mocks = {}
  for name, patcher in _BASE_PATCHES.items():
    mocks[name] = test_case.enterContext(patcher)
  return mocks


_ADD_ARGS = [
  "add",
  "--project",
  "test-project",
  "--zone",
  "us-central1-a",
  "--accelerator",
  "l4",
  "--yes",
]

_REMOVE_ARGS = [
  "remove",
  "--project",
  "test-project",
  "--zone",
  "us-central1-a",
  "gpu-l4-abcd",
  "--yes",
]

_LIST_ARGS = [
  "list",
  "--project",
  "test-project",
  "--zone",
  "us-central1-a",
]


class PoolAddTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.mocks = _start_patches(self)
    mock_stack = mock.MagicMock()
    mock_stack.up.return_value.summary.resource_changes = {"create": 1}
    self.mocks["get_stack"].return_value = mock_stack
    self.mocks["get_current_node_pools"].return_value = []

  def test_add_gpu_pool(self):
    result = self.runner.invoke(pool, _ADD_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Pool Added", result.output)

  def test_add_to_existing_pools(self):
    existing = NodePoolConfig(
      "tpu-v5p-1234",
      TpuConfig("v5p", 8, "2x2x2", "tpu-v5p-slice", "ct5p-hightpu-4t", 2),
    )
    self.mocks["get_current_node_pools"].return_value = [existing]

    result = self.runner.invoke(pool, _ADD_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Total pools after add: 2", result.output)

  def test_add_cpu_rejected(self):
    result = self.runner.invoke(
      pool,
      [
        "add",
        "--project",
        "test-project",
        "--accelerator",
        "cpu",
        "--yes",
      ],
    )

    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("Cannot add a CPU node pool", result.output)


class PoolRemoveTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.mocks = _start_patches(self)
    mock_stack = mock.MagicMock()
    mock_stack.up.return_value.summary.resource_changes = {"delete": 1}
    self.mocks["get_stack"].return_value = mock_stack

  def test_remove_existing_pool(self):
    existing = NodePoolConfig(
      "gpu-l4-abcd",
      GpuConfig("l4", 1, "nvidia-l4", "g2-standard-4"),
    )
    self.mocks["get_current_node_pools"].return_value = [existing]

    result = self.runner.invoke(pool, _REMOVE_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Pool Removed", result.output)
    self.assertIn("Remaining pools after remove: 0", result.output)

  def test_remove_nonexistent_pool_fails(self):
    self.mocks["get_current_node_pools"].return_value = []

    result = self.runner.invoke(pool, _REMOVE_ARGS)

    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("not found", result.output)


class PoolListTest(absltest.TestCase):
  """Tests for the pool list happy path."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.mocks = _start_patches(self)
    self.mock_stack = mock.MagicMock()
    self.mocks["get_stack"].return_value = self.mock_stack
    self.mock_infrastructure_state = self.enterContext(
      mock.patch("keras_remote.cli.commands.pool.infrastructure_state")
    )

  def test_list_shows_infrastructure_state(self):
    """Stack exists with outputs — displays infrastructure state."""
    self.mock_stack.outputs.return_value = {
      "cluster_name": mock.MagicMock(value="my-cluster"),
    }

    result = self.runner.invoke(pool, _LIST_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.mock_infrastructure_state.assert_called_once()

  def test_list_no_outputs_shows_warning(self):
    """Stack exists but outputs are empty — warns user."""
    self.mock_stack.outputs.return_value = {}

    result = self.runner.invoke(pool, _LIST_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("No infrastructure found", result.output)
    self.mock_infrastructure_state.assert_not_called()


class PoolAddRefreshFailureTest(absltest.TestCase):
  """Tests that pool add handles stack.refresh() failure gracefully."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.mocks = _start_patches(self)
    mock_stack = mock.MagicMock()
    mock_stack.refresh.side_effect = pulumi_errors.CommandError(
      "refresh failed"
    )
    mock_stack.up.return_value.summary.resource_changes = {"create": 1}
    self.mocks["get_stack"].return_value = mock_stack
    self.mocks["get_current_node_pools"].return_value = []

  def test_add_refresh_failure_still_proceeds(self):
    """stack.refresh() failure is warned but add continues since it may be the first run."""
    result = self.runner.invoke(pool, _ADD_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Failed to refresh stack state", result.output)
    self.assertIn("Pool Added", result.output)


class PoolAddUpdateFailureTest(absltest.TestCase):
  """Tests that pool add handles stack.up() failure gracefully."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.mocks = _start_patches(self)
    mock_stack = mock.MagicMock()
    mock_stack.up.side_effect = pulumi_errors.CommandError("update failed")
    self.mocks["get_stack"].return_value = mock_stack
    self.mocks["get_current_node_pools"].return_value = []

  def test_add_update_failure_warns(self):
    """stack.up() failure is caught and warned, not a crash."""
    result = self.runner.invoke(pool, _ADD_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Pulumi update encountered an issue", result.output)


class PoolRemoveUpdateFailureTest(absltest.TestCase):
  """Tests that pool remove handles stack.up() failure gracefully."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.mocks = _start_patches(self)
    mock_stack = mock.MagicMock()
    mock_stack.up.side_effect = pulumi_errors.CommandError("update failed")
    self.mocks["get_stack"].return_value = mock_stack
    existing = NodePoolConfig(
      "gpu-l4-abcd",
      GpuConfig("l4", 1, "nvidia-l4", "g2-standard-4"),
    )
    self.mocks["get_current_node_pools"].return_value = [existing]

  def test_remove_update_failure_warns(self):
    """stack.up() failure is caught and warned, not a crash."""
    result = self.runner.invoke(pool, _REMOVE_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Pulumi update encountered an issue", result.output)


class PoolAddNoStackTest(absltest.TestCase):
  """Tests that `pool add` fails gracefully when no stack exists."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.mocks = _start_patches(self)
    self.mocks["get_stack"].side_effect = pulumi_errors.CommandError(
      "stack not found"
    )

  def test_add_no_stack_shows_friendly_error(self):
    result = self.runner.invoke(pool, _ADD_ARGS)

    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("No Pulumi stack found", result.output)
    self.assertIn("keras-remote up", result.output)


class PoolRemoveNoStackTest(absltest.TestCase):
  """Tests that `pool remove` fails gracefully when no stack exists."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.mocks = _start_patches(self)
    self.mocks["get_stack"].side_effect = pulumi_errors.CommandError(
      "stack not found"
    )

  def test_remove_no_stack_shows_friendly_error(self):
    result = self.runner.invoke(pool, _REMOVE_ARGS)

    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("No Pulumi stack found", result.output)
    self.assertIn("keras-remote up", result.output)


class PoolListNoStackTest(absltest.TestCase):
  """Tests that `pool list` handles missing stack gracefully."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.mocks = _start_patches(self)
    self.mocks["get_stack"].side_effect = pulumi_errors.CommandError(
      "stack not found"
    )

  def test_list_no_stack_shows_warning(self):
    result = self.runner.invoke(pool, _LIST_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("No Pulumi stack found", result.output)
    self.assertIn("keras-remote up", result.output)


if __name__ == "__main__":
  absltest.main()
