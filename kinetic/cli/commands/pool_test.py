"""Tests for kinetic.cli.commands.pool — add, remove, list."""

from unittest import mock

import click
from absl.testing import absltest
from click.testing import CliRunner

from kinetic.cli.commands.pool import pool
from kinetic.cli.config import NodePoolConfig
from kinetic.cli.infra.state import StackState
from kinetic.core.accelerators import GpuConfig, TpuConfig

_SENTINEL = object()


def _make_state(node_pools=None, stack=_SENTINEL):
  """Create a StackState for testing."""
  if stack is _SENTINEL:
    stack = mock.MagicMock()
  return StackState(
    project="test-project",
    zone="us-central1-a",
    cluster_name="kinetic-cluster",
    node_pools=node_pools or [],
    stack=stack,
  )


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
    self.mock_load = self.enterContext(
      mock.patch("kinetic.cli.commands.pool.load_state")
    )
    self.mock_apply = self.enterContext(
      mock.patch("kinetic.cli.commands.pool.apply_update", return_value=True)
    )
    self.mock_gen = self.enterContext(
      mock.patch(
        "kinetic.cli.commands.pool.generate_pool_name",
        return_value="gpu-l4-abcd",
      )
    )
    self.mock_load.return_value = _make_state()

  def test_add_gpu_pool(self):
    result = self.runner.invoke(pool, _ADD_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Pool Added", result.output)
    self.mock_apply.assert_called_once()
    config = self.mock_apply.call_args[0][0]
    self.assertLen(config.node_pools, 1)
    self.assertIsNone(config.node_pools[0].reservation)

  def test_add_to_existing_pools(self):
    existing = NodePoolConfig(
      "tpu-v5p-1234",
      TpuConfig("v5p", 8, "2x2x2", "tpu-v5p-slice", "ct5p-hightpu-4t", 2),
    )
    self.mock_load.return_value = _make_state(node_pools=[existing])

    result = self.runner.invoke(pool, _ADD_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Total pools after add: 2", result.output)
    config = self.mock_apply.call_args[0][0]
    self.assertLen(config.node_pools, 2)

  def test_add_with_reservation(self):
    args = _ADD_ARGS + ["--reservation", "my-v6e-reservation"]
    result = self.runner.invoke(pool, args)

    self.assertEqual(result.exit_code, 0, result.output)
    config = self.mock_apply.call_args[0][0]
    self.assertEqual(config.node_pools[0].reservation, "my-v6e-reservation")

  def test_add_reservation_with_spot_rejected(self):
    args = _ADD_ARGS + ["--reservation", "my-reservation", "--spot"]
    result = self.runner.invoke(pool, args)

    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("Reservations cannot be used with Spot VMs", result.output)

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
    self.mock_load = self.enterContext(
      mock.patch("kinetic.cli.commands.pool.load_state")
    )
    self.mock_apply = self.enterContext(
      mock.patch("kinetic.cli.commands.pool.apply_update", return_value=True)
    )

  def test_remove_existing_pool(self):
    existing = NodePoolConfig(
      "gpu-l4-abcd",
      GpuConfig("l4", 1, "nvidia-l4", "g2-standard-4"),
    )
    self.mock_load.return_value = _make_state(node_pools=[existing])

    result = self.runner.invoke(pool, _REMOVE_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Pool Removed", result.output)
    self.assertIn("Remaining pools after remove: 0", result.output)
    config = self.mock_apply.call_args[0][0]
    self.assertEmpty(config.node_pools)

  def test_remove_nonexistent_pool_fails(self):
    self.mock_load.return_value = _make_state()

    result = self.runner.invoke(pool, _REMOVE_ARGS)

    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("not found", result.output)


class PoolListTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.mock_load = self.enterContext(
      mock.patch("kinetic.cli.commands.pool.load_state")
    )
    self.mock_infrastructure_state = self.enterContext(
      mock.patch("kinetic.cli.commands.pool.infrastructure_state")
    )

  def test_list_shows_infrastructure_state(self):
    mock_stack = mock.MagicMock()
    mock_stack.outputs.return_value = {
      "cluster_name": mock.MagicMock(value="my-cluster"),
    }
    self.mock_load.return_value = _make_state(stack=mock_stack)

    result = self.runner.invoke(pool, _LIST_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.mock_infrastructure_state.assert_called_once()

  def test_list_no_outputs_shows_warning(self):
    mock_stack = mock.MagicMock()
    mock_stack.outputs.return_value = {}
    self.mock_load.return_value = _make_state(stack=mock_stack)

    result = self.runner.invoke(pool, _LIST_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("No infrastructure found", result.output)
    self.mock_infrastructure_state.assert_not_called()

  def test_list_no_stack_shows_warning(self):
    self.mock_load.return_value = _make_state(stack=None)

    result = self.runner.invoke(pool, _LIST_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("No Pulumi stack found", result.output)


class PoolAddUpdateFailureTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.enterContext(
      mock.patch(
        "kinetic.cli.commands.pool.load_state",
        return_value=_make_state(),
      )
    )
    self.enterContext(
      mock.patch("kinetic.cli.commands.pool.apply_update", return_value=False)
    )
    self.enterContext(
      mock.patch(
        "kinetic.cli.commands.pool.generate_pool_name",
        return_value="gpu-l4-abcd",
      )
    )

  def test_add_update_failure_warns(self):
    result = self.runner.invoke(pool, _ADD_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Pool Update Failed", result.output)
    self.assertNotIn("Pool Added", result.output)


class PoolRemoveUpdateFailureTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    existing = NodePoolConfig(
      "gpu-l4-abcd",
      GpuConfig("l4", 1, "nvidia-l4", "g2-standard-4"),
    )
    self.enterContext(
      mock.patch(
        "kinetic.cli.commands.pool.load_state",
        return_value=_make_state(node_pools=[existing]),
      )
    )
    self.enterContext(
      mock.patch("kinetic.cli.commands.pool.apply_update", return_value=False)
    )

  def test_remove_update_failure_warns(self):
    result = self.runner.invoke(pool, _REMOVE_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Pool Update Failed", result.output)
    self.assertNotIn("Pool Removed", result.output)


class PoolAddNoStackTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.enterContext(
      mock.patch(
        "kinetic.cli.commands.pool.load_state",
        side_effect=click.ClickException("No Pulumi stack found"),
      )
    )
    self.enterContext(
      mock.patch(
        "kinetic.cli.commands.pool.generate_pool_name",
        return_value="gpu-l4-abcd",
      )
    )

  def test_add_no_stack_shows_friendly_error(self):
    result = self.runner.invoke(pool, _ADD_ARGS)

    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("No Pulumi stack found", result.output)


class PoolRemoveNoStackTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.enterContext(
      mock.patch(
        "kinetic.cli.commands.pool.load_state",
        side_effect=click.ClickException("No Pulumi stack found"),
      )
    )

  def test_remove_no_stack_shows_friendly_error(self):
    result = self.runner.invoke(pool, _REMOVE_ARGS)

    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("No Pulumi stack found", result.output)


if __name__ == "__main__":
  absltest.main()
