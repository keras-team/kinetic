"""Tests for kinetic.cli.commands.accelerators."""

from unittest import mock

from absl.testing import absltest
from click.testing import CliRunner

from kinetic.cli.commands.accelerators import accelerators
from kinetic.cli.config import NodePoolConfig
from kinetic.cli.infra.state import StackState
from kinetic.core.accelerators import GpuConfig, TpuConfig

_SENTINEL = object()


def _make_state(node_pools=None, stack=_SENTINEL):
  if stack is _SENTINEL:
    stack = mock.MagicMock()
  return StackState(
    project="test-project",
    zone="us-central1-a",
    cluster_name="kinetic-cluster",
    node_pools=node_pools or [],
    stack=stack,
  )


_LIVE_ARGS = [
  "--project",
  "test-project",
  "--zone",
  "us-central1-a",
  "--live",
]


class AcceleratorsDefaultTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.runner = CliRunner()

  def test_lists_gpus_and_tpus(self):
    result = self.runner.invoke(accelerators, [])

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("GPUs", result.output)
    self.assertIn("TPUs", result.output)
    self.assertIn("h100", result.output)
    self.assertIn("a100", result.output)
    self.assertIn("l4", result.output)
    self.assertIn("v6e", result.output)
    self.assertIn("v5litepod", result.output)

  def test_no_status_column_by_default(self):
    result = self.runner.invoke(accelerators, [])

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertNotIn("Status", result.output)
    self.assertNotIn("provisioned", result.output)

  def test_does_not_call_load_state(self):
    with mock.patch("kinetic.cli.infra.state.load_state") as mock_load:
      result = self.runner.invoke(accelerators, [])

    self.assertEqual(result.exit_code, 0, result.output)
    mock_load.assert_not_called()


class AcceleratorsLiveTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.mock_load = self.enterContext(
      mock.patch("kinetic.cli.infra.state.load_state")
    )

  def test_live_shows_provisioned_gpu(self):
    pool = NodePoolConfig(
      "gpu-l4-abcd",
      GpuConfig("l4", 1, "nvidia-l4", "g2-standard-4"),
    )
    self.mock_load.return_value = _make_state(node_pools=[pool])

    result = self.runner.invoke(accelerators, _LIVE_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("provisioned", result.output)
    self.assertIn("Status", result.output)

  def test_live_shows_provisioned_tpu(self):
    pool = NodePoolConfig(
      "tpu-v5p-1234",
      TpuConfig("v5p", 8, "2x2x2", "tpu-v5p-slice", "ct5p-hightpu-4t", 2),
    )
    self.mock_load.return_value = _make_state(node_pools=[pool])

    result = self.runner.invoke(accelerators, _LIVE_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("provisioned", result.output)

  def test_live_no_cluster_still_lists(self):
    self.mock_load.return_value = _make_state(stack=None)

    result = self.runner.invoke(accelerators, _LIVE_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("GPUs", result.output)
    self.assertIn("TPUs", result.output)
    self.assertIn("No provisioned accelerators found", result.output)

  def test_live_load_failure_still_lists(self):
    self.mock_load.side_effect = RuntimeError("boom")

    result = self.runner.invoke(accelerators, _LIVE_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("GPUs", result.output)
    self.assertIn("No provisioned accelerators found", result.output)


if __name__ == "__main__":
  absltest.main()
