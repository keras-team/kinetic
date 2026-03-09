"""Tests for keras_remote.cli.commands.up — resilience to partial failures."""

import subprocess
from unittest import mock

from absl.testing import absltest
from click.testing import CliRunner

from keras_remote.cli.commands.up import up
from keras_remote.cli.config import NodePoolConfig
from keras_remote.cli.infra.state import StackState
from keras_remote.core.accelerators import GpuConfig, TpuConfig


def _make_state(node_pools=None):
  """Create a StackState for testing."""
  return StackState(
    project="test-project",
    zone="us-central2-b",
    cluster_name="keras-remote-cluster",
    node_pools=node_pools or [],
    stack=mock.MagicMock(),
  )


# Shared CLI args that skip interactive prompts.
_CLI_ARGS = [
  "--project",
  "test-project",
  "--zone",
  "us-central2-b",
  "--accelerator",
  "cpu",
  "--yes",
]

# Patches applied to every test to bypass prerequisites and infrastructure.
_BASE_PATCHES = {
  "check_all": mock.patch("keras_remote.cli.commands.up.check_all"),
  "load_state": mock.patch("keras_remote.cli.commands.up.load_state"),
  "apply_update": mock.patch(
    "keras_remote.cli.commands.up.apply_update", return_value=True
  ),
  "configure_kubectl": mock.patch(
    "keras_remote.cli.commands.up.configure_kubectl",
  ),
  "install_lws": mock.patch("keras_remote.cli.commands.up.install_lws"),
  "install_gpu_drivers": mock.patch(
    "keras_remote.cli.commands.up.install_gpu_drivers",
  ),
}


def _start_patches(test_case):
  """Start all base patches and return a dict of mock objects."""
  mocks = {}
  for name, patcher in _BASE_PATCHES.items():
    mocks[name] = test_case.enterContext(patcher)
  # Default: empty state (first run)
  mocks["load_state"].return_value = _make_state()
  return mocks


class UpCommandResilienceTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.mocks = _start_patches(self)

  def test_full_success(self):
    """All steps succeed — exit code 0, 'Setup Complete' shown."""
    result = self.runner.invoke(up, _CLI_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Setup Complete", result.output)
    self.assertNotIn("Warnings", result.output)
    self.mocks["install_lws"].assert_called_once()
    self.mocks["configure_kubectl"].assert_called_once()
    self.mocks[
      "install_gpu_drivers"
    ].assert_not_called()  # CPU accelerator, no GPU drivers.

  def test_pulumi_failure_still_runs_post_deploy(self):
    """apply_update returns False — post-deploy steps still execute."""
    self.mocks["apply_update"].return_value = False

    result = self.runner.invoke(up, _CLI_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.mocks["configure_kubectl"].assert_called_once()
    self.mocks["install_lws"].assert_called_once()
    self.assertIn("Setup Completed With Warnings", result.output)
    self.assertIn("Pulumi provisioning encountered errors", result.output)

  def test_multiple_post_deploy_failures(self):
    """Multiple post-deploy failures are all reported."""
    self.mocks["configure_kubectl"].side_effect = subprocess.CalledProcessError(
      1, "gcloud"
    )
    self.mocks["install_lws"].side_effect = subprocess.CalledProcessError(
      1, "kubectl"
    )

    result = self.runner.invoke(up, _CLI_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("kubectl configuration", result.output)
    self.assertIn("LWS CRD installation", result.output)

  def test_pulumi_and_post_deploy_failures_combined(self):
    """Both Pulumi and post-deploy failures are reported together."""
    self.mocks["apply_update"].return_value = False
    self.mocks["install_lws"].side_effect = subprocess.CalledProcessError(
      1, "kubectl"
    )

    result = self.runner.invoke(up, _CLI_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Pulumi provisioning encountered errors", result.output)
    self.assertIn("LWS CRD installation", result.output)
    self.assertIn("re-run", result.output)


class UpCommandGpuDriverTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.mocks = _start_patches(self)

  def test_gpu_driver_failure_reported(self):
    """GPU driver installation failure is caught and reported."""
    self.mocks[
      "install_gpu_drivers"
    ].side_effect = subprocess.CalledProcessError(1, "kubectl")
    args = [
      "--project",
      "test-project",
      "--zone",
      "us-central1-a",
      "--accelerator",
      "t4",
      "--yes",
    ]

    result = self.runner.invoke(up, args)

    self.assertEqual(result.exit_code, 0, result.output)
    self.mocks["install_gpu_drivers"].assert_called_once()
    self.assertIn("GPU driver installation", result.output)


class UpCommandPoolPreservationTest(absltest.TestCase):
  """Tests that `up` preserves existing node pools and defers to pool commands."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.mocks = _start_patches(self)

  def test_preserves_existing_pools_ignores_accelerator_flag(self):
    """Re-running `up` with --accelerator keeps only existing pools."""
    existing = NodePoolConfig(
      "gpu-a100-1234",
      GpuConfig("a100", 1, "nvidia-tesla-a100", "a2-highgpu-1g"),
    )
    self.mocks["load_state"].return_value = _make_state(node_pools=[existing])

    args = [
      "--project",
      "test-project",
      "--zone",
      "us-central1-a",
      "--accelerator",
      "t4",
      "--yes",
    ]
    result = self.runner.invoke(up, args)

    self.assertEqual(result.exit_code, 0, result.output)
    config = self.mocks["apply_update"].call_args[0][0]
    self.assertLen(config.node_pools, 1)
    self.assertEqual(config.node_pools[0].name, "gpu-a100-1234")
    self.assertIn("pool add/remove", result.output)

  def test_cpu_rerun_preserves_existing_pools(self):
    """Re-running `up --accelerator cpu` preserves all existing pools."""
    existing = [
      NodePoolConfig(
        "gpu-t4-abcd",
        GpuConfig("t4", 1, "nvidia-tesla-t4", "n1-standard-4"),
      ),
      NodePoolConfig(
        "tpu-v5p-1234",
        TpuConfig("v5p", 8, "2x2x2", "tpu-v5p-slice", "ct5p-hightpu-4t", 2),
      ),
    ]
    self.mocks["load_state"].return_value = _make_state(node_pools=existing)

    args = [
      "--project",
      "test-project",
      "--zone",
      "us-central1-a",
      "--accelerator",
      "cpu",
      "--yes",
    ]
    result = self.runner.invoke(up, args)

    self.assertEqual(result.exit_code, 0, result.output)
    config = self.mocks["apply_update"].call_args[0][0]
    self.assertLen(config.node_pools, 2)
    self.assertIn("pool add/remove", result.output)

  def test_first_run_creates_pool_from_flag(self):
    """First run with --accelerator creates the requested pool."""
    self.mocks["load_state"].return_value = _make_state()

    args = [
      "--project",
      "test-project",
      "--zone",
      "us-central1-a",
      "--accelerator",
      "t4",
      "--yes",
    ]
    result = self.runner.invoke(up, args)

    self.assertEqual(result.exit_code, 0, result.output)
    config = self.mocks["apply_update"].call_args[0][0]
    self.assertLen(config.node_pools, 1)

  def test_first_run_no_existing_stack(self):
    """First run — load_state returns empty state, proceeds normally."""
    self.mocks["load_state"].return_value = _make_state()

    args = [
      "--project",
      "test-project",
      "--zone",
      "us-central1-a",
      "--accelerator",
      "t4",
      "--yes",
    ]
    result = self.runner.invoke(up, args)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Setup Complete", result.output)


if __name__ == "__main__":
  absltest.main()
