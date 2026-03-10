"""Tests for keras_remote.cli.infra.program — node pool and exports."""

from unittest import mock

from absl.testing import absltest, parameterized

from keras_remote.cli.config import NodePoolConfig
from keras_remote.cli.constants import (
  MAX_CLUSTER_CPU,
  MAX_CLUSTER_MEMORY_GB,
  NODE_MAX_RUN_DURATION_SECONDS,
)
from keras_remote.core.accelerators import GpuConfig, TpuConfig

# Patch the pulumi_gcp module before importing program, so the module-level
# import inside program.py picks up the mock.
with mock.patch.dict("sys.modules", {"pulumi_gcp": mock.MagicMock()}):
  from keras_remote.cli.infra import program


class TestCreateTpuNodePool(parameterized.TestCase):
  """Verify _create_tpu_node_pool sets placement_policy correctly."""

  @parameterized.named_parameters(
    dict(
      testcase_name="v5p_multi_host",
      tpu=TpuConfig("v5p", 8, "2x2x2", "tpu-v5p-slice", "ct5p-hightpu-4t", 2),
      expect_placement=True,
    ),
    dict(
      testcase_name="v6e_multi_host",
      tpu=TpuConfig("v6e", 8, "2x4", "tpu-v6e-slice", "ct6e-standard-4t", 2),
      expect_placement=True,
    ),
    dict(
      testcase_name="v3_single_host",
      tpu=TpuConfig("v3", 4, "2x2", "tpu-v3-podslice", "ct3-hightpu-4t", 1),
      expect_placement=False,
    ),
    dict(
      testcase_name="v5litepod_single_host",
      tpu=TpuConfig(
        "v5litepod", 4, "2x2", "tpu-v5-lite-podslice", "ct5lp-hightpu-4t", 1
      ),
      expect_placement=False,
    ),
  )
  @mock.patch.object(program, "gcp")
  def test_placement_policy(self, gcp_mock, tpu, expect_placement):
    cluster = mock.MagicMock()
    cluster.name = "test-cluster"

    program._create_tpu_node_pool(
      cluster, tpu, "us-central2-b", "my-project", f"tpu-{tpu.name}-abcd"
    )

    call_kwargs = gcp_mock.container.NodePool.call_args
    placement = call_kwargs.kwargs.get(
      "placement_policy", call_kwargs[1].get("placement_policy")
    )

    if expect_placement:
      self.assertIsNotNone(placement)
      gcp_mock.container.NodePoolPlacementPolicyArgs.assert_called_once_with(
        type="COMPACT",
        tpu_topology=tpu.topology,
      )
    else:
      self.assertIsNone(placement)

  @mock.patch.object(program, "gcp")
  def test_node_count_matches_config(self, gcp_mock):
    tpu = TpuConfig("v5p", 16, "2x2x4", "tpu-v5p-slice", "ct5p-hightpu-4t", 4)
    cluster = mock.MagicMock()
    cluster.name = "test-cluster"

    program._create_tpu_node_pool(
      cluster, tpu, "us-central2-b", "my-project", "tpu-v5p-abcd"
    )

    # Due to multi-host TPU workaround, initial_node_count is equal to num_nodes
    call_kwargs = gcp_mock.container.NodePool.call_args.kwargs
    self.assertEqual(call_kwargs.get("initial_node_count"), 0)
    autoscaling_kwargs = (
      gcp_mock.container.NodePoolAutoscalingArgs.call_args.kwargs
    )
    self.assertEqual(autoscaling_kwargs.get("max_node_count"), 4)
    self.assertEqual(autoscaling_kwargs.get("min_node_count"), 0)

  @mock.patch.object(program, "gcp")
  def test_pool_name_passed_through(self, gcp_mock):
    tpu = TpuConfig("v5p", 8, "2x2x2", "tpu-v5p-slice", "ct5p-hightpu-4t", 2)
    cluster = mock.MagicMock()
    cluster.name = "test-cluster"

    program._create_tpu_node_pool(
      cluster, tpu, "us-central2-b", "my-project", "tpu-v5p-f1a2"
    )

    positional_args = gcp_mock.container.NodePool.call_args[0]
    self.assertEqual(positional_args[0], "tpu-v5p-f1a2")


class TestCreateGpuNodePool(absltest.TestCase):
  @mock.patch.object(program, "gcp")
  def test_pool_name_passed_through(self, gcp_mock):
    gpu = GpuConfig("l4", 1, "nvidia-l4", "g2-standard-4")
    cluster = mock.MagicMock()
    cluster.name = "test-cluster"

    program._create_gpu_node_pool(
      cluster, gpu, "us-central1-a", "my-project", "gpu-l4-a3f2"
    )

    positional_args = gcp_mock.container.NodePool.call_args[0]
    self.assertEqual(positional_args[0], "gpu-l4-a3f2")


def _make_config(node_pools=None):
  """Create a mock InfraConfig for testing."""
  config = mock.MagicMock()
  config.project = "test-project"
  config.zone = "us-central1-a"
  config.cluster_name = "test-cluster"
  config.node_pools = node_pools or []
  return config


def _run_program_and_get_exports(config):
  """Run the Pulumi program and return a dict of exported key -> value."""
  with (
    mock.patch.object(program, "pulumi") as pulumi_mock,
    mock.patch.object(program, "gcp") as gcp_mock,
  ):
    # Make NodePool(...) return a mock whose .name.apply(fn) resolves with
    # the name= kwarg, matching real Pulumi behaviour.
    def _make_node_pool(*args, **kwargs):
      pool_mock = mock.MagicMock()
      pool_name = kwargs.get("name", args[0] if args else None)
      pool_mock.name.apply.side_effect = lambda fn: fn(pool_name)
      return pool_mock

    gcp_mock.container.NodePool.side_effect = _make_node_pool
    gcp_mock.artifactregistry.Repository.return_value.name.apply.side_effect = (
      lambda fn: fn(None)
    )
    # Make Output.all() simply collect resolved values into a list.
    pulumi_mock.Output.all.side_effect = lambda *args: list(args)

    program_fn = program.create_program(config)
    program_fn()
    return {
      call.args[0]: call.args[1] for call in pulumi_mock.export.call_args_list
    }


class TestAcceleratorExports(absltest.TestCase):
  """Verify accelerator metadata is exported correctly."""

  def test_gpu_exports(self):
    gpu = GpuConfig("l4", 1, "nvidia-l4", "g2-standard-4")
    node_pools = [NodePoolConfig("gpu-l4-a3f2", gpu)]
    exports = _run_program_and_get_exports(_make_config(node_pools))

    self.assertIn("accelerators", exports)
    accel_list = exports["accelerators"]
    self.assertLen(accel_list, 1)
    accel = accel_list[0]
    self.assertEqual(accel["type"], "GPU")
    self.assertEqual(accel["name"], "l4")
    self.assertEqual(accel["count"], 1)
    self.assertEqual(accel["machine_type"], "g2-standard-4")
    self.assertEqual(accel["node_pool"], "gpu-l4-a3f2")
    self.assertEqual(accel["node_count"], 1)

  def test_tpu_exports(self):
    tpu = TpuConfig("v5p", 8, "2x2x2", "tpu-v5p-slice", "ct5p-hightpu-4t", 2)
    node_pools = [NodePoolConfig("tpu-v5p-b7e1", tpu)]
    exports = _run_program_and_get_exports(_make_config(node_pools))

    self.assertIn("accelerators", exports)
    accel_list = exports["accelerators"]
    self.assertLen(accel_list, 1)
    accel = accel_list[0]
    self.assertEqual(accel["type"], "TPU")
    self.assertEqual(accel["name"], "v5p")
    self.assertEqual(accel["chips"], 8)
    self.assertEqual(accel["topology"], "2x2x2")
    self.assertEqual(accel["machine_type"], "ct5p-hightpu-4t")
    self.assertEqual(accel["node_pool"], "tpu-v5p-b7e1")
    self.assertEqual(accel["node_count"], 2)

  def test_cpu_only_exports_empty_list(self):
    exports = _run_program_and_get_exports(_make_config([]))

    self.assertIn("accelerators", exports)
    self.assertEqual(exports["accelerators"], [])

  def test_multiple_pools(self):
    gpu = GpuConfig("l4", 1, "nvidia-l4", "g2-standard-4")
    tpu = TpuConfig("v5p", 8, "2x2x2", "tpu-v5p-slice", "ct5p-hightpu-4t", 2)
    node_pools = [
      NodePoolConfig("gpu-l4-a3f2", gpu),
      NodePoolConfig("tpu-v5p-b7e1", tpu),
    ]
    exports = _run_program_and_get_exports(_make_config(node_pools))

    accel_list = exports["accelerators"]
    self.assertLen(accel_list, 2)
    self.assertEqual(accel_list[0]["type"], "GPU")
    self.assertEqual(accel_list[0]["node_pool"], "gpu-l4-a3f2")
    self.assertEqual(accel_list[1]["type"], "TPU")
    self.assertEqual(accel_list[1]["node_pool"], "tpu-v5p-b7e1")


class TestExportsDependOnResources(parameterized.TestCase):
  """Verify exports are derived from resource outputs, not static values.

  If an export were a plain dict/string instead of a resource output
  .apply(), it would resolve even when the resource fails to create,
  causing 'keras-remote status' to show stale accelerator info.
  """

  @parameterized.named_parameters(
    dict(
      testcase_name="gpu",
      node_pools=[
        NodePoolConfig(
          "gpu-l4-a3f2",
          GpuConfig("l4", 1, "nvidia-l4", "g2-standard-4"),
        )
      ],
    ),
    dict(
      testcase_name="tpu",
      node_pools=[
        NodePoolConfig(
          "tpu-v5p-b7e1",
          TpuConfig("v5p", 8, "2x2x2", "tpu-v5p-slice", "ct5p-hightpu-4t", 2),
        )
      ],
    ),
  )
  def test_accelerator_export_depends_on_node_pool(self, node_pools):
    """The accelerator export must be produced via NodePool.name.apply()."""
    with (
      mock.patch.object(program, "pulumi") as pulumi_mock,
      mock.patch.object(program, "gcp") as gcp_mock,
    ):
      program.create_program(_make_config(node_pools))()

      # The export value should use Output.all() which depends on
      # pool.name.apply() calls.
      node_pool_mock = gcp_mock.container.NodePool.return_value
      node_pool_mock.name.apply.assert_called_once()
      pulumi_mock.Output.all.assert_called_once()

  def test_ar_registry_export_depends_on_repository(self):
    """The ar_registry export must be produced via Repository.name.apply()."""
    with (
      mock.patch.object(program, "pulumi") as pulumi_mock,
      mock.patch.object(program, "gcp") as gcp_mock,
    ):
      program.create_program(_make_config([]))()

      repo_mock = gcp_mock.artifactregistry.Repository.return_value
      repo_mock.name.apply.assert_called_once()
      exported = {
        c.args[0]: c.args[1] for c in pulumi_mock.export.call_args_list
      }
      self.assertIs(exported["ar_registry"], repo_mock.name.apply.return_value)

  def test_cpu_only_accelerator_export_is_empty_list(self):
    """CPU-only config should export accelerators as empty list."""
    with (
      mock.patch.object(program, "pulumi") as pulumi_mock,
      mock.patch.object(program, "gcp") as gcp_mock,
    ):
      program.create_program(_make_config([]))()

      # No node pool created, so NodePool should not be called.
      gcp_mock.container.NodePool.assert_not_called()
      exported = {
        c.args[0]: c.args[1] for c in pulumi_mock.export.call_args_list
      }
      self.assertEqual(exported["accelerators"], [])


class TestClusterAutoscalingAndNAP(absltest.TestCase):
  """Verify cluster autoscaling and NAP are enabled correctly."""

  def test_cluster_autoscaling_config(self):
    """The cluster should have OPTIMIZE_UTILIZATION and NAP enabled."""
    with (
      mock.patch.object(program, "pulumi"),
      mock.patch.object(program, "gcp") as gcp_mock,
    ):
      program.create_program(_make_config(None))()

      gcp_mock.container.ClusterClusterAutoscalingArgs.assert_called_once()
      call_args = (
        gcp_mock.container.ClusterClusterAutoscalingArgs.call_args.kwargs
      )
      self.assertTrue(call_args.get("enabled"))
      self.assertEqual(
        call_args.get("autoscaling_profile"), "OPTIMIZE_UTILIZATION"
      )

      gcp_mock.container.ClusterClusterAutoscalingAutoProvisioningDefaultsArgs.assert_called_once()
      gcp_mock.container.ClusterClusterAutoscalingAutoProvisioningDefaultsManagementArgs.assert_called_once_with(
        auto_upgrade=True, auto_repair=True
      )

      gcp_mock.container.ClusterClusterAutoscalingResourceLimitArgs.assert_any_call(
        resource_type="cpu", maximum=MAX_CLUSTER_CPU
      )
      gcp_mock.container.ClusterClusterAutoscalingResourceLimitArgs.assert_any_call(
        resource_type="memory", maximum=MAX_CLUSTER_MEMORY_GB
      )


class TestScaleToZeroNodePools(parameterized.TestCase):
  """Verify accelerator node pools can scale to zero and have maxRunDuration."""

  @parameterized.named_parameters(
    dict(
      testcase_name="gpu",
      accelerator=GpuConfig("l4", 1, "nvidia-l4", "g2-standard-4"),
      expected_max_count=10,
    ),
    dict(
      testcase_name="tpu_v5p",
      accelerator=TpuConfig(
        "v5p", 16, "2x2x4", "tpu-v5p-slice", "ct5p-hightpu-4t", 4
      ),
      expected_max_count=4,  # 16 chips / 4 per VM
    ),
  )
  @mock.patch.object(program, "gcp")
  def test_node_pool_scale_to_zero(
    self, gcp_mock, accelerator, expected_max_count
  ):
    cluster = mock.MagicMock()
    cluster.name = "test-cluster"

    if isinstance(accelerator, GpuConfig):
      program._create_gpu_node_pool(
        cluster, accelerator, "us-central2-b", "my-project", "test-pool"
      )
    else:
      program._create_tpu_node_pool(
        cluster, accelerator, "us-central2-b", "my-project", "test-pool"
      )

    call_kwargs = gcp_mock.container.NodePool.call_args.kwargs
    self.assertEqual(call_kwargs.get("initial_node_count"), 0)

    autoscaling_kwargs = (
      gcp_mock.container.NodePoolAutoscalingArgs.call_args.kwargs
    )
    self.assertEqual(autoscaling_kwargs.get("min_node_count"), 0)
    self.assertEqual(
      autoscaling_kwargs.get("max_node_count"), expected_max_count
    )

    mgmt_kwargs = gcp_mock.container.NodePoolManagementArgs.call_args.kwargs
    self.assertTrue(mgmt_kwargs.get("auto_repair"))
    self.assertTrue(mgmt_kwargs.get("auto_upgrade"))

    node_config_kwargs = (
      gcp_mock.container.NodePoolNodeConfigArgs.call_args.kwargs
    )
    self.assertEqual(
      node_config_kwargs.get("max_run_duration"),
      f"{NODE_MAX_RUN_DURATION_SECONDS}s",
    )


if __name__ == "__main__":
  absltest.main()
