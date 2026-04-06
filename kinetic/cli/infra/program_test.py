"""Tests for kinetic.cli.infra.program — node pool and K8s resources."""

from unittest import mock

from absl.testing import absltest, parameterized

from kinetic.cli.config import NodePoolConfig
from kinetic.core.accelerators import GpuConfig, TpuConfig

# Patch pulumi provider modules before importing program, so the module-level
# imports inside program.py pick up the mocks.
with mock.patch.dict(
  "sys.modules",
  {
    "pulumi_command": mock.MagicMock(),
    "pulumi_gcp": mock.MagicMock(),
    "pulumi_kubernetes": mock.MagicMock(),
  },
):
  from kinetic.cli.infra import program


class TestCreateTpuNodePool(parameterized.TestCase):
  """Verify _create_tpu_node_pool sets placement_policy correctly.

  Multi-host TPUs require COMPACT placement with an explicit topology;
  single-host slices must NOT have placement_policy or GKE rejects
  the node pool.
  """

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
      cluster,
      tpu,
      "us-central2-b",
      "my-project",
      f"tpu-{tpu.name}-abcd",
      "sa@test.iam.gserviceaccount.com",
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


def _make_config(node_pools=None):
  """Create a mock InfraConfig for testing."""
  config = mock.MagicMock()
  config.project = "test-project"
  config.zone = "us-central1-a"
  config.cluster_name = "test-cluster"
  config.node_pools = node_pools or []
  return config


class TestGpuDriverConditional(absltest.TestCase):
  """GPU driver DaemonSet must only be installed when GPU pools are present."""

  def _run_program(self, config=None):
    config = config or _make_config()
    with (
      mock.patch.object(program, "pulumi"),
      mock.patch.object(program, "command"),
      mock.patch.object(program, "gcp"),
      mock.patch.object(program, "k8s") as k8s_mock,
    ):
      program.create_program(config)()
    return k8s_mock

  def test_installed_when_gpu_pools_present(self):
    gpu = GpuConfig("l4", 1, "nvidia-l4", "g2-standard-4")
    config = _make_config([NodePoolConfig("gpu-l4-a3f2", gpu)])
    k8s_mock = self._run_program(config)

    gpu_calls = [
      c
      for c in k8s_mock.yaml.ConfigFile.call_args_list
      if c.args[0] == "nvidia-gpu-drivers"
    ]
    self.assertLen(gpu_calls, 1)

  def test_not_installed_for_cpu_only(self):
    k8s_mock = self._run_program(_make_config([]))

    gpu_calls = [
      c
      for c in k8s_mock.yaml.ConfigFile.call_args_list
      if c.args[0] == "nvidia-gpu-drivers"
    ]
    self.assertEmpty(gpu_calls)

  def test_not_installed_for_tpu_only(self):
    tpu = TpuConfig("v5p", 8, "2x2x2", "tpu-v5p-slice", "ct5p-hightpu-4t", 2)
    config = _make_config([NodePoolConfig("tpu-v5p-b7e1", tpu)])
    k8s_mock = self._run_program(config)

    gpu_calls = [
      c
      for c in k8s_mock.yaml.ConfigFile.call_args_list
      if c.args[0] == "nvidia-gpu-drivers"
    ]
    self.assertEmpty(gpu_calls)


if __name__ == "__main__":
  absltest.main()
