"""Tests for kinetic.backend.k8s_utils — shared K8s utilities."""

from unittest import mock
from unittest.mock import MagicMock

from absl.testing import absltest, parameterized
from google.cloud import container_v1
from kubernetes.client.rest import ApiException
from kubernetes.config import ConfigException

from kinetic.backend.k8s_utils import (
  _check_node_pool_exists_cached,
  _pod_exit_summary,
  check_pod_scheduling,
  collect_pod_failure_details,
  load_kube_config,
  parse_accelerator,
)


class TestParseAccelerator(absltest.TestCase):
  def test_cpu(self):
    result = parse_accelerator("cpu")
    self.assertEqual(result["node_selector"], {})
    self.assertEqual(result["resource_limits"], {})
    self.assertEqual(result["resource_requests"], {})
    self.assertEqual(result["tolerations"], [])
    self.assertEqual(result["jax_platform"], "cpu")

  def test_gpu_l4(self):
    result = parse_accelerator("l4")
    self.assertEqual(
      result["node_selector"],
      {"cloud.google.com/gke-accelerator": "nvidia-l4"},
    )
    self.assertEqual(result["resource_limits"], {"nvidia.com/gpu": "1"})
    self.assertEqual(result["resource_requests"], {"nvidia.com/gpu": "1"})
    self.assertEqual(result["jax_platform"], "gpu")
    self.assertLen(result["tolerations"], 1)
    self.assertEqual(result["tolerations"][0]["key"], "nvidia.com/gpu")
    self.assertEqual(result["tolerations"][0]["operator"], "Exists")
    self.assertEqual(result["tolerations"][0]["effect"], "NoSchedule")

  def test_gpu_a100x4(self):
    result = parse_accelerator("a100x4")
    self.assertEqual(result["resource_limits"], {"nvidia.com/gpu": "4"})
    self.assertEqual(result["resource_requests"], {"nvidia.com/gpu": "4"})

  def test_tpu_v3_8(self):
    result = parse_accelerator("v3-4")
    self.assertIn(
      "cloud.google.com/gke-tpu-accelerator", result["node_selector"]
    )
    self.assertIn("cloud.google.com/gke-tpu-topology", result["node_selector"])
    self.assertEqual(result["resource_limits"], {"google.com/tpu": "4"})
    self.assertEqual(result["resource_requests"], {"google.com/tpu": "4"})
    self.assertEqual(result["jax_platform"], "tpu")
    self.assertLen(result["tolerations"], 1)
    self.assertEqual(result["tolerations"][0]["key"], "google.com/tpu")

  def test_tpu_v3_16_multi_node(self):
    # v3-16 has 4 nodes and 16 total chips -> 4 chips per node
    result = parse_accelerator("v3-16")
    self.assertEqual(result["resource_limits"], {"google.com/tpu": "4"})
    self.assertEqual(result["resource_requests"], {"google.com/tpu": "4"})
    self.assertEqual(
      result["node_selector"]["cloud.google.com/gke-tpu-topology"], "4x4"
    )

  def test_tpu_v5litepod_4(self):
    result = parse_accelerator("v5litepod-4")
    self.assertEqual(
      result["node_selector"],
      {
        "cloud.google.com/gke-tpu-accelerator": "tpu-v5-lite-podslice",
        "cloud.google.com/gke-tpu-topology": "2x2",
      },
    )
    self.assertEqual(result["resource_limits"], {"google.com/tpu": "4"})

  def test_spot_gpu(self):
    result = parse_accelerator("l4:spot")
    self.assertEqual(
      result["node_selector"]["cloud.google.com/gke-spot"], "true"
    )
    # Check for spot toleration
    spot_tol = [
      t
      for t in result["tolerations"]
      if t.get("key") == "cloud.google.com/gke-spot"
    ]
    self.assertLen(spot_tol, 1)
    self.assertEqual(spot_tol[0]["value"], "true")

  def test_spot_tpu(self):
    result = parse_accelerator("v6e-8:spot")
    self.assertEqual(
      result["node_selector"]["cloud.google.com/gke-spot"], "true"
    )
    # Check for spot toleration
    spot_tol = [
      t
      for t in result["tolerations"]
      if t.get("key") == "cloud.google.com/gke-spot"
    ]
    self.assertLen(spot_tol, 1)
    self.assertEqual(spot_tol[0]["value"], "true")
    # Should still have TPU toleration
    self.assertTrue(
      any(t.get("key") == "google.com/tpu" for t in result["tolerations"])
    )


class TestLoadKubeConfig(absltest.TestCase):
  def setUp(self):
    super().setUp()
    load_kube_config.cache_clear()
    self.addCleanup(load_kube_config.cache_clear)

  def test_kubeconfig_fallback(self):
    """Falls back to local kubeconfig when in-cluster config is unavailable."""
    with (
      mock.patch(
        "kinetic.backend.k8s_utils.config.load_incluster_config",
        side_effect=ConfigException("not in cluster"),
      ),
      mock.patch("kinetic.backend.k8s_utils.config.load_kube_config"),
    ):
      load_kube_config()


class TestCheckNodePoolExistsCached(absltest.TestCase):
  def setUp(self):
    super().setUp()
    _check_node_pool_exists_cached.cache_clear()

    self.mock_get_cluster_info = self.enterContext(
      mock.patch(
        "kinetic.backend.k8s_utils._get_cluster_info",
        return_value=("test-project", "us-central1-c", "test-cluster"),
      )
    )
    self.mock_gke_client = MagicMock()
    self.enterContext(
      mock.patch(
        "kinetic.backend.k8s_utils.container_v1.ClusterManagerClient",
        return_value=self.mock_gke_client,
      )
    )
    self.mock_warning = self.enterContext(
      mock.patch("kinetic.backend.k8s_utils.logging.warning")
    )

  def _make_pool(
    self,
    accelerator_type="",
    labels=None,
    spot=False,
    machine_type="",
    tpu_topology="",
    resource_labels=None,
  ):
    pool = container_v1.NodePool(
      config=container_v1.NodeConfig(
        accelerators=(
          [container_v1.AcceleratorConfig(accelerator_type=accelerator_type)]
          if accelerator_type
          else []
        ),
        labels=labels or {},
        spot=spot,
        machine_type=machine_type,
        resource_labels=resource_labels or {},
      ),
      placement_policy=container_v1.NodePool.PlacementPolicy(
        tpu_topology=tpu_topology,
      ),
    )
    return pool

  def _set_pools(self, pools):
    self.mock_gke_client.list_node_pools.return_value = (
      container_v1.ListNodePoolsResponse(node_pools=pools)
    )

  def test_gpu_match(self):
    self._set_pools(
      [
        self._make_pool(
          accelerator_type="nvidia-l4",
          labels={"existing-label": "true"},
        )
      ]
    )

    result = _check_node_pool_exists_cached(
      (("cloud.google.com/gke-accelerator", "nvidia-l4"),)
    )
    self.assertTrue(result)

  def test_tpu_match(self):
    self._set_pools(
      [
        self._make_pool(
          accelerator_type="tpu-v5-lite-podslice",
          machine_type="ct5lp-hightpu-4t",
        )
      ]
    )

    result = _check_node_pool_exists_cached(
      (
        ("cloud.google.com/gke-tpu-accelerator", "tpu-v5-lite-podslice"),
        ("cloud.google.com/gke-tpu-topology", "2x2"),
      )
    )
    self.assertTrue(result)

  def test_tpu_multi_node_match(self):
    """Test that it correctly identifies a 4-chip-per-node pool for v6e-16."""
    self._set_pools(
      [
        self._make_pool(
          accelerator_type="tpu-v6e-slice",
          machine_type="ct6e-standard-4t",
        )
      ]
    )

    result = _check_node_pool_exists_cached(
      (
        ("cloud.google.com/gke-tpu-accelerator", "tpu-v6e-slice"),
        ("cloud.google.com/gke-tpu-topology", "4x4"),
        ("cloud.google.com/gke-accelerator-count", "4"),
      )
    )
    self.assertTrue(result)

  def test_no_match(self):
    self._set_pools([self._make_pool(accelerator_type="nvidia-t4")])

    result = _check_node_pool_exists_cached(
      (("cloud.google.com/gke-accelerator", "nvidia-l4"),)
    )
    self.assertFalse(result)

  def test_graceful_degradation_on_api_error(self):
    self.mock_gke_client.list_node_pools.side_effect = Exception("API error")

    result = _check_node_pool_exists_cached(
      (("cloud.google.com/gke-accelerator", "nvidia-l4"),)
    )
    self.assertTrue(result)
    self.mock_warning.assert_called_once()
    self.assertIn(
      "Could not verify node pool existence", self.mock_warning.call_args[0][0]
    )

  def test_no_cluster_info_degrades_gracefully(self):
    self.mock_get_cluster_info.return_value = None

    result = _check_node_pool_exists_cached(
      (("cloud.google.com/gke-accelerator", "nvidia-l4"),)
    )
    self.assertTrue(result)
    self.mock_warning.assert_called_once()
    self.assertIn(
      "Could not determine GKE cluster", self.mock_warning.call_args[0][0]
    )

  def test_uses_correct_parent_path(self):
    self._set_pools([])

    _check_node_pool_exists_cached(
      (("cloud.google.com/gke-accelerator", "nvidia-l4"),)
    )
    self.mock_gke_client.list_node_pools.assert_called_once_with(
      parent="projects/test-project/locations/us-central1-c/clusters/test-cluster"
    )


class TestPodExitSummary(absltest.TestCase):
  def _make_pod(self, exit_code=None, reason=None, message=None):
    pod = MagicMock()
    cs = MagicMock()
    cs.name = "kinetic-worker"
    terminated = MagicMock()
    terminated.exit_code = exit_code
    terminated.reason = reason
    terminated.message = message
    cs.state.terminated = terminated
    cs.last_state.terminated = None
    pod.status.container_statuses = [cs]
    pod.status.init_container_statuses = None
    return pod

  def test_exit_code_and_reason(self):
    pod = self._make_pod(exit_code=137, reason="OOMKilled")
    result = _pod_exit_summary(pod)
    self.assertIn("137", result)
    self.assertIn("OOMKilled", result)

  def test_exit_code_only(self):
    pod = self._make_pod(exit_code=1, reason=None)
    result = _pod_exit_summary(pod)
    self.assertEqual(result, "exit code 1")

  def test_no_container_statuses(self):
    pod = MagicMock()
    pod.status.container_statuses = None
    pod.status.init_container_statuses = None
    self.assertIsNone(_pod_exit_summary(pod))

  def test_no_terminated_state(self):
    pod = MagicMock()
    cs = MagicMock()
    cs.name = "kinetic-worker"
    cs.state.terminated = None
    cs.last_state.terminated = None
    pod.status.container_statuses = [cs]
    pod.status.init_container_statuses = None
    self.assertIsNone(_pod_exit_summary(pod))

  def test_falls_back_to_last_state(self):
    pod = MagicMock()
    cs = MagicMock()
    cs.name = "kinetic-worker"
    cs.state.terminated = None
    terminated = MagicMock()
    terminated.exit_code = 2
    terminated.reason = "Error"
    terminated.message = None
    cs.last_state.terminated = terminated
    pod.status.container_statuses = [cs]
    pod.status.init_container_statuses = None
    result = _pod_exit_summary(pod)
    self.assertIn("exit code 2", result)
    self.assertIn("Error", result)

  def test_prioritizes_kinetic_worker(self):
    pod = MagicMock()
    sidecar = MagicMock()
    sidecar.name = "logging-sidecar"
    sidecar.state.terminated = MagicMock(exit_code=0, reason=None, message=None)
    sidecar.last_state.terminated = None
    worker = MagicMock()
    worker.name = "kinetic-worker"
    worker.state.terminated = MagicMock(
      exit_code=137, reason="OOMKilled", message=None
    )
    worker.last_state.terminated = None
    pod.status.container_statuses = [sidecar, worker]
    pod.status.init_container_statuses = None
    result = _pod_exit_summary(pod)
    self.assertIn("137", result)
    self.assertIn("OOMKilled", result)

  def test_init_container_failure(self):
    pod = MagicMock()
    pod.status.container_statuses = []
    init_cs = MagicMock()
    init_cs.name = "install-deps"
    init_cs.state.terminated = MagicMock(
      exit_code=1, reason="Error", message=None
    )
    init_cs.last_state.terminated = None
    pod.status.init_container_statuses = [init_cs]
    result = _pod_exit_summary(pod)
    self.assertIn("exit code 1", result)
    self.assertIn("Error", result)


class TestCollectPodFailureDetails(absltest.TestCase):
  def test_includes_exit_info_and_logs(self):
    mock_core = MagicMock()
    pod = MagicMock()
    pod.metadata.name = "test-pod-abc"

    terminated = MagicMock()
    terminated.exit_code = 1
    terminated.reason = "Error"
    terminated.message = None
    cs = MagicMock()
    cs.name = "kinetic-worker"
    cs.state.terminated = terminated
    cs.last_state.terminated = None
    pod.status.container_statuses = [cs]
    pod.status.init_container_statuses = None

    mock_core.list_namespaced_pod.return_value.items = [pod]
    mock_core.read_namespaced_pod_log.return_value = (
      "Traceback (most recent call last):\n"
      '  File "run.py", line 1\n'
      "ImportError: no module named foo\n"
    )

    result = collect_pod_failure_details(mock_core, "job-1", "default")
    self.assertIn("exit code 1", result)
    self.assertIn("Error", result)
    self.assertIn("ImportError", result)

  def test_empty_when_no_pods(self):
    mock_core = MagicMock()
    mock_core.list_namespaced_pod.return_value.items = []

    result = collect_pod_failure_details(mock_core, "job-1", "default")
    self.assertEqual(result, "")

  def test_logs_only_when_no_terminated_state(self):
    mock_core = MagicMock()
    pod = MagicMock()
    pod.metadata.name = "test-pod"
    pod.status.phase = "Failed"
    cs = MagicMock()
    cs.name = "kinetic-worker"
    cs.state.terminated = None
    cs.last_state.terminated = None
    pod.status.container_statuses = [cs]
    pod.status.init_container_statuses = None
    mock_core.list_namespaced_pod.return_value.items = [pod]
    mock_core.read_namespaced_pod_log.return_value = "some output\n"

    result = collect_pod_failure_details(mock_core, "job-1", "default")
    self.assertIn("some output", result)
    self.assertNotIn("exit code", result)

  def test_skips_non_failed_pods(self):
    mock_core = MagicMock()
    running_pod = MagicMock()
    running_pod.metadata.name = "running-pod"
    running_pod.status.phase = "Running"
    cs = MagicMock()
    cs.name = "kinetic-worker"
    cs.state.terminated = None
    cs.last_state.terminated = None
    running_pod.status.container_statuses = [cs]
    running_pod.status.init_container_statuses = None

    mock_core.list_namespaced_pod.return_value.items = [running_pod]

    result = collect_pod_failure_details(mock_core, "job-1", "default")
    self.assertEqual(result, "")
    mock_core.read_namespaced_pod_log.assert_not_called()

  def test_caps_at_five_pods(self):
    mock_core = MagicMock()
    pods = []
    for i in range(7):
      pod = MagicMock()
      pod.metadata.name = f"pod-{i}"
      pod.status.phase = "Failed"
      cs = MagicMock()
      cs.name = "kinetic-worker"
      cs.state.terminated = MagicMock(exit_code=1, reason="Error", message=None)
      cs.last_state.terminated = None
      pod.status.container_statuses = [cs]
      pod.status.init_container_statuses = None
      pods.append(pod)

    mock_core.list_namespaced_pod.return_value.items = pods
    mock_core.read_namespaced_pod_log.return_value = "error\n"

    result = collect_pod_failure_details(mock_core, "job-1", "default")
    self.assertIn("... (additional failed pods omitted)", result)
    # Only 5 pods should have logs fetched.
    self.assertEqual(mock_core.read_namespaced_pod_log.call_count, 5)

  def test_api_exception_on_list_pods(self):
    mock_core = MagicMock()
    mock_core.list_namespaced_pod.side_effect = ApiException(
      status=500, reason="Server Error"
    )

    result = collect_pod_failure_details(mock_core, "job-1", "default")
    self.assertEqual(result, "")


class TestCheckPodScheduling(parameterized.TestCase):
  def _make_pending_pod(self, message, node_selector=None):
    pod = MagicMock()
    pod.status.phase = "Pending"
    pod.spec.node_selector = node_selector
    condition = MagicMock()
    condition.type = "PodScheduled"
    condition.status = "False"
    condition.message = message
    pod.status.conditions = [condition]
    return pod

  @parameterized.named_parameters(
    dict(
      testcase_name="insufficient_gpu",
      condition_message="Insufficient nvidia.com/gpu",
      log_match="Insufficient nvidia.com/gpu",
      node_selector=None,
    ),
    dict(
      testcase_name="node_selector_mismatch",
      condition_message="didn't match Pod's node affinity/selector",
      log_match="Selector: cloud.google.com/gke-accelerator: nvidia-l4",
      node_selector={"cloud.google.com/gke-accelerator": "nvidia-l4"},
    ),
  )
  @mock.patch(
    "kinetic.backend.k8s_utils._validate_node_pool_exists",
    return_value=True,
  )
  @mock.patch("kinetic.backend.k8s_utils.logging.info")
  def test_scheduling_failure_logs(
    self, mock_info, mock_validate, condition_message, log_match, node_selector
  ):
    mock_core = MagicMock()
    pod = self._make_pending_pod(condition_message, node_selector=node_selector)
    mock_core.list_namespaced_pod.return_value.items = [pod]

    check_pod_scheduling(mock_core, "job-1", "default", set())

    # Verify it was called with something that contains log_match
    self.assertTrue(mock_info.called)
    if len(mock_info.call_args[0]) > 1:
      call_arg = mock_info.call_args[0][0] % mock_info.call_args[0][1:]
    else:
      call_arg = mock_info.call_args[0][0]
    self.assertIn(log_match, call_arg)

  @mock.patch(
    "kinetic.backend.k8s_utils._validate_node_pool_exists",
    return_value=False,
  )
  def test_scheduling_failure_raises_missing_node_pool(self, mock_validate):
    mock_core = MagicMock()
    node_selector = {"cloud.google.com/gke-accelerator": "nvidia-l4"}
    pod = self._make_pending_pod(
      "didn't match Pod's node affinity/selector", node_selector=node_selector
    )
    mock_core.list_namespaced_pod.return_value.items = [pod]

    error_match = "No GKE node pool exists.*kinetic pool add"
    with self.assertRaisesRegex(RuntimeError, error_match):
      check_pod_scheduling(mock_core, "job-1", "default", set())

  def test_running_pod_no_error(self):
    mock_core = MagicMock()
    pod = MagicMock()
    pod.status.phase = "Running"
    pod.status.conditions = []
    mock_core.list_namespaced_pod.return_value.items = [pod]

    check_pod_scheduling(
      mock_core, "job-1", "default", set()
    )  # should not raise

  def test_pending_no_conditions(self):
    mock_core = MagicMock()
    pod = MagicMock()
    pod.status.phase = "Pending"
    pod.status.conditions = None
    mock_core.list_namespaced_pod.return_value.items = [pod]

    check_pod_scheduling(
      mock_core, "job-1", "default", set()
    )  # should not raise


if __name__ == "__main__":
  absltest.main()
