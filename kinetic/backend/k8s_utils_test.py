"""Tests for kinetic.backend.k8s_utils — shared K8s utilities."""

from unittest import mock
from unittest.mock import MagicMock

from absl.testing import absltest, parameterized
from google.cloud import container_v1
from kubernetes import client as k8s_client
from kubernetes.config import ConfigException

from kinetic.backend.k8s_utils import (
  GCSFUSE_CSI_DRIVER,
  _check_image_pull_errors,
  _check_node_pool_exists_cached,
  build_gcs_fuse_v1_volumes,
  build_gcs_fuse_volumes,
  check_pod_scheduling,
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


class TestCheckPodScheduling(parameterized.TestCase):
  def _make_pending_pod(self, message, node_selector=None):
    pod = MagicMock()
    pod.status.phase = "Pending"
    pod.status.container_statuses = None
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
    pod.status.container_statuses = None
    mock_core.list_namespaced_pod.return_value.items = [pod]

    check_pod_scheduling(
      mock_core, "job-1", "default", set()
    )  # should not raise


class TestCheckImagePullErrors(absltest.TestCase):
  def test_no_crash_when_container_statuses_is_none(self):
    pod = MagicMock()
    pod.status.container_statuses = None
    _check_image_pull_errors(pod)  # should not raise

  def test_no_crash_when_waiting_is_none(self):
    pod = MagicMock()
    cs = MagicMock()
    cs.state.waiting = None
    pod.status.container_statuses = [cs]
    _check_image_pull_errors(pod)  # should not raise

  def test_image_pull_backoff_detected_in_scheduling_check(self):
    """ImagePullBackOff on a Pending pod propagates through check_pod_scheduling."""
    mock_core = MagicMock()
    pod = MagicMock()
    pod.metadata.name = "pod-1"
    pod.status.phase = "Pending"
    pod.status.conditions = None

    cs = MagicMock()
    cs.image = "kinetic/base-gpu:0.0.1"
    cs.state.waiting.reason = "ImagePullBackOff"
    cs.state.waiting.message = "back-off pulling image"
    pod.status.container_statuses = [cs]

    mock_core.list_namespaced_pod.return_value.items = [pod]

    with self.assertRaisesRegex(RuntimeError, "Container image pull failed"):
      check_pod_scheduling(mock_core, "job-1", "default", set())


class TestBuildGcsFuseVolumes(absltest.TestCase):
  """Tests for build_gcs_fuse_volumes."""

  def test_empty_specs_returns_empty(self):
    annotations, vols, mounts = build_gcs_fuse_volumes(None)
    self.assertIsNone(annotations)
    self.assertEmpty(vols)
    self.assertEmpty(mounts)

  def test_directory_spec_uses_only_dir(self):
    specs = [
      {
        "gcs_uri": "gs://bucket/ns/data-cache/abc123",
        "mount_path": "/data/train",
        "is_dir": True,
        "read_only": True,
      }
    ]
    _, vols, mounts = build_gcs_fuse_volumes(specs)
    mount_opts = vols[0]["csi"]["volumeAttributes"]["mountOptions"]
    self.assertIn("only-dir=ns/data-cache/abc123", mount_opts)
    self.assertEqual(mounts[0]["mountPath"], "/data/train")

  def test_single_file_scopes_to_parent_dir(self):
    """File-level URI scopes only-dir to the parent (hash) directory."""
    specs = [
      {
        "gcs_uri": "gs://bucket/ns/data-cache/abc123/config.json",
        "mount_path": "/tmp/fuse-data/0",
        "is_dir": False,
        "read_only": True,
      }
    ]
    _, vols, mounts = build_gcs_fuse_volumes(specs)
    mount_opts = vols[0]["csi"]["volumeAttributes"]["mountOptions"]
    # Should scope to the hash dir, NOT the entire data-cache/ tree
    self.assertIn("only-dir=ns/data-cache/abc123", mount_opts)
    self.assertNotIn("only-dir=ns/data-cache,", mount_opts)

  def test_gcs_native_single_file(self):
    """GCS-native file URI scopes only-dir to the containing directory."""
    specs = [
      {
        "gcs_uri": "gs://bucket/datasets/configs/model.json",
        "mount_path": "/tmp/fuse-data/0",
        "is_dir": False,
        "read_only": True,
      }
    ]
    _, vols, mounts = build_gcs_fuse_volumes(specs)
    mount_opts = vols[0]["csi"]["volumeAttributes"]["mountOptions"]
    self.assertIn("only-dir=datasets/configs", mount_opts)

  def test_annotations_set(self):
    specs = [
      {
        "gcs_uri": "gs://bucket/data/",
        "mount_path": "/data",
        "is_dir": True,
        "read_only": True,
      }
    ]
    annotations, _, _ = build_gcs_fuse_volumes(specs)
    self.assertEqual(annotations, {"gke-gcsfuse/volumes": "true"})


class TestBuildGcsFuseV1Volumes(absltest.TestCase):
  """Tests for build_gcs_fuse_v1_volumes (V1 object wrapper)."""

  def test_empty_specs_returns_empty(self):
    annotations, vols, mounts = build_gcs_fuse_v1_volumes(None)
    self.assertIsNone(annotations)
    self.assertEmpty(vols)
    self.assertEmpty(mounts)

  def test_returns_v1_types(self):
    specs = [
      {
        "gcs_uri": "gs://bucket/data/train/",
        "mount_path": "/data",
        "is_dir": True,
        "read_only": True,
      }
    ]
    annotations, vols, mounts = build_gcs_fuse_v1_volumes(specs)
    self.assertIsNotNone(annotations)
    self.assertLen(vols, 1)
    self.assertLen(mounts, 1)
    self.assertIsInstance(vols[0], k8s_client.V1Volume)
    self.assertIsInstance(mounts[0], k8s_client.V1VolumeMount)

  def test_v1_volume_attributes(self):
    specs = [
      {
        "gcs_uri": "gs://my-bucket/datasets/imagenet/",
        "mount_path": "/data",
        "is_dir": True,
        "read_only": True,
      }
    ]
    _, vols, mounts = build_gcs_fuse_v1_volumes(specs)
    vol = vols[0]
    self.assertEqual(vol.name, "gcs-fuse-0")
    self.assertEqual(vol.csi.driver, GCSFUSE_CSI_DRIVER)
    self.assertEqual(vol.csi.volume_attributes["bucketName"], "my-bucket")

    mount = mounts[0]
    self.assertEqual(mount.name, "gcs-fuse-0")
    self.assertEqual(mount.mount_path, "/data")
    self.assertTrue(mount.read_only)


if __name__ == "__main__":
  absltest.main()
