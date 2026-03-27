"""Tests for kinetic.backend.gke_client — K8s job submission and monitoring."""

import json
import subprocess
from unittest import mock
from unittest.mock import MagicMock

from absl.testing import absltest, parameterized
from kubernetes.client.rest import ApiException
from kubernetes.config import ConfigException

from kinetic.backend.gke_client import (
  _check_node_pool_exists_cached,
  _check_pod_scheduling,
  _create_job_spec,
  _load_kube_config,
  _parse_accelerator,
  get_job_logs,
  get_job_pod_name,
  get_job_status,
  job_exists,
  wait_for_job,
)
from kinetic.backend.gke_client import (
  list_jobs as list_gke_jobs,
)
from kinetic.job_status import JobStatus


class TestParseAccelerator(absltest.TestCase):
  def test_cpu(self):
    result = _parse_accelerator("cpu")
    self.assertEqual(result["node_selector"], {})
    self.assertEqual(result["resource_limits"], {})
    self.assertEqual(result["resource_requests"], {})
    self.assertEqual(result["tolerations"], [])
    self.assertEqual(result["jax_platform"], "cpu")

  def test_gpu_l4(self):
    result = _parse_accelerator("l4")
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
    result = _parse_accelerator("a100x4")
    self.assertEqual(result["resource_limits"], {"nvidia.com/gpu": "4"})
    self.assertEqual(result["resource_requests"], {"nvidia.com/gpu": "4"})

  def test_tpu_v3_8(self):
    result = _parse_accelerator("v3-4")
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
    result = _parse_accelerator("v3-16")
    self.assertEqual(result["resource_limits"], {"google.com/tpu": "4"})
    self.assertEqual(result["resource_requests"], {"google.com/tpu": "4"})
    self.assertEqual(
      result["node_selector"]["cloud.google.com/gke-tpu-topology"], "4x4"
    )

  def test_tpu_v5litepod_4(self):
    result = _parse_accelerator("v5litepod-4")
    self.assertEqual(
      result["node_selector"],
      {
        "cloud.google.com/gke-tpu-accelerator": "tpu-v5-lite-podslice",
        "cloud.google.com/gke-tpu-topology": "2x2",
      },
    )
    self.assertEqual(result["resource_limits"], {"google.com/tpu": "4"})

  def test_spot_gpu(self):
    result = _parse_accelerator("l4:spot")
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
    result = _parse_accelerator("v6e-8:spot")
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


class TestCreateJobSpec(absltest.TestCase):
  def _make_gpu_config(self):
    return {
      "node_selector": {"cloud.google.com/gke-accelerator": "nvidia-l4"},
      "resource_limits": {"nvidia.com/gpu": "1"},
      "resource_requests": {"nvidia.com/gpu": "1"},
      "tolerations": [
        {
          "key": "nvidia.com/gpu",
          "operator": "Exists",
          "effect": "NoSchedule",
        }
      ],
      "jax_platform": "gpu",
    }

  def _make_cpu_config(self):
    return {
      "node_selector": {},
      "resource_limits": {},
      "resource_requests": {},
      "tolerations": [],
      "jax_platform": "cpu",
    }

  def test_gpu_job_structure(self):
    job = _create_job_spec(
      job_name="kinetic-job-abc",
      container_uri="us-docker.pkg.dev/proj/repo/img:tag",
      accel_config=self._make_gpu_config(),
      job_id="job-abc",
      bucket_name="proj-kinetic-jobs",
      namespace="default",
    )
    self.assertEqual(job.metadata.name, "kinetic-job-abc")
    self.assertEqual(job.metadata.namespace, "default")

    container = job.spec.template.spec.containers[0]
    self.assertEqual(container.image, "us-docker.pkg.dev/proj/repo/img:tag")
    self.assertEqual(
      container.command, ["python3", "-u", "/app/remote_runner.py"]
    )

    # Check env vars
    env_names = {e.name: e.value for e in container.env}
    self.assertEqual(env_names["KERAS_BACKEND"], "jax")
    self.assertEqual(env_names["JAX_PLATFORMS"], "gpu")
    self.assertEqual(env_names["JOB_ID"], "job-abc")
    self.assertEqual(env_names["GCS_BUCKET"], "proj-kinetic-jobs")

    # Check GCS args
    self.assertIn("gs://proj-kinetic-jobs/job-abc/context.zip", container.args)
    self.assertIn("gs://proj-kinetic-jobs/job-abc/payload.pkl", container.args)
    self.assertIn("gs://proj-kinetic-jobs/job-abc/result.pkl", container.args)

    # Job spec fields
    self.assertEqual(job.spec.backoff_limit, 0)
    self.assertEqual(job.spec.ttl_seconds_after_finished, 600)
    self.assertEqual(job.spec.template.spec.restart_policy, "Never")

    # Labels on both job and pod template
    self.assertEqual(job.metadata.labels["app"], "kinetic")
    self.assertEqual(job.metadata.labels["job-id"], "job-abc")
    self.assertEqual(job.spec.template.metadata.labels["app"], "kinetic")
    self.assertEqual(job.spec.template.metadata.labels["job-id"], "job-abc")

    # GPU-specific: node selector and tolerations
    self.assertIn(
      "cloud.google.com/gke-accelerator",
      job.spec.template.spec.node_selector,
    )
    tolerations = job.spec.template.spec.tolerations
    self.assertLen(tolerations, 1)
    self.assertEqual(tolerations[0].key, "nvidia.com/gpu")

  def test_cpu_job_structure(self):
    job = _create_job_spec(
      job_name="cpu-job",
      container_uri="img",
      accel_config=self._make_cpu_config(),
      job_id="j",
      bucket_name="b",
      namespace="ns",
    )
    self.assertIsNone(job.spec.template.spec.node_selector)


class TestWaitForJob(absltest.TestCase):
  def setUp(self):
    super().setUp()

    self.mock_streamer = MagicMock()
    self.enterContext(
      mock.patch(
        "kinetic.backend.gke_client.LogStreamer",
        return_value=self.mock_streamer,
      )
    )
    self.mock_streamer.__enter__ = MagicMock(return_value=self.mock_streamer)
    self.mock_streamer.__exit__ = MagicMock(return_value=False)

  def _make_mock_job(self):
    job = MagicMock()
    job.metadata.name = "kinetic-job-abc"
    return job

  def test_first_poll_success(self):
    mock_batch = MagicMock()
    mock_status = MagicMock()
    mock_status.status.succeeded = 1
    mock_status.status.failed = None
    mock_batch.read_namespaced_job_status.return_value = mock_status

    mock_core = MagicMock()
    mock_core.list_namespaced_pod.return_value.items = []

    with (
      mock.patch(
        "kinetic.backend.gke_client._batch_v1",
        return_value=mock_batch,
      ),
      mock.patch(
        "kinetic.backend.gke_client._core_v1",
        return_value=mock_core,
      ),
    ):
      result = wait_for_job(self._make_mock_job())
    self.assertEqual(result, "success")
    self.mock_streamer.start.assert_not_called()

  def test_first_poll_failure(self):
    mock_batch = MagicMock()
    mock_status = MagicMock()
    mock_status.status.succeeded = None
    mock_status.status.failed = 1
    mock_batch.read_namespaced_job_status.return_value = mock_status

    mock_core = MagicMock()
    mock_core.list_namespaced_pod.return_value.items = []

    with (
      mock.patch(
        "kinetic.backend.gke_client._batch_v1",
        return_value=mock_batch,
      ),
      mock.patch(
        "kinetic.backend.gke_client._core_v1",
        return_value=mock_core,
      ),
      self.assertRaisesRegex(RuntimeError, "failed"),
    ):
      wait_for_job(self._make_mock_job())

  def test_timeout_raises(self):
    mock_batch = MagicMock()
    mock_status = MagicMock()
    mock_status.status.succeeded = None
    mock_status.status.failed = None
    mock_batch.read_namespaced_job_status.return_value = mock_status

    with (
      mock.patch(
        "kinetic.backend.gke_client._batch_v1",
        return_value=mock_batch,
      ),
      mock.patch("kinetic.backend.gke_client._core_v1"),
      mock.patch("kinetic.backend.gke_client.time.sleep"),
      self.assertRaisesRegex(RuntimeError, "timed out"),
    ):
      wait_for_job(self._make_mock_job(), timeout=0)

  def test_polls_until_success(self):
    mock_batch = MagicMock()
    running = MagicMock()
    running.status.succeeded = None
    running.status.failed = None
    succeeded = MagicMock()
    succeeded.status.succeeded = 1
    succeeded.status.failed = None
    mock_batch.read_namespaced_job_status.side_effect = [running, succeeded]

    mock_core = MagicMock()
    mock_core.list_namespaced_pod.return_value.items = []

    with (
      mock.patch(
        "kinetic.backend.gke_client._batch_v1",
        return_value=mock_batch,
      ),
      mock.patch(
        "kinetic.backend.gke_client._core_v1",
        return_value=mock_core,
      ),
      mock.patch("kinetic.backend.gke_client.time.sleep") as mock_sleep,
    ):
      result = wait_for_job(self._make_mock_job(), poll_interval=5)
    self.assertEqual(result, "success")
    mock_sleep.assert_called_with(5)

  def test_starts_streaming_when_pod_running(self):
    mock_batch = MagicMock()
    running = MagicMock()
    running.status.succeeded = None
    running.status.failed = None
    succeeded = MagicMock()
    succeeded.status.succeeded = 1
    succeeded.status.failed = None
    mock_batch.read_namespaced_job_status.side_effect = [running, succeeded]

    running_pod = MagicMock()
    running_pod.status.phase = "Running"
    running_pod.metadata.name = "kinetic-job-abc-pod"

    mock_core = MagicMock()
    mock_core.list_namespaced_pod.return_value.items = [running_pod]

    with (
      mock.patch(
        "kinetic.backend.gke_client._batch_v1",
        return_value=mock_batch,
      ),
      mock.patch(
        "kinetic.backend.gke_client._core_v1",
        return_value=mock_core,
      ),
      mock.patch("kinetic.backend.gke_client.time.sleep"),
    ):
      result = wait_for_job(self._make_mock_job())

    self.assertEqual(result, "success")
    self.mock_streamer.start.assert_called_once_with("kinetic-job-abc-pod")

  def test_no_streaming_when_pod_pending(self):
    mock_batch = MagicMock()
    running = MagicMock()
    running.status.succeeded = None
    running.status.failed = None
    succeeded = MagicMock()
    succeeded.status.succeeded = 1
    succeeded.status.failed = None
    mock_batch.read_namespaced_job_status.side_effect = [running, succeeded]

    pending_pod = MagicMock()
    pending_pod.status.phase = "Pending"
    pending_pod.status.conditions = None

    mock_core = MagicMock()
    mock_core.list_namespaced_pod.return_value.items = [pending_pod]

    with (
      mock.patch(
        "kinetic.backend.gke_client._batch_v1",
        return_value=mock_batch,
      ),
      mock.patch(
        "kinetic.backend.gke_client._core_v1",
        return_value=mock_core,
      ),
      mock.patch("kinetic.backend.gke_client.time.sleep"),
    ):
      result = wait_for_job(self._make_mock_job())

    self.assertEqual(result, "success")
    self.mock_streamer.start.assert_not_called()


class TestAsyncObservationHelpers(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.mock_batch = self.enterContext(
      mock.patch("kinetic.backend.gke_client._batch_v1")
    ).return_value
    self.mock_core = self.enterContext(
      mock.patch("kinetic.backend.gke_client._core_v1")
    ).return_value

  def _make_job_status(self, *, succeeded=None, failed=None):
    job_status = MagicMock()
    job_status.status.succeeded = succeeded
    job_status.status.failed = failed
    return job_status

  def _make_pod(self, phase, name="pod-1"):
    pod = MagicMock()
    pod.status.phase = phase
    pod.metadata.name = name
    return pod

  def test_get_job_status_succeeded(self):
    self.mock_batch.read_namespaced_job_status.return_value = (
      self._make_job_status(succeeded=1)
    )

    status = get_job_status("kinetic-job-1")

    self.assertEqual(status, JobStatus.SUCCEEDED)

  def test_get_job_status_failed(self):
    self.mock_batch.read_namespaced_job_status.return_value = (
      self._make_job_status(failed=1)
    )

    status = get_job_status("kinetic-job-1")

    self.assertEqual(status, JobStatus.FAILED)

  def test_get_job_status_running(self):
    self.mock_batch.read_namespaced_job_status.return_value = (
      self._make_job_status()
    )
    self.mock_core.list_namespaced_pod.return_value.items = [
      self._make_pod("Running")
    ]

    status = get_job_status("kinetic-job-1")

    self.assertEqual(status, JobStatus.RUNNING)

  def test_get_job_status_pending_when_no_pod_yet(self):
    self.mock_batch.read_namespaced_job_status.return_value = (
      self._make_job_status()
    )
    self.mock_core.list_namespaced_pod.return_value.items = []

    status = get_job_status("kinetic-job-1")

    self.assertEqual(status, JobStatus.PENDING)

  def test_get_job_status_not_found(self):
    self.mock_batch.read_namespaced_job_status.side_effect = ApiException(
      status=404, reason="Not Found"
    )

    status = get_job_status("kinetic-job-1")

    self.assertEqual(status, JobStatus.NOT_FOUND)

  def test_get_job_pod_name_prefers_running_pod(self):
    self.mock_core.list_namespaced_pod.return_value.items = [
      self._make_pod("Pending", name="pending-pod"),
      self._make_pod("Running", name="running-pod"),
    ]

    pod_name = get_job_pod_name("kinetic-job-1")

    self.assertEqual(pod_name, "running-pod")

  def test_get_job_logs_reads_selected_pod(self):
    self.mock_core.list_namespaced_pod.return_value.items = [
      self._make_pod("Running", name="running-pod")
    ]
    self.mock_core.read_namespaced_pod_log.return_value = "log output"

    logs = get_job_logs("kinetic-job-1", tail_lines=20)

    self.assertEqual(logs, "log output")
    self.mock_core.read_namespaced_pod_log.assert_called_once_with(
      "running-pod",
      "default",
      tail_lines=20,
    )

  def test_get_job_logs_missing_pod_raises(self):
    self.mock_core.list_namespaced_pod.return_value.items = []

    with self.assertRaisesRegex(RuntimeError, "No pod found"):
      get_job_logs("kinetic-job-1")

  def test_job_exists_true(self):
    self.mock_batch.read_namespaced_job_status.return_value = (
      self._make_job_status()
    )

    self.assertTrue(job_exists("kinetic-job-1"))

  def test_job_exists_false_for_404(self):
    self.mock_batch.read_namespaced_job_status.side_effect = ApiException(
      status=404, reason="Not Found"
    )

    self.assertFalse(job_exists("kinetic-job-1"))

  def test_job_exists_raises_on_non_404(self):
    self.mock_batch.read_namespaced_job_status.side_effect = ApiException(
      status=500, reason="Internal Server Error"
    )

    with self.assertRaisesRegex(RuntimeError, "Failed to read job status"):
      job_exists("kinetic-job-1")

  def test_list_jobs_returns_labelled_jobs(self):
    job_with_label = MagicMock()
    job_with_label.metadata.labels = {"job-id": "job-1"}
    job_with_label.metadata.name = "kinetic-job-1"
    job_without_label = MagicMock()
    job_without_label.metadata.labels = {}
    job_without_label.metadata.name = "ignored"
    self.mock_batch.list_namespaced_job.return_value.items = [
      job_with_label,
      job_without_label,
    ]

    jobs = list_gke_jobs("team-ns")

    self.assertEqual(
      jobs,
      [{"job_id": "job-1", "k8s_name": "kinetic-job-1"}],
    )
    self.mock_batch.list_namespaced_job.assert_called_once_with(
      namespace="team-ns",
      label_selector="app=kinetic",
    )


class TestLoadKubeConfig(absltest.TestCase):
  def setUp(self):
    super().setUp()
    _load_kube_config.cache_clear()
    self.addCleanup(_load_kube_config.cache_clear)

  def test_kubeconfig_fallback(self):
    """Falls back to local kubeconfig when in-cluster config is unavailable."""
    with (
      mock.patch(
        "kinetic.backend.gke_client.config.load_incluster_config",
        side_effect=ConfigException("not in cluster"),
      ),
      mock.patch("kinetic.backend.gke_client.config.load_kube_config"),
    ):
      _load_kube_config()


class TestCheckNodePoolExistsCached(absltest.TestCase):
  def setUp(self):
    super().setUp()
    _check_node_pool_exists_cached.cache_clear()

    self.enterContext(
      mock.patch(
        "kinetic.backend.gke_client.config.kube_config.list_kube_config_contexts",
        return_value=(
          None,
          {"name": "gke_test-project_us-central1-c_test-cluster"},
        ),
      )
    )
    self.mock_run = self.enterContext(
      mock.patch(
        "kinetic.backend.gke_client.subprocess.check_output",
        text=True,
        stderr=subprocess.DEVNULL,
      )
    )
    self.mock_warning = self.enterContext(
      mock.patch("kinetic.backend.gke_client.logging.warning")
    )

  def test_gpu_match(self):
    self.mock_run.return_value = json.dumps(
      [
        {
          "config": {
            "accelerators": [{"acceleratorType": "nvidia-l4"}],
            "labels": {"existing-label": "true"},
          }
        }
      ]
    )

    result = _check_node_pool_exists_cached(
      (("cloud.google.com/gke-accelerator", "nvidia-l4"),)
    )
    self.assertTrue(result)

  def test_tpu_match(self):
    self.mock_run.return_value = json.dumps(
      [
        {
          "config": {
            "machineType": "ct5lp-hightpu-4t",
            "accelerators": [{"acceleratorType": "tpu-v5-lite-podslice"}],
            "labels": {},
          }
        }
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
    self.mock_run.return_value = json.dumps(
      [
        {
          "config": {
            "machineType": "ct6e-standard-4t",
            "accelerators": [{"acceleratorType": "tpu-v6e-slice"}],
            "labels": {},
          }
        }
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
    self.mock_run.return_value = json.dumps(
      [
        {
          "config": {
            "accelerators": [{"acceleratorType": "nvidia-t4"}],
          }
        }
      ]
    )

    result = _check_node_pool_exists_cached(
      (("cloud.google.com/gke-accelerator", "nvidia-l4"),)
    )
    self.assertFalse(result)

  def test_graceful_degradation_on_subprocess_error(self):
    self.mock_run.side_effect = Exception("gcloud not found")

    result = _check_node_pool_exists_cached(
      (("cloud.google.com/gke-accelerator", "nvidia-l4"),)
    )
    self.assertTrue(result)
    self.mock_warning.assert_called_once()
    self.assertIn(
      "Could not verify node pool existence", self.mock_warning.call_args[0][0]
    )

  def test_kubeconfig_parse_missing_context(self):
    with mock.patch(
      "kinetic.backend.gke_client.config.kube_config.list_kube_config_contexts",
      return_value=(None, {"name": "minikube"}),
    ):
      self.mock_run.return_value = "[]"
      _check_node_pool_exists_cached(
        (("cloud.google.com/gke-accelerator", "nvidia-l4"),)
      )
      cmd = self.mock_run.call_args[0][0]
      self.assertEqual(
        cmd, ["gcloud", "container", "node-pools", "list", "--format", "json"]
      )

  def test_kubeconfig_parse_success(self):
    self.mock_run.return_value = "[]"
    _check_node_pool_exists_cached(
      (("cloud.google.com/gke-accelerator", "nvidia-l4"),)
    )
    cmd = self.mock_run.call_args[0][0]
    self.assertIn("--cluster", cmd)
    self.assertIn("test-cluster", cmd)
    self.assertIn("--location", cmd)
    self.assertIn("us-central1-c", cmd)
    self.assertIn("--project", cmd)
    self.assertIn("test-project", cmd)


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
    "kinetic.backend.gke_client._validate_node_pool_exists",
    return_value=True,
  )
  @mock.patch("kinetic.backend.gke_client.logging.info")
  def test_scheduling_failure_logs(
    self, mock_info, mock_validate, condition_message, log_match, node_selector
  ):
    mock_core = MagicMock()
    pod = self._make_pending_pod(condition_message, node_selector=node_selector)
    mock_core.list_namespaced_pod.return_value.items = [pod]

    _check_pod_scheduling(mock_core, "job-1", "default", set())

    # Verify it was called with something that contains log_match
    self.assertTrue(mock_info.called)
    if len(mock_info.call_args[0]) > 1:
      call_arg = mock_info.call_args[0][0] % mock_info.call_args[0][1:]
    else:
      call_arg = mock_info.call_args[0][0]
    self.assertIn(log_match, call_arg)

  @mock.patch(
    "kinetic.backend.gke_client._validate_node_pool_exists",
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
      _check_pod_scheduling(mock_core, "job-1", "default", set())

  def test_running_pod_no_error(self):
    mock_core = MagicMock()
    pod = MagicMock()
    pod.status.phase = "Running"
    pod.status.conditions = []
    mock_core.list_namespaced_pod.return_value.items = [pod]

    _check_pod_scheduling(
      mock_core, "job-1", "default", set()
    )  # should not raise

  def test_pending_no_conditions(self):
    mock_core = MagicMock()
    pod = MagicMock()
    pod.status.phase = "Pending"
    pod.status.conditions = None
    mock_core.list_namespaced_pod.return_value.items = [pod]

    _check_pod_scheduling(
      mock_core, "job-1", "default", set()
    )  # should not raise


if __name__ == "__main__":
  absltest.main()
