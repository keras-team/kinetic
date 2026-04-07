"""Tests for kinetic.backend.gke_client — K8s job submission and monitoring."""

from unittest import mock
from unittest.mock import MagicMock

from absl.testing import absltest
from kubernetes.client.rest import ApiException

from kinetic.backend.gke_client import (
  _create_job_spec,
  get_job_logs,
  get_job_pod_name,
  get_job_status,
  job_exists,
  wait_for_job,
)
from kinetic.backend.gke_client import (
  list_jobs as list_gke_jobs,
)
from kinetic.backend.k8s_utils import (
  GCSFUSE_CSI_DRIVER,
  GCSFUSE_VOLUMES_ANNOTATION,
)
from kinetic.job_status import JobStatus


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

  def test_no_fuse_no_volumes_or_annotations(self):
    job = _create_job_spec(
      job_name="no-fuse",
      container_uri="img",
      accel_config=self._make_gpu_config(),
      job_id="j",
      bucket_name="b",
      namespace="ns",
    )
    self.assertIsNone(job.spec.template.metadata.annotations)
    self.assertIsNone(job.spec.template.spec.volumes)
    container = job.spec.template.spec.containers[0]
    self.assertIsNone(container.volume_mounts)

  def test_fuse_single_volume(self):
    fuse_specs = [
      {
        "gcs_uri": "gs://my-bucket/datasets/imagenet/",
        "mount_path": "/data",
        "is_dir": True,
        "read_only": True,
      }
    ]
    job = _create_job_spec(
      job_name="fuse-job",
      container_uri="img",
      accel_config=self._make_gpu_config(),
      job_id="j",
      bucket_name="b",
      namespace="ns",
      fuse_volume_specs=fuse_specs,
    )
    # Annotation
    annotations = job.spec.template.metadata.annotations
    self.assertEqual(annotations[GCSFUSE_VOLUMES_ANNOTATION], "true")

    # CSI volume
    volumes = job.spec.template.spec.volumes
    self.assertLen(volumes, 1)
    vol = volumes[0]
    self.assertEqual(vol.name, "gcs-fuse-0")
    self.assertEqual(vol.csi.driver, GCSFUSE_CSI_DRIVER)
    self.assertEqual(vol.csi.volume_attributes["bucketName"], "my-bucket")
    self.assertIn(
      "only-dir=datasets/imagenet", vol.csi.volume_attributes["mountOptions"]
    )
    self.assertIn("implicit-dirs", vol.csi.volume_attributes["mountOptions"])

    # Volume mount
    container = job.spec.template.spec.containers[0]
    self.assertLen(container.volume_mounts, 1)
    mount = container.volume_mounts[0]
    self.assertEqual(mount.name, "gcs-fuse-0")
    self.assertEqual(mount.mount_path, "/data")
    self.assertTrue(mount.read_only)

  def test_fuse_multiple_volumes(self):
    fuse_specs = [
      {
        "gcs_uri": "gs://bucket-a/data/",
        "mount_path": "/data1",
        "is_dir": True,
        "read_only": True,
      },
      {
        "gcs_uri": "gs://bucket-b/models/",
        "mount_path": "/data2",
        "is_dir": True,
        "read_only": True,
      },
    ]
    job = _create_job_spec(
      job_name="fuse-multi",
      container_uri="img",
      accel_config=self._make_gpu_config(),
      job_id="j",
      bucket_name="b",
      namespace="ns",
      fuse_volume_specs=fuse_specs,
    )
    volumes = job.spec.template.spec.volumes
    self.assertLen(volumes, 2)
    self.assertEqual(volumes[0].name, "gcs-fuse-0")
    self.assertEqual(volumes[1].name, "gcs-fuse-1")

    container = job.spec.template.spec.containers[0]
    self.assertLen(container.volume_mounts, 2)
    self.assertEqual(container.volume_mounts[0].mount_path, "/data1")
    self.assertEqual(container.volume_mounts[1].mount_path, "/data2")

  def test_fuse_bucket_root_no_only_dir(self):
    fuse_specs = [
      {
        "gcs_uri": "gs://my-bucket/",
        "mount_path": "/data",
        "is_dir": True,
        "read_only": True,
      }
    ]
    job = _create_job_spec(
      job_name="fuse-root",
      container_uri="img",
      accel_config=self._make_gpu_config(),
      job_id="j",
      bucket_name="b",
      namespace="ns",
      fuse_volume_specs=fuse_specs,
    )
    vol = job.spec.template.spec.volumes[0]
    self.assertEqual(vol.csi.volume_attributes["mountOptions"], "implicit-dirs")
    self.assertNotIn("only-dir", vol.csi.volume_attributes["mountOptions"])

  def test_fuse_single_file_mounts_parent_dir(self):
    fuse_specs = [
      {
        "gcs_uri": "gs://my-bucket/data/weights.h5",
        "mount_path": "/weights",
        "is_dir": False,
        "read_only": True,
      }
    ]
    job = _create_job_spec(
      job_name="fuse-file",
      container_uri="img",
      accel_config=self._make_gpu_config(),
      job_id="j",
      bucket_name="b",
      namespace="ns",
      fuse_volume_specs=fuse_specs,
    )
    vol = job.spec.template.spec.volumes[0]
    # Should mount the parent directory "data", not the file path.
    self.assertIn("only-dir=data", vol.csi.volume_attributes["mountOptions"])
    self.assertNotIn("weights.h5", vol.csi.volume_attributes["mountOptions"])


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
        "kinetic.backend.k8s_utils.core_v1",
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
        "kinetic.backend.k8s_utils.core_v1",
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
      mock.patch("kinetic.backend.k8s_utils.core_v1"),
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
        "kinetic.backend.k8s_utils.core_v1",
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
        "kinetic.backend.k8s_utils.core_v1",
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
        "kinetic.backend.k8s_utils.core_v1",
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
      mock.patch("kinetic.backend.k8s_utils.core_v1")
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


if __name__ == "__main__":
  absltest.main()
