"""Tests for kinetic.backend.pathways_client — LWS job submission and monitoring."""

from unittest import mock
from unittest.mock import MagicMock

from absl.testing import absltest
from kubernetes.client.rest import ApiException

from kinetic.backend.k8s_utils import (
  GCSFUSE_CSI_DRIVER,
  GCSFUSE_VOLUMES_ANNOTATION,
)
from kinetic.backend.pathways_client import (
  LWS_GROUP,
  LWS_PLURAL,
  LWS_VERSION,
  _create_lws_spec,
  _get_lws_version,
  cleanup_job,
  get_job_logs,
  get_job_status,
  job_exists,
  submit_pathways_job,
  wait_for_job,
)
from kinetic.backend.pathways_client import (
  list_jobs as list_pathways_jobs,
)
from kinetic.job_status import JobStatus

_MODULE = "kinetic.backend.pathways_client"


class TestGetLwsVersion(absltest.TestCase):
  def test_returns_preferred_version(self):
    """Test that if the LWS API group is found, we return its preferred version."""
    mock_api = MagicMock()
    group = MagicMock()
    group.name = LWS_GROUP
    group.preferred_version.version = "v2"
    mock_api.get_api_versions.return_value.groups = [group]

    with mock.patch(f"{_MODULE}._apis_api", return_value=mock_api):
      self.assertEqual(_get_lws_version(), "v2")

  def test_group_not_found_falls_back(self):
    """Test that if the LWS API group is not found, we fall back to the default version."""
    mock_api = MagicMock()
    other_group = MagicMock()
    other_group.name = "other.group.io"
    mock_api.get_api_versions.return_value.groups = [other_group]

    with mock.patch(f"{_MODULE}._apis_api", return_value=mock_api):
      self.assertEqual(_get_lws_version(), LWS_VERSION)

  def test_api_exception_falls_back(self):
    """Test that if the API call to get versions fails, we fall back to the default version."""
    mock_api = MagicMock()
    mock_api.get_api_versions.side_effect = ApiException(
      status=500, reason="Server Error"
    )

    with mock.patch(f"{_MODULE}._apis_api", return_value=mock_api):
      self.assertEqual(_get_lws_version(), LWS_VERSION)


class TestCreateLwsSpec(absltest.TestCase):
  def _make_tpu_accel_config(self):
    return {
      "node_selector": {
        "cloud.google.com/gke-tpu-accelerator": "tpu-v5-lite-podslice",
        "cloud.google.com/gke-tpu-topology": "2x2",
      },
      "resource_limits": {"google.com/tpu": "4"},
      "resource_requests": {"google.com/tpu": "4"},
      "tolerations": [
        {"key": "google.com/tpu", "operator": "Exists", "effect": "NoSchedule"}
      ],
      "jax_platform": "tpu",
    }

  def _make_cpu_accel_config(self):
    return {
      "node_selector": {},
      "resource_limits": {},
      "resource_requests": {},
      "tolerations": [],
      "jax_platform": "cpu",
    }

  def _make_spec(self, **overrides):
    defaults = {
      "job_name": "keras-pathways-abc",
      "container_uri": "us-docker.pkg.dev/proj/repo/img:tag",
      "accel_config": self._make_tpu_accel_config(),
      "job_id": "abc",
      "bucket_name": "my-bucket",
      "num_workers": 3,
      "namespace": "default",
    }
    defaults.update(overrides)
    return _create_lws_spec(**defaults)

  def test_basic_spec_structure(self):
    """Test metadata, replicas, and container spec are correctly populated."""
    spec = self._make_spec(job_id="j1", bucket_name="bkt", num_workers=3)
    # Metadata.
    self.assertEqual(spec["metadata"]["name"], "keras-pathways-abc")
    self.assertEqual(spec["metadata"]["namespace"], "default")
    self.assertEqual(spec["metadata"]["labels"]["app"], "kinetic-pathways")
    self.assertEqual(spec["metadata"]["labels"]["job-id"], "j1")
    # Replicas and size.
    self.assertEqual(spec["spec"]["replicas"], 1)
    self.assertEqual(spec["spec"]["leaderWorkerTemplate"]["size"], 4)
    # Container.
    container = spec["spec"]["leaderWorkerTemplate"]["leaderTemplate"]["spec"][
      "containers"
    ][0]
    self.assertEqual(container["name"], "kinetic-worker")
    self.assertEqual(container["image"], "us-docker.pkg.dev/proj/repo/img:tag")
    self.assertEqual(
      container["command"], ["python3", "-u", "/app/remote_runner.py"]
    )
    self.assertEqual(
      container["args"],
      [
        "gs://bkt/j1/context.zip",
        "gs://bkt/j1/payload.pkl",
        "gs://bkt/j1/result.pkl",
      ],
    )

  def test_env_vars(self):
    spec = self._make_spec(
      accel_config=self._make_tpu_accel_config(),
      job_id="j1",
      bucket_name="bkt",
      num_workers=3,
    )
    container = spec["spec"]["leaderWorkerTemplate"]["leaderTemplate"]["spec"][
      "containers"
    ][0]
    env = {e["name"]: e["value"] for e in container["env"]}
    self.assertEqual(env["KERAS_BACKEND"], "jax")
    self.assertEqual(env["JAX_PLATFORMS"], "tpu")
    self.assertEqual(env["JOB_ID"], "j1")
    self.assertEqual(env["GCS_BUCKET"], "bkt")
    self.assertEqual(
      env["MEGASCALE_COORDINATOR_ADDRESS"], "$(LWS_LEADER_ADDRESS)"
    )
    self.assertEqual(env["MEGASCALE_NUM_SLICES"], "4")
    self.assertEqual(env["TPU_WORKER_ID"], "$(LWS_WORKER_INDEX)")

  def test_spot_spec(self):
    """Test that spot selectors and tolerations are added when present."""
    accel_config = self._make_tpu_accel_config()
    accel_config["node_selector"]["cloud.google.com/gke-spot"] = "true"
    accel_config["tolerations"].append(
      {
        "key": "cloud.google.com/gke-spot",
        "operator": "Equal",
        "value": "true",
        "effect": "NoSchedule",
      }
    )

    spec = self._make_spec(accel_config=accel_config)
    pod_spec = spec["spec"]["leaderWorkerTemplate"]["leaderTemplate"]["spec"]

    self.assertEqual(
      pod_spec["nodeSelector"]["cloud.google.com/gke-spot"], "true"
    )
    spot_tol = [
      t
      for t in pod_spec["tolerations"]
      if t.get("key") == "cloud.google.com/gke-spot"
    ]
    self.assertLen(spot_tol, 1)
    self.assertEqual(spot_tol[0]["value"], "true")

  def test_tpu_accel_config(self):
    """Test resources, tolerations, and node selector for TPU config."""
    spec = self._make_spec(accel_config=self._make_tpu_accel_config())
    container = spec["spec"]["leaderWorkerTemplate"]["leaderTemplate"]["spec"][
      "containers"
    ][0]
    pod_spec = spec["spec"]["leaderWorkerTemplate"]["leaderTemplate"]["spec"]
    # Resources.
    self.assertEqual(container["resources"]["limits"], {"google.com/tpu": "4"})
    self.assertEqual(
      container["resources"]["requests"], {"google.com/tpu": "4"}
    )
    # Tolerations.
    self.assertLen(pod_spec["tolerations"], 1)
    self.assertEqual(pod_spec["tolerations"][0]["key"], "google.com/tpu")
    self.assertEqual(pod_spec["tolerations"][0]["operator"], "Exists")
    self.assertEqual(pod_spec["tolerations"][0]["effect"], "NoSchedule")
    # Node selector.
    self.assertIn(
      "cloud.google.com/gke-tpu-accelerator", pod_spec["nodeSelector"]
    )

  def test_cpu_accel_config(self):
    """Test that tolerations and node selector are omitted for CPU config."""
    spec = self._make_spec(accel_config=self._make_cpu_accel_config())
    pod_spec = spec["spec"]["leaderWorkerTemplate"]["leaderTemplate"]["spec"]
    self.assertNotIn("tolerations", pod_spec)
    self.assertNotIn("nodeSelector", pod_spec)

  def test_pod_labels(self):
    spec = self._make_spec(job_name="my-job", job_id="j1")
    labels = spec["spec"]["leaderWorkerTemplate"]["leaderTemplate"]["metadata"][
      "labels"
    ]
    self.assertEqual(labels["app"], "kinetic-pathways")
    self.assertEqual(labels["job-id"], "j1")
    self.assertEqual(labels["job-name"], "my-job")

  def test_custom_version(self):
    spec = self._make_spec(version="v2")
    self.assertEqual(spec["apiVersion"], f"{LWS_GROUP}/v2")

  def test_zero_workers(self):
    spec = self._make_spec(num_workers=0)
    self.assertEqual(spec["spec"]["leaderWorkerTemplate"]["size"], 1)
    container = spec["spec"]["leaderWorkerTemplate"]["leaderTemplate"]["spec"][
      "containers"
    ][0]
    env = {e["name"]: e["value"] for e in container["env"]}
    self.assertEqual(env["MEGASCALE_NUM_SLICES"], "1")

  def test_multiple_workers(self):
    spec = self._make_spec(num_workers=7)
    self.assertEqual(spec["spec"]["leaderWorkerTemplate"]["size"], 8)
    container = spec["spec"]["leaderWorkerTemplate"]["leaderTemplate"]["spec"][
      "containers"
    ][0]
    env = {e["name"]: e["value"] for e in container["env"]}
    self.assertEqual(env["MEGASCALE_NUM_SLICES"], "8")

  def test_no_fuse_no_volumes_or_annotations(self):
    spec = self._make_spec()
    pod = spec["spec"]["leaderWorkerTemplate"]["leaderTemplate"]
    self.assertNotIn("annotations", pod["metadata"])
    self.assertNotIn("volumes", pod["spec"])
    container = pod["spec"]["containers"][0]
    self.assertNotIn("volumeMounts", container)

  def test_fuse_single_volume(self):
    fuse_specs = [
      {
        "gcs_uri": "gs://my-bucket/datasets/imagenet/",
        "mount_path": "/data",
        "is_dir": True,
        "read_only": True,
      }
    ]
    spec = self._make_spec(fuse_volume_specs=fuse_specs)
    pod = spec["spec"]["leaderWorkerTemplate"]["leaderTemplate"]

    # Annotation
    self.assertEqual(
      pod["metadata"]["annotations"][GCSFUSE_VOLUMES_ANNOTATION], "true"
    )

    # CSI volume
    volumes = pod["spec"]["volumes"]
    self.assertLen(volumes, 1)
    vol = volumes[0]
    self.assertEqual(vol["name"], "gcs-fuse-0")
    self.assertEqual(vol["csi"]["driver"], GCSFUSE_CSI_DRIVER)
    self.assertEqual(vol["csi"]["volumeAttributes"]["bucketName"], "my-bucket")
    self.assertIn(
      "only-dir=datasets/imagenet",
      vol["csi"]["volumeAttributes"]["mountOptions"],
    )

    # Volume mount
    container = pod["spec"]["containers"][0]
    self.assertLen(container["volumeMounts"], 1)
    mount = container["volumeMounts"][0]
    self.assertEqual(mount["name"], "gcs-fuse-0")
    self.assertEqual(mount["mountPath"], "/data")
    self.assertTrue(mount["readOnly"])

  def test_fuse_multiple_volumes(self):
    fuse_specs = [
      {
        "gcs_uri": "gs://a/data/",
        "mount_path": "/d1",
        "is_dir": True,
        "read_only": True,
      },
      {
        "gcs_uri": "gs://b/models/",
        "mount_path": "/d2",
        "is_dir": True,
        "read_only": True,
      },
    ]
    spec = self._make_spec(fuse_volume_specs=fuse_specs)
    pod = spec["spec"]["leaderWorkerTemplate"]["leaderTemplate"]
    volumes = pod["spec"]["volumes"]
    self.assertLen(volumes, 2)
    self.assertEqual(volumes[0]["name"], "gcs-fuse-0")
    self.assertEqual(volumes[1]["name"], "gcs-fuse-1")

    mounts = pod["spec"]["containers"][0]["volumeMounts"]
    self.assertLen(mounts, 2)
    self.assertEqual(mounts[0]["mountPath"], "/d1")
    self.assertEqual(mounts[1]["mountPath"], "/d2")

  def test_fuse_bucket_root_no_only_dir(self):
    fuse_specs = [
      {
        "gcs_uri": "gs://my-bucket/",
        "mount_path": "/data",
        "is_dir": True,
        "read_only": True,
      }
    ]
    spec = self._make_spec(fuse_volume_specs=fuse_specs)
    pod = spec["spec"]["leaderWorkerTemplate"]["leaderTemplate"]
    vol = pod["spec"]["volumes"][0]
    self.assertEqual(
      vol["csi"]["volumeAttributes"]["mountOptions"], "implicit-dirs"
    )


class TestSubmitPathwaysJob(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.enterContext(
      mock.patch(f"{_MODULE}._get_lws_version", return_value="v1")
    )
    self.mock_custom_api = self.enterContext(
      mock.patch(f"{_MODULE}._custom_api")
    ).return_value

  def _call(self, **overrides):
    defaults = {
      "display_name": "my-job",
      "container_uri": "img:tag",
      "accelerator": "v5litepod-4",
      "project": "proj",
      "job_id": "j1",
      "bucket_name": "bkt",
    }
    defaults.update(overrides)
    return submit_pathways_job(**defaults)

  def _get_created_body(self):
    return self.mock_custom_api.create_namespaced_custom_object.call_args[1][
      "body"
    ]

  def test_multi_node_tpu(self):
    # v3-16 → 4 nodes → 3 workers
    self._call(accelerator="v3-16")
    body = self._get_created_body()
    self.assertEqual(body["spec"]["leaderWorkerTemplate"]["size"], 4)

  def test_single_node_tpu(self):
    # v5litepod-4 → 1 node → 0 workers
    self._call(accelerator="v5litepod-4")
    body = self._get_created_body()
    self.assertEqual(body["spec"]["leaderWorkerTemplate"]["size"], 1)

  def test_non_tpu_accelerator(self):
    self._call(accelerator="l4")
    body = self._get_created_body()
    self.assertEqual(body["spec"]["leaderWorkerTemplate"]["size"], 1)

  def test_job_name_derived_from_job_id(self):
    self._call(job_id="xyz")
    body = self._get_created_body()
    self.assertEqual(body["metadata"]["name"], "keras-pathways-xyz")


class TestWaitForJob(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.mock_print_pod_logs = self.enterContext(
      mock.patch("kinetic.backend.k8s_utils.print_pod_logs")
    )
    self.mock_check_pod_scheduling = self.enterContext(
      mock.patch("kinetic.backend.k8s_utils.check_pod_scheduling")
    )
    self.mock_sleep = self.enterContext(mock.patch(f"{_MODULE}.time.sleep"))
    self.mock_time = self.enterContext(
      mock.patch(f"{_MODULE}.time.time", return_value=0)
    )
    self.mock_core = self.enterContext(
      mock.patch("kinetic.backend.k8s_utils.core_v1")
    ).return_value

    self.mock_streamer = MagicMock()
    self.enterContext(
      mock.patch(f"{_MODULE}.LogStreamer", return_value=self.mock_streamer)
    )
    self.mock_streamer.__enter__ = MagicMock(return_value=self.mock_streamer)
    self.mock_streamer.__exit__ = MagicMock(return_value=False)

  def _make_pod(self, phase, container_statuses=None):
    pod = MagicMock()
    pod.status.phase = phase
    pod.status.container_statuses = container_statuses
    return pod

  def _make_container_status(
    self, state_terminated=None, last_state_terminated=None
  ):
    cs = MagicMock()
    cs.state.terminated = state_terminated
    cs.last_state.terminated = last_state_terminated
    return cs

  def _make_terminated(self, exit_code):
    t = MagicMock()
    t.exit_code = exit_code
    return t

  def test_immediate_success_phase(self):
    self.mock_core.read_namespaced_pod.return_value = self._make_pod(
      "Succeeded"
    )
    result = wait_for_job("j1")
    self.assertEqual(result, "success")
    self.mock_streamer.start.assert_not_called()

  def test_immediate_failure_phase(self):
    self.mock_core.read_namespaced_pod.return_value = self._make_pod("Failed")
    with self.assertRaisesRegex(RuntimeError, "failed"):
      wait_for_job("j1")
    self.mock_print_pod_logs.assert_called_once()

  def test_pending_calls_check_scheduling(self):
    pending = self._make_pod("Pending", container_statuses=None)
    succeeded = self._make_pod("Succeeded")
    self.mock_core.read_namespaced_pod.side_effect = [pending, succeeded]
    result = wait_for_job("j1")
    self.assertEqual(result, "success")
    self.mock_check_pod_scheduling.assert_called_once()

  def test_timeout_raises(self):
    # time.time() is also called by the logging module, so use a callable
    # that returns increasing values instead of a finite list.
    counter = iter(range(0, 100000, 3601))
    self.mock_time.side_effect = lambda: next(counter)
    self.mock_core.read_namespaced_pod.return_value = self._make_pod(
      "Running", container_statuses=None
    )
    with self.assertRaisesRegex(RuntimeError, "timed out"):
      wait_for_job("j1", timeout=3600)

  def test_polls_until_success(self):
    running = self._make_pod("Running", container_statuses=None)
    succeeded = self._make_pod("Succeeded")
    self.mock_core.read_namespaced_pod.side_effect = [running, succeeded]
    result = wait_for_job("j1", poll_interval=7)
    self.assertEqual(result, "success")
    self.mock_sleep.assert_called_with(7)

  def test_pod_404_retries(self):
    succeeded = self._make_pod("Succeeded")
    self.mock_core.read_namespaced_pod.side_effect = [
      ApiException(status=404, reason="Not Found"),
      succeeded,
    ]
    result = wait_for_job("j1")
    self.assertEqual(result, "success")
    self.mock_sleep.assert_called_once()

  def test_container_terminated_exit_0(self):
    cs = self._make_container_status(state_terminated=self._make_terminated(0))
    pod = self._make_pod("Running", container_statuses=[cs])
    self.mock_core.read_namespaced_pod.return_value = pod
    result = wait_for_job("j1")
    self.assertEqual(result, "success")

  def test_container_terminated_nonzero_exit(self):
    cs = self._make_container_status(state_terminated=self._make_terminated(1))
    pod = self._make_pod("Running", container_statuses=[cs])
    self.mock_core.read_namespaced_pod.return_value = pod
    with self.assertRaisesRegex(RuntimeError, "exit code 1"):
      wait_for_job("j1")
    self.mock_print_pod_logs.assert_called_once()

  def test_last_state_terminated_exit_0(self):
    cs = self._make_container_status(
      state_terminated=None,
      last_state_terminated=self._make_terminated(0),
    )
    pod = self._make_pod("Running", container_statuses=[cs])
    self.mock_core.read_namespaced_pod.return_value = pod
    result = wait_for_job("j1")
    self.assertEqual(result, "success")

  def test_last_state_terminated_nonzero_exit(self):
    cs = self._make_container_status(
      state_terminated=None,
      last_state_terminated=self._make_terminated(137),
    )
    pod = self._make_pod("Running", container_statuses=[cs])
    self.mock_core.read_namespaced_pod.return_value = pod
    with self.assertRaisesRegex(RuntimeError, "exit code 137"):
      wait_for_job("j1")
    self.mock_print_pod_logs.assert_called_once()

  def test_leader_pod_name(self):
    self.mock_core.read_namespaced_pod.return_value = self._make_pod(
      "Succeeded"
    )
    wait_for_job("j1")
    self.mock_core.read_namespaced_pod.assert_called_with(
      "keras-pathways-j1-0", "default"
    )

  def test_starts_streaming_when_pod_running(self):
    running = self._make_pod("Running", container_statuses=None)
    succeeded = self._make_pod("Succeeded")
    self.mock_core.read_namespaced_pod.side_effect = [running, succeeded]
    result = wait_for_job("j1")
    self.assertEqual(result, "success")
    self.mock_streamer.start.assert_called_once_with("keras-pathways-j1-0")

  def test_no_streaming_when_pod_pending(self):
    pending = self._make_pod("Pending", container_statuses=None)
    succeeded = self._make_pod("Succeeded")
    self.mock_core.read_namespaced_pod.side_effect = [pending, succeeded]
    result = wait_for_job("j1")
    self.assertEqual(result, "success")
    self.mock_streamer.start.assert_not_called()


class TestCleanupJob(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.enterContext(
      mock.patch(f"{_MODULE}._get_lws_version", return_value="v1")
    )
    self.mock_custom_api = self.enterContext(
      mock.patch(f"{_MODULE}._custom_api")
    ).return_value
    self.enterContext(mock.patch(f"{_MODULE}.job_exists", return_value=False))

  def test_deletes_lws(self):
    cleanup_job("my-job")
    self.mock_custom_api.delete_namespaced_custom_object.assert_called_once_with(
      group=LWS_GROUP,
      version="v1",
      namespace="default",
      plural=LWS_PLURAL,
      name="my-job",
    )

  def test_404_silently_ignored(self):
    self.mock_custom_api.delete_namespaced_custom_object.side_effect = (
      ApiException(status=404, reason="Not Found")
    )
    cleanup_job("my-job")  # should not raise

  def test_other_api_error_warns_but_no_raise(self):
    self.mock_custom_api.delete_namespaced_custom_object.side_effect = (
      ApiException(status=500, reason="Server Error")
    )
    cleanup_job("my-job")  # should not raise


class TestAsyncObservationHelpers(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.enterContext(
      mock.patch(f"{_MODULE}._get_lws_version", return_value="v1")
    )
    self.mock_core = self.enterContext(
      mock.patch("kinetic.backend.k8s_utils.core_v1")
    ).return_value
    self.mock_custom_api = self.enterContext(
      mock.patch(f"{_MODULE}._custom_api")
    ).return_value

  def _make_pod(self, phase, exit_code=None):
    pod = MagicMock()
    pod.status.phase = phase
    if exit_code is None:
      pod.status.container_statuses = None
    else:
      container_status = MagicMock()
      container_status.state.terminated = MagicMock(exit_code=exit_code)
      container_status.last_state.terminated = None
      pod.status.container_statuses = [container_status]
    return pod

  def test_get_job_status_running(self):
    self.mock_core.read_namespaced_pod.return_value = self._make_pod("Running")

    status = get_job_status("keras-pathways-job-1")

    self.assertEqual(status, JobStatus.RUNNING)

  def test_get_job_status_succeeded_from_phase(self):
    self.mock_core.read_namespaced_pod.return_value = self._make_pod(
      "Succeeded"
    )

    status = get_job_status("keras-pathways-job-1")

    self.assertEqual(status, JobStatus.SUCCEEDED)

  def test_get_job_status_failed_from_terminated_container(self):
    self.mock_core.read_namespaced_pod.return_value = self._make_pod(
      "Running", exit_code=7
    )

    status = get_job_status("keras-pathways-job-1")

    self.assertEqual(status, JobStatus.FAILED)

  def test_get_job_status_pending_when_lws_exists_but_pod_missing(self):
    self.mock_core.read_namespaced_pod.side_effect = ApiException(
      status=404, reason="Not Found"
    )
    self.mock_custom_api.get_namespaced_custom_object.return_value = {"ok": 1}

    status = get_job_status("keras-pathways-job-1")

    self.assertEqual(status, JobStatus.PENDING)

  def test_get_job_status_not_found_when_lws_missing(self):
    self.mock_core.read_namespaced_pod.side_effect = ApiException(
      status=404, reason="Not Found"
    )
    self.mock_custom_api.get_namespaced_custom_object.side_effect = (
      ApiException(status=404, reason="Not Found")
    )

    status = get_job_status("keras-pathways-job-1")

    self.assertEqual(status, JobStatus.NOT_FOUND)

  def test_get_job_status_failed_from_phase(self):
    self.mock_core.read_namespaced_pod.return_value = self._make_pod("Failed")

    status = get_job_status("keras-pathways-job-1")

    self.assertEqual(status, JobStatus.FAILED)

  def test_get_job_status_succeeded_from_terminated_container(self):
    self.mock_core.read_namespaced_pod.return_value = self._make_pod(
      "Running", exit_code=0
    )

    status = get_job_status("keras-pathways-job-1")

    self.assertEqual(status, JobStatus.SUCCEEDED)

  def test_get_job_status_failed_from_last_state(self):
    pod = MagicMock()
    pod.status.phase = "Running"
    container_status = MagicMock()
    container_status.state.terminated = None
    container_status.last_state.terminated = MagicMock(exit_code=1)
    pod.status.container_statuses = [container_status]
    self.mock_core.read_namespaced_pod.return_value = pod

    status = get_job_status("keras-pathways-job-1")

    self.assertEqual(status, JobStatus.FAILED)

  def test_get_job_status_succeeded_from_last_state(self):
    pod = MagicMock()
    pod.status.phase = "Running"
    container_status = MagicMock()
    container_status.state.terminated = None
    container_status.last_state.terminated = MagicMock(exit_code=0)
    pod.status.container_statuses = [container_status]
    self.mock_core.read_namespaced_pod.return_value = pod

    status = get_job_status("keras-pathways-job-1")

    self.assertEqual(status, JobStatus.SUCCEEDED)

  def test_get_job_status_api_error_raises(self):
    self.mock_core.read_namespaced_pod.side_effect = ApiException(
      status=500, reason="Internal Server Error"
    )

    with self.assertRaisesRegex(RuntimeError, "Failed to read leader pod"):
      get_job_status("keras-pathways-job-1")

  def test_get_job_logs_reads_leader_pod(self):
    self.mock_core.read_namespaced_pod_log.return_value = "log output"

    logs = get_job_logs("keras-pathways-job-1", tail_lines=25)

    self.assertEqual(logs, "log output")
    self.mock_core.read_namespaced_pod_log.assert_called_once_with(
      "keras-pathways-job-1-0",
      "default",
      tail_lines=25,
    )

  def test_get_job_logs_missing_leader_raises(self):
    self.mock_core.read_namespaced_pod_log.side_effect = ApiException(
      status=404, reason="Not Found"
    )

    with self.assertRaisesRegex(RuntimeError, "No leader pod found"):
      get_job_logs("keras-pathways-job-1")

  def test_job_exists_true(self):
    self.mock_custom_api.get_namespaced_custom_object.return_value = {"ok": 1}

    self.assertTrue(job_exists("keras-pathways-job-1"))

  def test_job_exists_false_for_404(self):
    self.mock_custom_api.get_namespaced_custom_object.side_effect = (
      ApiException(status=404, reason="Not Found")
    )

    self.assertFalse(job_exists("keras-pathways-job-1"))

  def test_list_jobs_returns_labelled_objects(self):
    self.mock_custom_api.list_namespaced_custom_object.return_value = {
      "items": [
        {
          "metadata": {
            "name": "keras-pathways-job-1",
            "labels": {"job-id": "job-1"},
          }
        },
        {
          "metadata": {
            "name": "ignored",
            "labels": {},
          }
        },
      ]
    }

    jobs = list_pathways_jobs("team-ns")

    self.assertEqual(
      jobs,
      [{"job_id": "job-1", "k8s_name": "keras-pathways-job-1"}],
    )
    self.mock_custom_api.list_namespaced_custom_object.assert_called_once_with(
      group=LWS_GROUP,
      version="v1",
      namespace="team-ns",
      plural=LWS_PLURAL,
      label_selector="app=kinetic-pathways",
    )


if __name__ == "__main__":
  absltest.main()
