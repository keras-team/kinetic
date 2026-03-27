"""Pathways (LeaderWorkerSet) job submission for kinetic."""

import functools
import time

from absl import logging
from kubernetes import client
from kubernetes.client.rest import ApiException

from kinetic.backend.gke_client import (
  _check_pod_scheduling,
  _core_v1,
  _load_kube_config,
  _parse_accelerator,
  _print_pod_logs,
)
from kinetic.backend.log_streaming import LogStreamer
from kinetic.core import accelerators
from kinetic.job_status import JobStatus

LWS_GROUP = "leaderworkerset.x-k8s.io"
LWS_VERSION = "v1"
LWS_PLURAL = "leaderworkersets"


@functools.lru_cache(maxsize=1)
def _custom_api():
  """Return a cached CustomObjectsApi client, loading kubeconfig on first call."""
  _load_kube_config()
  return client.CustomObjectsApi()


@functools.lru_cache(maxsize=1)
def _apis_api():
  """Return a cached ApisApi client, loading kubeconfig on first call."""
  _load_kube_config()
  return client.ApisApi()


def _get_job_name(job_id: str) -> str:
  """Get the standardized Pathways job name for a given job ID."""
  return f"keras-pathways-{job_id}"


def _get_lws_version(group=LWS_GROUP):
  """Get the preferred version for the LeaderWorkerSet API."""
  api = _apis_api()
  try:
    api_groups = api.get_api_versions().groups
    for api_group in api_groups:
      if api_group.name == group:
        return api_group.preferred_version.version

    # If we didn't find the group, raise ApiException to fallback
    raise ApiException(status=404, reason=f"API group {group} not found")
  except ApiException:
    logging.warning(
      "Failed to retrieve LWS API version from cluster. Defaulting to '%s'",
      LWS_VERSION,
    )
    return LWS_VERSION


def submit_pathways_job(
  display_name,
  container_uri,
  accelerator,
  project,
  job_id,
  bucket_name,
  namespace="default",
  spot=False,
):
  """Submit a LeaderWorkerSet to GKE cluster.

  Args:
      display_name: Job display name (used for K8s LWS name)
      container_uri: Docker container image URI
      accelerator: TPU type (must be TpuConfig)
      project: GCP project ID
      job_id: Unique job identifier
      bucket_name: GCS bucket name for artifacts
      namespace: Kubernetes namespace (default: "default")

  Returns:
      dict: The created LeaderWorkerSet object
  """
  lws_version = _get_lws_version()

  parsed_config = accelerators.parse_accelerator(accelerator, spot=spot)
  accel_config = _parse_accelerator(accelerator, spot=spot)
  job_name = _get_job_name(job_id)

  if (
    isinstance(parsed_config, accelerators.TpuConfig)
    and parsed_config.num_nodes > 1
  ):
    num_workers = parsed_config.num_nodes - 1
  else:
    num_workers = 0

  lws_manifest = _create_lws_spec(
    job_name=job_name,
    container_uri=container_uri,
    accel_config=accel_config,
    job_id=job_id,
    bucket_name=bucket_name,
    num_workers=num_workers,
    namespace=namespace,
    version=lws_version,
  )

  custom_api = _custom_api()

  try:
    created_lws = custom_api.create_namespaced_custom_object(
      group=LWS_GROUP,
      version=lws_version,
      namespace=namespace,
      plural=LWS_PLURAL,
      body=lws_manifest,
    )
    logging.info(f"Submitted Pathways job (LWS): {job_name}")
    logging.info(
      "View job with: kubectl get %s %s -n %s", LWS_PLURAL, job_name, namespace
    )
    return created_lws
  except ApiException as e:
    if e.status == 404:
      raise RuntimeError(
        "LeaderWorkerSet CRD not found. Please ensure it is "
        "installed on the cluster. You can install it by running "
        "the `kinetic up` command, or by following the "
        "official LWS installation guide."
      ) from e
    else:
      raise RuntimeError(
        f"Kubernetes API error: {e.status} - {e.reason}: {e.body}"
      ) from e


def wait_for_job(job_id, namespace="default", timeout=3600, poll_interval=10):
  """Wait for Pathways Job (LeaderWorkerSet) to complete."""
  core_v1 = _core_v1()

  job_name = _get_job_name(job_id)
  start_time = time.time()
  logged_running = False

  # The leader pod is suffixed with '-0' by LWS
  leader_pod_name = _get_leader_pod_name(job_name)

  logged_pending = set()
  with LogStreamer(core_v1, namespace) as streamer:
    while True:
      elapsed = time.time() - start_time
      if elapsed > timeout:
        raise RuntimeError(
          f"Pathways job {job_name} timed out after {timeout}s"
        )

      try:
        pod = core_v1.read_namespaced_pod(leader_pod_name, namespace)
        if not logged_running:
          logging.info(f"Found pod: {leader_pod_name}")
          logged_running = True

        if pod.status.phase == "Succeeded":
          logging.info(f"[REMOTE] Job {job_name} completed successfully")
          return "success"

        if pod.status.phase == "Failed":
          _print_pod_logs(core_v1, job_name, namespace)
          raise RuntimeError(f"Pathways job {job_name} failed")

        elif pod.status.phase == "Pending":
          _check_pod_scheduling(core_v1, job_name, namespace, logged_pending)
          logging.debug("Pod is Pending...")

        elif pod.status.phase == "Running":
          streamer.start(leader_pod_name)

      except ApiException as e:
        if e.status == 404:
          # Pod might not be created yet
          pod = None
        else:
          raise RuntimeError(
            f"Failed to read leader pod status: {e.reason}"
          ) from e

      if pod is not None and pod.status.container_statuses:
        container_status = pod.status.container_statuses[0]

        # Check current state
        if container_status.state.terminated:
          if container_status.state.terminated.exit_code == 0:
            logging.info(f"[REMOTE] Job {job_name} completed successfully")
            return "success"
          else:
            _print_pod_logs(core_v1, job_name, namespace)
            raise RuntimeError(
              f"Pathways job {job_name} failed with exit code "
              f"{container_status.state.terminated.exit_code}"
            )

        # Check last state (in case it restarted)
        if container_status.last_state.terminated:
          if container_status.last_state.terminated.exit_code == 0:
            logging.info(
              f"[REMOTE] Job {job_name} completed successfully (restarted)"
            )
            return "success"
          else:
            _print_pod_logs(core_v1, job_name, namespace)
            raise RuntimeError(
              f"Pathways job {job_name} failed previously with "
              f"exit code {container_status.last_state.terminated.exit_code}"
            )

      time.sleep(poll_interval)


def cleanup_job(
  job_name, namespace="default", timeout: float = 180, poll_interval: float = 2
):
  """Delete LeaderWorkerSet.

  Blocks until the API confirms the resource is gone (404).

  Args:
      job_name: Name of the LeaderWorkerSet
      namespace: Kubernetes namespace
      timeout: Maximum seconds to wait for deletion (default 180)
      poll_interval: Seconds between existence checks (default 2)
  """
  lws_version = _get_lws_version()
  custom_api = _custom_api()

  try:
    custom_api.delete_namespaced_custom_object(
      group=LWS_GROUP,
      version=lws_version,
      namespace=namespace,
      plural=LWS_PLURAL,
      name=job_name,
    )
    logging.info(f"Deleted LeaderWorkerSet: {job_name}")
  except ApiException as e:
    if e.status == 404:
      # Job already deleted
      return
    else:
      logging.warning(
        "Failed to delete LeaderWorkerSet %s: %s",
        job_name,
        e.reason,
      )
      return

  # Deletion is async; poll until the resource is gone.
  max_attempts = max(1, timeout // poll_interval)
  for _ in range(max_attempts):
    if not job_exists(job_name, namespace):
      return
    time.sleep(poll_interval)
  logging.warning(
    "Timed out waiting for LeaderWorkerSet %s to be deleted", job_name
  )


def job_exists(job_name, namespace="default") -> bool:
  """Return whether a namespaced LeaderWorkerSet currently exists."""
  lws_version = _get_lws_version()
  custom_api = _custom_api()
  try:
    custom_api.get_namespaced_custom_object(
      group=LWS_GROUP,
      version=lws_version,
      namespace=namespace,
      plural=LWS_PLURAL,
      name=job_name,
    )
    return True
  except ApiException as e:
    if e.status == 404:
      return False
    raise RuntimeError(
      f"Failed to read LeaderWorkerSet {job_name}: {e.reason}"
    ) from e


def get_job_status(job_name, namespace="default") -> JobStatus:
  """Return the current Pathways job status for async observation APIs."""
  core_v1 = _core_v1()
  leader_pod_name = _get_leader_pod_name(job_name)

  try:
    pod = core_v1.read_namespaced_pod(leader_pod_name, namespace)
  except ApiException as e:
    if e.status == 404:
      return (
        JobStatus.PENDING
        if job_exists(job_name, namespace)
        else JobStatus.NOT_FOUND
      )
    raise RuntimeError(f"Failed to read leader pod status: {e.reason}") from e

  if pod.status.phase == "Succeeded":
    return JobStatus.SUCCEEDED
  if pod.status.phase == "Failed":
    return JobStatus.FAILED
  if pod.status.container_statuses:
    container_status = pod.status.container_statuses[0]
    if container_status.state.terminated:
      return (
        JobStatus.SUCCEEDED
        if container_status.state.terminated.exit_code == 0
        else JobStatus.FAILED
      )
    if container_status.last_state.terminated:
      return (
        JobStatus.SUCCEEDED
        if container_status.last_state.terminated.exit_code == 0
        else JobStatus.FAILED
      )
  if pod.status.phase == "Running":
    return JobStatus.RUNNING
  return JobStatus.PENDING


def get_job_logs(
  job_name, namespace="default", tail_lines: int | None = None
) -> str:
  """Return logs for the leader pod of a Pathways job."""
  core_v1 = _core_v1()
  leader_pod_name = _get_leader_pod_name(job_name)

  log_kwargs = {}
  if tail_lines is not None:
    log_kwargs["tail_lines"] = tail_lines
  try:
    return core_v1.read_namespaced_pod_log(
      leader_pod_name,
      namespace,
      **log_kwargs,
    )
  except ApiException as e:
    if e.status == 404:
      raise RuntimeError(
        f"No leader pod found for Pathways job {job_name}"
      ) from e
    raise RuntimeError(f"Failed to read leader pod logs: {e.reason}") from e


def get_job_pod_name(job_name, namespace="default") -> str | None:
  """Return the leader pod name for a Pathways job if it exists."""
  core_v1 = _core_v1()
  leader_pod_name = _get_leader_pod_name(job_name)
  try:
    core_v1.read_namespaced_pod(leader_pod_name, namespace)
  except ApiException as e:
    if e.status == 404:
      return None
    raise RuntimeError(f"Failed to read leader pod status: {e.reason}") from e
  return leader_pod_name


def list_jobs(namespace="default") -> list[dict[str, str]]:
  """List live Pathways jobs managed by Kinetic in a namespace."""
  lws_version = _get_lws_version()
  custom_api = _custom_api()
  objects = custom_api.list_namespaced_custom_object(
    group=LWS_GROUP,
    version=lws_version,
    namespace=namespace,
    plural=LWS_PLURAL,
    label_selector="app=kinetic-pathways",
  )

  results = []
  for item in objects.get("items", []):
    metadata = item.get("metadata", {})
    labels = metadata.get("labels", {})
    job_id = labels.get("job-id")
    name = metadata.get("name")
    if job_id is None or name is None:
      continue
    results.append(
      {
        "job_id": job_id,
        "k8s_name": name,
      }
    )
  return results


def _create_lws_spec(
  job_name,
  container_uri,
  accel_config,
  job_id,
  bucket_name,
  num_workers,
  namespace,
  version=LWS_VERSION,
):
  """Create a LeaderWorkerSet manifest."""

  env_vars = [
    {"name": "KERAS_BACKEND", "value": "jax"},
    {
      "name": "JAX_PLATFORMS",
      "value": accel_config.get("jax_platform", "cpu"),
    },
    {"name": "JOB_ID", "value": job_id},
    {"name": "GCS_BUCKET", "value": bucket_name},
    {
      "name": "MEGASCALE_COORDINATOR_ADDRESS",
      "value": "$(LWS_LEADER_ADDRESS)",
    },
    {"name": "MEGASCALE_NUM_SLICES", "value": str(num_workers + 1)},
    {"name": "TPU_WORKER_ID", "value": "$(LWS_WORKER_INDEX)"},
  ]

  tolerations = []
  for t in accel_config["tolerations"]:
    entry = {"key": t["key"], "operator": t["operator"], "effect": t["effect"]}
    if "value" in t:
      entry["value"] = t["value"]
    tolerations.append(entry)

  pod_template = {
    "metadata": {
      "labels": {
        "app": "kinetic-pathways",
        "job-id": job_id,
        "job-name": job_name,
      }
    },
    "spec": {
      "containers": [
        {
          "name": "kinetic-worker",
          "image": container_uri,
          "command": ["python3", "-u", "/app/remote_runner.py"],
          "args": [
            f"gs://{bucket_name}/{job_id}/context.zip",
            f"gs://{bucket_name}/{job_id}/payload.pkl",
            f"gs://{bucket_name}/{job_id}/result.pkl",
          ],
          "env": env_vars,
          "resources": {
            "limits": {
              k: str(v) for k, v in accel_config["resource_limits"].items()
            },
            "requests": {
              k: str(v) for k, v in accel_config["resource_requests"].items()
            },
          },
        }
      ],
    },
  }

  if tolerations:
    pod_template["spec"]["tolerations"] = tolerations

  if accel_config.get("node_selector"):
    pod_template["spec"]["nodeSelector"] = accel_config["node_selector"]

  return {
    "apiVersion": f"{LWS_GROUP}/{version}",
    "kind": "LeaderWorkerSet",
    "metadata": {
      "name": job_name,
      "namespace": namespace,
      "labels": {"app": "kinetic-pathways", "job-id": job_id},
    },
    "spec": {
      "replicas": 1,
      "leaderWorkerTemplate": {
        "size": num_workers + 1,  # 1 leader + N workers
        "restartPolicy": "RecreateGroupOnPodRestart",
        "leaderTemplate": pod_template,
        "workerTemplate": pod_template,
      },
    },
  }


def _get_leader_pod_name(job_name: str) -> str:
  """Get the leader pod name for a LeaderWorkerSet."""
  return f"{job_name}-0"
