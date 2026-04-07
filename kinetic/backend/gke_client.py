"""GKE job submission for kinetic."""

import functools
import time
from contextlib import suppress

from absl import logging
from kubernetes import client
from kubernetes.client.rest import ApiException

from kinetic.backend import k8s_utils
from kinetic.backend.log_streaming import LogStreamer
from kinetic.cli.constants import KINETIC_KSA_NAME
from kinetic.credentials import invalidate_credential_cache
from kinetic.job_status import JobStatus


def submit_k8s_job(
  display_name,
  container_uri,
  accelerator,
  project,
  job_id,
  bucket_name,
  namespace="default",
  spot=False,
  requirements_uri=None,
  fuse_volume_specs=None,
):
  """Submit a Kubernetes Job to GKE cluster.

  Args:
      display_name: Job display name (used for K8s job name)
      container_uri: Docker container image URI
      accelerator: GPU type (e.g., 'l4', 'a100', 'nvidia-l4')
      project: GCP project ID
      job_id: Unique job identifier
      bucket_name: GCS bucket name for artifacts
      namespace: Kubernetes namespace (default: "default")
      requirements_uri: Optional GCS URI to requirements.txt for runtime
          install (prebuilt image mode).

  Returns:
      kubernetes.client.V1Job object
  """
  # Parse accelerator configuration
  accel_config = k8s_utils.parse_accelerator(accelerator, spot=spot)

  # Create job specification
  job_name = f"kinetic-{job_id}"
  job = _create_job_spec(
    job_name=job_name,
    container_uri=container_uri,
    accel_config=accel_config,
    job_id=job_id,
    bucket_name=bucket_name,
    namespace=namespace,
    requirements_uri=requirements_uri,
    fuse_volume_specs=fuse_volume_specs,
  )

  # Submit job
  batch_v1 = _batch_v1()

  try:
    created_job = batch_v1.create_namespaced_job(namespace=namespace, body=job)
    logging.info("Submitted K8s job: %s", job_name)
    logging.info("View job with: kubectl get job %s -n %s", job_name, namespace)
    logging.info(
      "View logs with: kubectl logs -l job-name=%s -n %s", job_name, namespace
    )
    return created_job
  except ApiException as e:
    if e.status in (401, 403):
      invalidate_credential_cache()
      raise RuntimeError(
        f"Permission denied creating K8s Job. Ensure your kubeconfig "
        f"has 'create' permission for Jobs in namespace '{namespace}'. "
        f"Run: kubectl auth can-i create jobs -n {namespace}"
      ) from e
    elif e.status == 404:
      raise RuntimeError(
        f"Namespace '{namespace}' not found. Create it with: "
        f"kubectl create namespace {namespace}"
      ) from e
    elif e.status == 409:
      raise RuntimeError(
        f"Job '{job_name}' already exists. "
        f"Clean up with: kubectl delete job {job_name} -n {namespace}"
      ) from e
    else:
      raise RuntimeError(
        f"Kubernetes API error: {e.status} - {e.reason}: {e.body}"
      ) from e


def wait_for_job(job, namespace="default", timeout=3600, poll_interval=10):
  """Wait for Kubernetes Job to complete.

  Args:
      job: Kubernetes Job object
      namespace: Kubernetes namespace
      timeout: Maximum time to wait in seconds (default: 1 hour)
      poll_interval: Time between status checks in seconds

  Returns:
      Job status: 'success'

  Raises:
      RuntimeError: If job fails or times out
  """
  batch_v1 = _batch_v1()
  core_v1 = k8s_utils.core_v1()

  job_name = job.metadata.name
  start_time = time.time()
  logged_running = False
  logged_pending = set()

  with LogStreamer(core_v1, namespace) as streamer:
    while True:
      # Check timeout
      elapsed = time.time() - start_time
      if elapsed > timeout:
        raise RuntimeError(f"GKE job {job_name} timed out after {timeout}s")

      # Get job status
      try:
        job_status = batch_v1.read_namespaced_job_status(job_name, namespace)
      except ApiException as e:
        raise RuntimeError(f"Failed to read job status: {e.reason}") from e

      # Check completion conditions
      if job_status.status.succeeded and job_status.status.succeeded >= 1:
        logging.info("Job %s completed successfully", job_name)
        return "success"

      if job_status.status.failed and job_status.status.failed >= 1:
        # Get pod logs for debugging
        k8s_utils.print_pod_logs(core_v1, job_name, namespace)
        raise RuntimeError(f"GKE job {job_name} failed")

      # Check for pod scheduling issues
      k8s_utils.check_pod_scheduling(
        core_v1, job_name, namespace, logged_pending
      )

      # Start log streaming when pod is running
      with suppress(ApiException):
        pods = core_v1.list_namespaced_pod(
          namespace, label_selector=f"job-name={job_name}"
        )
        for pod in pods.items:
          if pod.status.phase == "Running":
            streamer.start(pod.metadata.name)
            break

      # Job still running
      if not logged_running:
        logging.info("Job %s running...", job_name)
        logged_running = True

      time.sleep(poll_interval)


def cleanup_job(
  job_name, namespace="default", timeout: float = 180, poll_interval: float = 2
):
  """Delete completed Kubernetes Job and its pods.

  Blocks until the API confirms the resource is gone (404).

  Args:
      job_name: Name of the Kubernetes Job
      namespace: Kubernetes namespace
      timeout: Maximum seconds to wait for deletion (default 180)
      poll_interval: Seconds between existence checks (default 2)
  """
  batch_v1 = _batch_v1()

  try:
    # Delete job with propagation policy to also delete pods
    batch_v1.delete_namespaced_job(
      name=job_name,
      namespace=namespace,
      body=client.V1DeleteOptions(propagation_policy="Foreground"),
    )
    logging.info("Deleted K8s job: %s", job_name)
  except ApiException as e:
    if e.status == 404:
      # Job already deleted
      return
    else:
      logging.warning("Failed to delete job %s: %s", job_name, e.reason)
      return

  # Foreground deletion is async; poll until the resource is gone.
  max_attempts = int(max(1, timeout // poll_interval))
  for _ in range(max_attempts):
    if not job_exists(job_name, namespace):
      return
    time.sleep(poll_interval)
  logging.warning("Timed out waiting for job %s to be deleted", job_name)


def job_exists(job_name, namespace="default") -> bool:
  """Return whether a namespaced GKE Job currently exists."""
  batch_v1 = _batch_v1()
  try:
    batch_v1.read_namespaced_job_status(job_name, namespace)
    return True
  except ApiException as e:
    if e.status == 404:
      return False
    raise RuntimeError(f"Failed to read job status: {e.reason}") from e


def get_job_status(job_name, namespace="default") -> JobStatus:
  """Return the current job status for async observation APIs."""
  batch_v1 = _batch_v1()
  core_v1 = k8s_utils.core_v1()

  try:
    job_status = batch_v1.read_namespaced_job_status(job_name, namespace)
  except ApiException as e:
    if e.status == 404:
      return JobStatus.NOT_FOUND
    raise RuntimeError(f"Failed to read job status: {e.reason}") from e

  if job_status.status.succeeded and job_status.status.succeeded >= 1:
    return JobStatus.SUCCEEDED
  if job_status.status.failed and job_status.status.failed >= 1:
    return JobStatus.FAILED

  pod = _select_job_pod(core_v1, job_name, namespace)

  if pod is not None and pod.status.phase == "Running":
    return JobStatus.RUNNING
  return JobStatus.PENDING


def get_job_pod_name(job_name, namespace="default") -> str | None:
  """Return the most relevant pod name for a GKE Job, if any exists."""
  core_v1 = k8s_utils.core_v1()
  pod = _select_job_pod(core_v1, job_name, namespace)
  if pod is None:
    return None
  return pod.metadata.name


def get_job_logs(
  job_name, namespace="default", tail_lines: int | None = None
) -> str:
  """Return logs for the active pod of a GKE Job."""
  core_v1 = k8s_utils.core_v1()
  pod = _select_job_pod(core_v1, job_name, namespace)
  if pod is None:
    raise RuntimeError(f"No pod found for GKE job {job_name}")

  log_kwargs = {}
  if tail_lines is not None:
    log_kwargs["tail_lines"] = tail_lines
  return core_v1.read_namespaced_pod_log(
    pod.metadata.name,
    namespace,
    **log_kwargs,
  )


def list_jobs(namespace="default") -> list[dict[str, str]]:
  """List live GKE Jobs managed by Kinetic in a namespace."""
  batch_v1 = _batch_v1()
  jobs = batch_v1.list_namespaced_job(
    namespace=namespace,
    label_selector="app=kinetic",
  )

  results = []
  for job in jobs.items:
    labels = job.metadata.labels or {}
    job_id = labels.get("job-id")
    if job_id is None:
      continue
    results.append(
      {
        "job_id": job_id,
        "k8s_name": job.metadata.name,
      }
    )
  return results


@functools.lru_cache(maxsize=1)
def _batch_v1():
  """Return a cached BatchV1Api client, loading kubeconfig on first call."""
  k8s_utils.load_kube_config()
  return client.BatchV1Api()


def _create_job_spec(
  job_name,
  container_uri,
  accel_config,
  job_id,
  bucket_name,
  namespace,
  requirements_uri=None,
  fuse_volume_specs=None,
):
  """Create Kubernetes Job specification.

  Args:
      job_name: Name for the K8s Job
      container_uri: Docker image URI
      accel_config: Accelerator configuration from _parse_accelerator_for_gke
      job_id: Unique job identifier
      bucket_name: GCS bucket for artifacts
      namespace: Kubernetes namespace
      requirements_uri: Optional GCS URI to requirements.txt for runtime
          install (prebuilt image mode).

  Returns:
      V1Job object ready for creation
  """
  # Environment variables for remote_runner.py
  env_vars = [
    client.V1EnvVar(name="KERAS_BACKEND", value="jax"),
    client.V1EnvVar(
      name="JAX_PLATFORMS", value=accel_config.get("jax_platform", "gpu")
    ),
    client.V1EnvVar(name="JOB_ID", value=job_id),
    client.V1EnvVar(name="GCS_BUCKET", value=bucket_name),
  ]

  # Container arguments: context, payload, result, [requirements]
  container_args = [
    f"gs://{bucket_name}/{job_id}/context.zip",
    f"gs://{bucket_name}/{job_id}/payload.pkl",
    f"gs://{bucket_name}/{job_id}/result.pkl",
  ]
  if requirements_uri:
    container_args.append(requirements_uri)

  # Container specification
  container = client.V1Container(
    name="kinetic-worker",
    image=container_uri,
    command=["python3", "-u", "/app/remote_runner.py"],
    args=container_args,
    env=env_vars,
    resources=client.V1ResourceRequirements(
      limits={k: str(v) for k, v in accel_config["resource_limits"].items()},
      requests={
        k: str(v) for k, v in accel_config["resource_requests"].items()
      },
    ),
  )

  # GCS FUSE CSI volumes (lazy-mounted from GCS via the CSI driver).
  fuse_annotations, fuse_volumes, fuse_mounts = (
    k8s_utils.build_gcs_fuse_v1_volumes(fuse_volume_specs)
  )
  if fuse_annotations:
    if container.volume_mounts is None:
      container.volume_mounts = []
    container.volume_mounts.extend(fuse_mounts)

  # Build tolerations
  tolerations = [
    client.V1Toleration(
      key=t["key"],
      operator=t["operator"],
      effect=t["effect"],
    )
    for t in accel_config["tolerations"]
  ]

  # Pod template specification
  pod_spec_kwargs = {
    "containers": [container],
    "tolerations": tolerations if tolerations else None,
    "restart_policy": "Never",
    "service_account_name": KINETIC_KSA_NAME,
  }
  # Only set node_selector if non-empty (for GPU nodes)
  if accel_config.get("node_selector"):
    pod_spec_kwargs["node_selector"] = accel_config["node_selector"]
  if fuse_volumes:
    pod_spec_kwargs.setdefault("volumes", []).extend(fuse_volumes)

  pod_template = client.V1PodTemplateSpec(
    metadata=client.V1ObjectMeta(
      labels={"app": "kinetic", "job-id": job_id, "job-name": job_name},
      annotations=fuse_annotations,
    ),
    spec=client.V1PodSpec(**pod_spec_kwargs),
  )

  # Job specification
  job_spec = client.V1JobSpec(
    template=pod_template,
    backoff_limit=0,  # No retries - fail immediately
    ttl_seconds_after_finished=600,  # Auto-cleanup after 10 minutes
  )

  # Complete Job object
  job = client.V1Job(
    api_version="batch/v1",
    kind="Job",
    metadata=client.V1ObjectMeta(
      name=job_name,
      namespace=namespace,
      labels={"app": "kinetic", "job-id": job_id},
    ),
    spec=job_spec,
  )

  return job


def _select_job_pod(core_v1_client, job_name, namespace):
  pods = k8s_utils.list_job_pods(core_v1_client, job_name, namespace)
  for phase in ("Running", "Pending"):
    for pod in pods:
      if pod.status.phase == phase:
        return pod
  if not pods:
    return None
  return pods[0]
