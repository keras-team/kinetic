"""GKE job submission for keras_remote."""

import functools
import json
import subprocess
import time
from contextlib import suppress

from absl import logging
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from keras_remote.backend.log_streaming import LogStreamer
from keras_remote.core import accelerators
from keras_remote.core.accelerators import TpuConfig


def submit_k8s_job(
  display_name,
  container_uri,
  accelerator,
  project,
  job_id,
  bucket_name,
  gcs_prefix=None,
  namespace="default",
):
  """Submit a Kubernetes Job to GKE cluster.

  Args:
      display_name: Job display name (used for K8s job name)
      container_uri: Docker container image URI
      accelerator: GPU type (e.g., 'l4', 'a100', 'nvidia-l4')
      project: GCP project ID
      job_id: Unique job identifier
      bucket_name: GCS bucket name for artifacts
      gcs_prefix: Namespace-scoped GCS prefix (e.g. "default/job-abc").
          Defaults to job_id for backward compatibility.
      namespace: Kubernetes namespace (default: "default")

  Returns:
      kubernetes.client.V1Job object
  """
  if gcs_prefix is None:
    gcs_prefix = job_id

  # Load kubeconfig
  _load_kube_config()

  # Parse accelerator configuration
  accel_config = _parse_accelerator(accelerator)

  # Create job specification
  job_name = f"keras-remote-{job_id}"
  job = _create_job_spec(
    job_name=job_name,
    container_uri=container_uri,
    accel_config=accel_config,
    job_id=job_id,
    bucket_name=bucket_name,
    gcs_prefix=gcs_prefix,
    namespace=namespace,
  )

  # Submit job
  batch_v1 = client.BatchV1Api()

  try:
    created_job = batch_v1.create_namespaced_job(namespace=namespace, body=job)
    logging.info("Submitted K8s job: %s", job_name)
    logging.info("View job with: kubectl get job %s -n %s", job_name, namespace)
    logging.info(
      "View logs with: kubectl logs -l job-name=%s -n %s", job_name, namespace
    )
    return created_job
  except ApiException as e:
    if e.status == 403:
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
  _load_kube_config()
  batch_v1 = client.BatchV1Api()
  core_v1 = client.CoreV1Api()

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
        _print_pod_logs(core_v1, job_name, namespace)
        raise RuntimeError(f"GKE job {job_name} failed")

      # Check for pod scheduling issues
      _check_pod_scheduling(core_v1, job_name, namespace, logged_pending)

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


def cleanup_job(job_name, namespace="default"):
  """Delete completed Kubernetes Job and its pods.

  Args:
      job_name: Name of the Kubernetes Job
      namespace: Kubernetes namespace
  """
  _load_kube_config()
  batch_v1 = client.BatchV1Api()

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
      pass
    else:
      logging.warning("Failed to delete job %s: %s", job_name, e.reason)


def validate_preflight(
  accelerator, project, cluster, zone, namespace="default"
):
  """Check if the required node pool exists for the accelerator.

  Args:
      accelerator: Accelerator string (e.g., 'l4', 'v3-8')
      project: GCP project ID
      cluster: GKE cluster name
      zone: GCP zone
      namespace: Kubernetes namespace

  Raises:
      RuntimeError: If no nodes match the required accelerator selector.
  """
  _load_kube_config()
  accel_config = _parse_accelerator(accelerator)
  node_selector = accel_config.get("node_selector")

  if not node_selector:
    return  # CPU or no selector required

  core_v1 = client.CoreV1Api()
  try:
    # Construct label selector string: "key1=val1,key2=val2"
    label_selector = ",".join([f"{k}={v}" for k, v in node_selector.items()])
    nodes = core_v1.list_node(label_selector=label_selector)

    if not nodes.items:
      selector_str = ", ".join([f"{k}: {v}" for k, v in node_selector.items()])
      logging.info(
        "Preflight check: No currently running nodes match selector: %s. "
        "Proceeding under the assumption that the cluster will auto-provision with scale-to-zero enabled.",
        selector_str,
      )
  except ApiException as e:
    # If we can't list nodes due to permissions, log a warning but proceed
    # to avoid blocking users with restricted kubeconfig.
    logging.warning("Preflight check: Failed to query nodes: %s", e.reason)


def check_quota(namespace, accelerator):
  """Check K8s ResourceQuota usage before submitting a job.

  For the default namespace this is a no-op.  For non-default namespaces
  the function reads the ResourceQuota status and compares requested
  resources against hard limits and current usage.

  Raises:
      RuntimeError: If the job would exceed the namespace quota.
  """
  if namespace == "default":
    return

  _load_kube_config()
  core_v1 = client.CoreV1Api()

  try:
    quotas = core_v1.list_namespaced_resource_quota(namespace)
  except ApiException as e:
    if e.status == 403:
      logging.warning(
        "Quota pre-check: permission denied reading ResourceQuota "
        "in namespace '%s'. Proceeding without quota check.",
        namespace,
      )
      return
    raise

  if not quotas.items:
    return  # No quota configured for this namespace

  accel_config = _parse_accelerator(accelerator)

  for quota in quotas.items:
    hard = quota.status.hard or {}
    used = quota.status.used or {}

    # GPU check
    gpu_key = "requests.nvidia.com/gpu"
    requested_gpus = int(
      accel_config["resource_requests"].get("nvidia.com/gpu", "0")
    )
    if gpu_key in hard and requested_gpus > 0:
      limit = int(hard[gpu_key])
      in_use = int(used.get(gpu_key, "0"))
      if in_use + requested_gpus > limit:
        raise RuntimeError(
          f"GPU quota exceeded in namespace '{namespace}': "
          f"{in_use}/{limit} GPUs in use, your job requests {requested_gpus}"
        )

    # TPU check
    tpu_key = "requests.google.com/tpu"
    requested_tpus = int(
      accel_config["resource_requests"].get("google.com/tpu", "0")
    )
    if tpu_key in hard and requested_tpus > 0:
      limit = int(hard[tpu_key])
      in_use = int(used.get(tpu_key, "0"))
      if in_use + requested_tpus > limit:
        raise RuntimeError(
          f"TPU quota exceeded in namespace '{namespace}': "
          f"{in_use}/{limit} TPUs in use, your job requests {requested_tpus}"
        )

    # Job count check
    job_key = "count/jobs.batch"
    if job_key in hard:
      limit = int(hard[job_key])
      in_use = int(used.get(job_key, "0"))
      if in_use >= limit:
        raise RuntimeError(
          f"Job count quota exceeded in namespace '{namespace}': "
          f"{in_use}/{limit} jobs running"
        )


def _parse_accelerator(accelerator):
  """Convert accelerator string to GKE pod spec fields."""
  parsed = accelerators.parse_accelerator(accelerator)

  if parsed is None:
    return {
      "node_selector": {},
      "resource_limits": {},
      "resource_requests": {},
      "tolerations": [],
      "jax_platform": "cpu",
    }

  if isinstance(parsed, TpuConfig):
    return {
      "node_selector": {
        "cloud.google.com/gke-tpu-accelerator": parsed.gke_accelerator,
        "cloud.google.com/gke-tpu-topology": parsed.topology,
      },
      "resource_limits": {"google.com/tpu": str(parsed.chips)},
      "resource_requests": {"google.com/tpu": str(parsed.chips)},
      "tolerations": [
        {"key": "google.com/tpu", "operator": "Exists", "effect": "NoSchedule"}
      ],
      "jax_platform": "tpu",
    }

  # GpuConfig
  return {
    "node_selector": {"cloud.google.com/gke-accelerator": parsed.gke_label},
    "resource_limits": {"nvidia.com/gpu": str(parsed.count)},
    "resource_requests": {"nvidia.com/gpu": str(parsed.count)},
    "tolerations": [
      {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
    ],
    "jax_platform": "gpu",
  }


def _load_kube_config():
  """Load Kubernetes configuration.

  Attempts to load config in order:
  1. In-cluster config (if running inside K8s)
  2. Kubeconfig from KUBECONFIG env or ~/.kube/config

  Raises:
      RuntimeError: If unable to load any configuration
  """
  try:
    # Try in-cluster config first (for running inside K8s)
    config.load_incluster_config()
    return
  except config.ConfigException:
    pass

  try:
    # Fall back to kubeconfig
    config.load_kube_config()
    return
  except config.ConfigException as e:
    raise RuntimeError(
      f"Failed to load Kubernetes configuration. "
      f"Ensure you have run 'gcloud container clusters get-credentials <cluster-name>' "
      f"or have a valid kubeconfig. Error: {e}"
    ) from e


def _create_job_spec(
  job_name,
  container_uri,
  accel_config,
  job_id,
  bucket_name,
  gcs_prefix,
  namespace,
):
  """Create Kubernetes Job specification.

  Args:
      job_name: Name for the K8s Job
      container_uri: Docker image URI
      accel_config: Accelerator configuration from _parse_accelerator_for_gke
      job_id: Unique job identifier
      bucket_name: GCS bucket for artifacts
      gcs_prefix: Namespace-scoped GCS prefix (e.g. "default/job-abc")
      namespace: Kubernetes namespace

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

  # Container specification
  container = client.V1Container(
    name="keras-remote-worker",
    image=container_uri,
    command=["python3", "-u", "/app/remote_runner.py"],
    args=[
      f"gs://{bucket_name}/{gcs_prefix}/context.zip",
      f"gs://{bucket_name}/{gcs_prefix}/payload.pkl",
      f"gs://{bucket_name}/{gcs_prefix}/result.pkl",
    ],
    env=env_vars,
    resources=client.V1ResourceRequirements(
      limits=accel_config["resource_limits"],
      requests=accel_config["resource_requests"],
    ),
  )

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
  }
  # Only set node_selector if non-empty (for GPU nodes)
  if accel_config.get("node_selector"):
    pod_spec_kwargs["node_selector"] = accel_config["node_selector"]
  # Set service account for non-default namespaces (Workload Identity)
  if namespace != "default":
    pod_spec_kwargs["service_account_name"] = f"{namespace}-runner"

  pod_template = client.V1PodTemplateSpec(
    metadata=client.V1ObjectMeta(
      labels={"app": "keras-remote", "job-id": job_id, "job-name": job_name}
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
      labels={"app": "keras-remote", "job-id": job_id},
    ),
    spec=job_spec,
  )

  return job


def _print_pod_logs(core_v1, job_name, namespace):
  """Print pod logs for debugging failed jobs."""
  with suppress(ApiException):
    pods = core_v1.list_namespaced_pod(
      namespace, label_selector=f"job-name={job_name}"
    )

    for pod in pods.items:
      with suppress(ApiException):
        logs = core_v1.read_namespaced_pod_log(
          pod.metadata.name, namespace, tail_lines=100
        )
        logging.info("Pod %s logs:\n%s", pod.metadata.name, logs)


@functools.lru_cache(maxsize=16)
def _check_node_pool_exists_cached(selector_items) -> bool:
  """Use gcloud to verify that a GKE NodePool matches the pod node selector.

  Note: This caches results for the process lifetime. If a user creates a new
  node pool in another terminal (e.g. `keras-remote pool add`) during a long-running
  session, this may return stale results. This is acceptable for our current
  scale-to-zero model with ephemeral sessions.
  """
  selector = dict(selector_items)
  try:
    cmd = ["gcloud", "container", "node-pools", "list", "--format", "json"]

    # Attempt to extract exact cluster context from kubeconfig
    try:
      _, active_context = config.kube_config.list_kube_config_contexts()
      context_name = active_context.get("name", "")
      if context_name.startswith("gke_"):
        parts = context_name.split("_")
        if len(parts) >= 4:
          project = parts[1]
          location = parts[2]
          cluster = "_".join(parts[3:])
          cmd.extend(
            ["--cluster", cluster, "--location", location, "--project", project]
          )
    except Exception as e:
      logging.warning(
        "Could not determine cluster context from kubeconfig: %s", e
      )

    out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    pools = json.loads(out)
    for pool in pools:
      config_dict = pool.get("config", {})
      pool_labels = config_dict.get("labels", {}).copy()

      # Map GKE injected node labels for accelerators mapping
      accelerators = config_dict.get("accelerators", [])
      if accelerators:
        accel_type = accelerators[0].get("acceleratorType", "")
        if accel_type.startswith("tpu-"):
          pool_labels["cloud.google.com/gke-tpu-accelerator"] = accel_type
        else:
          pool_labels["cloud.google.com/gke-accelerator"] = accel_type

      # TPU mapping fallback
      machine_type = config_dict.get("machineType", "")
      if machine_type.startswith("ct"):
        # We roughly map TPU topology presence for preflight
        pool_labels["cloud.google.com/gke-tpu-topology"] = selector.get(
          "cloud.google.com/gke-tpu-topology", ""
        )

      if all(pool_labels.get(k) == str(v) for k, v in selector.items()):
        return True
    return False
  except Exception as e:
    # Degrade gracefully, but inform the user that the check failed.
    logging.warning(
      "Could not verify node pool existence via `gcloud`. "
      "Proceeding with assumption that it exists. Error: %s",
      e,
    )
    return True


def _validate_node_pool_exists(selector: dict) -> bool:
  if not selector:
    return True
  return _check_node_pool_exists_cached(tuple(sorted(selector.items())))


def _check_pod_scheduling(core_v1, job_name, namespace, logged_pending):
  """Check for pod scheduling issues and raise helpful errors."""
  with suppress(ApiException):
    pods = core_v1.list_namespaced_pod(
      namespace, label_selector=f"job-name={job_name}"
    )
    for pod in pods.items:
      pod_name = pod.metadata.name
      if pod.status.phase == "Pending":
        for condition in pod.status.conditions or []:
          if condition.type == "PodScheduled" and condition.status == "False":
            msg = condition.message or ""

            is_insufficient = (
              "Insufficient nvidia.com/gpu" in msg
              or "Insufficient google.com/tpu" in msg
            )
            is_mismatch = (
              "didn't match Pod's node affinity/selector" in msg
              or "node selector" in msg.lower()
            )

            if is_insufficient or is_mismatch:
              selector = pod.spec.node_selector or {}
              if not _validate_node_pool_exists(selector):
                selector_str = (
                  ", ".join([f"{k}: {v}" for k, v in selector.items()])
                  if selector
                  else "None"
                )
                raise RuntimeError(
                  f"No GKE node pool exists with selector '{selector_str}'. "
                  "Please use 'keras-remote pool add' to configure this accelerator."
                )

              if pod_name not in logged_pending:
                selector_str = (
                  ", ".join([f"{k}: {v}" for k, v in selector.items()])
                  if selector
                  else "None"
                )
                logging.info(
                  "Pod %s is Pending: %s.\n"
                  "  Selector: %s\n"
                  "  Waiting for GKE Cluster Autoscaler to provision a new node... (scale-to-zero)\n"
                  "  Note: If this hangs indefinitely, ensure your GCP project has adequate quota.",
                  pod_name,
                  msg.split(". ")[0],
                  selector_str,
                )
                logged_pending.add(pod_name)
