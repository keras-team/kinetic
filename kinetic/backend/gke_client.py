"""GKE job submission for kinetic."""

import functools
import json
import subprocess
import time
from contextlib import suppress

from absl import logging
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from kinetic.backend.log_streaming import LogStreamer
from kinetic.core import accelerators
from kinetic.core.accelerators import TpuConfig
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

  Returns:
      kubernetes.client.V1Job object
  """
  # Load kubeconfig
  _load_kube_config()

  # Parse accelerator configuration
  accel_config = _parse_accelerator(accelerator, spot=spot)

  # Create job specification
  job_name = f"kinetic-{job_id}"
  job = _create_job_spec(
    job_name=job_name,
    container_uri=container_uri,
    accel_config=accel_config,
    job_id=job_id,
    bucket_name=bucket_name,
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
      return
    else:
      logging.warning("Failed to delete job %s: %s", job_name, e.reason)
      return

  # Foreground deletion is async; poll until the resource is gone.
  max_attempts = max(1, timeout // poll_interval)
  for _ in range(max_attempts):
    if not job_exists(job_name, namespace):
      return
    time.sleep(poll_interval)
  logging.warning("Timed out waiting for job %s to be deleted", job_name)


def job_exists(job_name, namespace="default") -> bool:
  """Return whether a namespaced GKE Job currently exists."""
  _load_kube_config()
  batch_v1 = client.BatchV1Api()
  try:
    batch_v1.read_namespaced_job_status(job_name, namespace)
    return True
  except ApiException as e:
    if e.status == 404:
      return False
    raise RuntimeError(f"Failed to read job status: {e.reason}") from e


def get_job_status(job_name, namespace="default") -> JobStatus:
  """Return the current job status for async observation APIs."""
  _load_kube_config()
  batch_v1 = client.BatchV1Api()
  core_v1 = client.CoreV1Api()

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
  _load_kube_config()
  core_v1 = client.CoreV1Api()
  pod = _select_job_pod(core_v1, job_name, namespace)
  if pod is None:
    return None
  return pod.metadata.name


def get_job_logs(
  job_name, namespace="default", tail_lines: int | None = None
) -> str:
  """Return logs for the active pod of a GKE Job."""
  _load_kube_config()
  core_v1 = client.CoreV1Api()
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
  _load_kube_config()
  batch_v1 = client.BatchV1Api()
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


def _parse_accelerator(accelerator, spot=False):
  """Convert accelerator string to GKE pod spec fields."""
  parsed = accelerators.parse_accelerator(accelerator, spot=spot)

  if parsed is None:
    return {
      "node_selector": {},
      "resource_limits": {},
      "resource_requests": {},
      "tolerations": [],
      "jax_platform": "cpu",
    }

  if isinstance(parsed, TpuConfig):
    # For TPU Podslices (multi-node), resource requests must be per-node.
    # num_nodes is 1 for single-host TPUs (v3-8, v4-8, v5litepod-1/4/8).
    chips_per_node = parsed.chips // parsed.num_nodes
    config = {
      "node_selector": {
        "cloud.google.com/gke-tpu-accelerator": parsed.gke_accelerator,
        "cloud.google.com/gke-tpu-topology": parsed.topology,
      },
      "resource_limits": {"google.com/tpu": str(chips_per_node)},
      "resource_requests": {"google.com/tpu": str(chips_per_node)},
      "tolerations": [
        {"key": "google.com/tpu", "operator": "Exists", "effect": "NoSchedule"}
      ],
      "jax_platform": "tpu",
    }

    if parsed.spot:
      config["node_selector"]["cloud.google.com/gke-spot"] = "true"
      config["tolerations"].append(
        {
          "key": "cloud.google.com/gke-spot",
          "operator": "Equal",
          "value": "true",
          "effect": "NoSchedule",
        }
      )
    return config

  # GpuConfig
  config = {
    "node_selector": {"cloud.google.com/gke-accelerator": parsed.gke_label},
    "resource_limits": {"nvidia.com/gpu": str(parsed.count)},
    "resource_requests": {"nvidia.com/gpu": str(parsed.count)},
    "tolerations": [
      {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
    ],
    "jax_platform": "gpu",
  }
  if parsed.spot:
    config["node_selector"]["cloud.google.com/gke-spot"] = "true"
    config["tolerations"].append(
      {
        "key": "cloud.google.com/gke-spot",
        "operator": "Equal",
        "value": "true",
        "effect": "NoSchedule",
      }
    )
  return config


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
  job_name, container_uri, accel_config, job_id, bucket_name, namespace
):
  """Create Kubernetes Job specification.

  Args:
      job_name: Name for the K8s Job
      container_uri: Docker image URI
      accel_config: Accelerator configuration from _parse_accelerator_for_gke
      job_id: Unique job identifier
      bucket_name: GCS bucket for artifacts
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
    name="kinetic-worker",
    image=container_uri,
    command=["python3", "-u", "/app/remote_runner.py"],
    args=[
      f"gs://{bucket_name}/{job_id}/context.zip",
      f"gs://{bucket_name}/{job_id}/payload.pkl",
      f"gs://{bucket_name}/{job_id}/result.pkl",
    ],
    env=env_vars,
    resources=client.V1ResourceRequirements(
      limits={k: str(v) for k, v in accel_config["resource_limits"].items()},
      requests={
        k: str(v) for k, v in accel_config["resource_requests"].items()
      },
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

  pod_template = client.V1PodTemplateSpec(
    metadata=client.V1ObjectMeta(
      labels={"app": "kinetic", "job-id": job_id, "job-name": job_name}
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


def _print_pod_logs(core_v1, job_name, namespace):
  """Print pod logs for debugging failed jobs."""
  with suppress(ApiException):
    for pod in _list_job_pods(core_v1, job_name, namespace):
      with suppress(ApiException):
        logs = core_v1.read_namespaced_pod_log(
          pod.metadata.name, namespace, tail_lines=100
        )
        logging.info("Pod %s logs:\n%s", pod.metadata.name, logs)


def _list_job_pods(core_v1, job_name, namespace):
  pods = core_v1.list_namespaced_pod(
    namespace, label_selector=f"job-name={job_name}"
  )
  return pods.items


def _select_job_pod(core_v1, job_name, namespace):
  pods = _list_job_pods(core_v1, job_name, namespace)
  for phase in ("Running", "Pending"):
    for pod in pods:
      if pod.status.phase == phase:
        return pod
  if not pods:
    return None
  return pods[0]


@functools.lru_cache(maxsize=16)
def _check_node_pool_exists_cached(selector_items) -> bool:
  """Use gcloud to verify that a GKE NodePool matches the pod node selector.

  Note: This caches results for the process lifetime. If a user creates a new
  node pool in another terminal (e.g. `kinetic pool add`) during a long-running
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

      # Spot VM mapping
      if config_dict.get("spot"):
        pool_labels["cloud.google.com/gke-spot"] = "true"

      # Map GKE injected node labels for accelerators mapping
      accel_config_list = config_dict.get("accelerators", [])
      if accel_config_list:
        accel_type = accel_config_list[0].get("acceleratorType", "")
        if accel_type.startswith("tpu-"):
          pool_labels["cloud.google.com/gke-tpu-accelerator"] = accel_type
        else:
          pool_labels["cloud.google.com/gke-accelerator"] = accel_type

      # TPU topology mapping from placement policy
      placement_policy = pool.get("placementPolicy", {})
      if placement_policy and placement_policy.get("tpuTopology"):
        pool_labels["cloud.google.com/gke-tpu-topology"] = placement_policy[
          "tpuTopology"
        ]

      # TPU mapping fallback
      machine_type = config_dict.get("machineType", "")

      # Check resource labels for TPU type (common in v5e/v5litepod)
      resource_labels = config_dict.get("resourceLabels", {})
      if "goog-gke-accelerator-type" in resource_labels:
        pool_labels["cloud.google.com/gke-tpu-accelerator"] = resource_labels[
          "goog-gke-accelerator-type"
        ]

      if machine_type.startswith("ct") and not pool_labels.get(
        "cloud.google.com/gke-tpu-topology"
      ):
        # We roughly map TPU topology presence for preflight
        pool_labels["cloud.google.com/gke-tpu-topology"] = selector.get(
          "cloud.google.com/gke-tpu-topology", ""
        )

      # Infer accelerator count from machine type using registry
      # This is robust because it uses the same source of truth as the Pod spec generation
      for tpu_spec in accelerators.TPUS.values():
        for chips, topo_spec in tpu_spec.topologies.items():
          if topo_spec.machine_type == machine_type:
            pool_labels["cloud.google.com/gke-accelerator-count"] = str(
              chips // topo_spec.num_nodes
            )
            break

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
                  "Please use 'kinetic pool add' to configure this accelerator."
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
                  "  Waiting for nodes to become available (this may take a few minutes for new pools or scale-up)\n"
                  "  Note: If this hangs indefinitely, ensure your GCP project has adequate quota.",
                  pod_name,
                  msg.split(". ")[0],
                  selector_str,
                )
                logged_pending.add(pod_name)
