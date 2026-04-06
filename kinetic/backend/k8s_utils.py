"""Shared Kubernetes utilities used by both GKE and Pathways backends."""

import functools
from contextlib import suppress

from absl import logging
from google.cloud import container_v1
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from kinetic.core import accelerators
from kinetic.core.accelerators import TpuConfig

# GKE node selector / resource label keys.
_LABEL_TPU_ACCELERATOR = "cloud.google.com/gke-tpu-accelerator"
_LABEL_TPU_TOPOLOGY = "cloud.google.com/gke-tpu-topology"
_LABEL_GPU_ACCELERATOR = "cloud.google.com/gke-accelerator"
_LABEL_ACCELERATOR_COUNT = "cloud.google.com/gke-accelerator-count"
_LABEL_SPOT = "cloud.google.com/gke-spot"
_RESOURCE_TPU = "google.com/tpu"
_RESOURCE_GPU = "nvidia.com/gpu"
_RESOURCE_LABEL_TPU_TYPE = "goog-gke-accelerator-type"


def parse_accelerator(accelerator, spot=False):
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
    accel_config = {
      "node_selector": {
        _LABEL_TPU_ACCELERATOR: parsed.gke_accelerator,
        _LABEL_TPU_TOPOLOGY: parsed.topology,
      },
      "resource_limits": {_RESOURCE_TPU: str(chips_per_node)},
      "resource_requests": {_RESOURCE_TPU: str(chips_per_node)},
      "tolerations": [
        {"key": _RESOURCE_TPU, "operator": "Exists", "effect": "NoSchedule"}
      ],
      "jax_platform": "tpu",
    }
  else:
    # GpuConfig
    accel_config = {
      "node_selector": {_LABEL_GPU_ACCELERATOR: parsed.gke_label},
      "resource_limits": {_RESOURCE_GPU: str(parsed.count)},
      "resource_requests": {_RESOURCE_GPU: str(parsed.count)},
      "tolerations": [
        {"key": _RESOURCE_GPU, "operator": "Exists", "effect": "NoSchedule"}
      ],
      "jax_platform": "gpu",
    }

  if parsed.spot:
    accel_config["node_selector"][_LABEL_SPOT] = "true"
    accel_config["tolerations"].append(
      {
        "key": _LABEL_SPOT,
        "operator": "Equal",
        "value": "true",
        "effect": "NoSchedule",
      }
    )
  return accel_config


@functools.lru_cache(maxsize=1)
def load_kube_config():
  """Load Kubernetes configuration (one-shot, cached).

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


@functools.lru_cache(maxsize=1)
def core_v1():
  """Return a cached CoreV1Api client, loading kubeconfig on first call."""
  load_kube_config()
  return client.CoreV1Api()


def list_job_pods(core_v1_client, job_name, namespace):
  pods = core_v1_client.list_namespaced_pod(
    namespace, label_selector=f"job-name={job_name}"
  )
  return pods.items


def print_pod_logs(core_v1_client, job_name, namespace):
  """Print pod logs for debugging failed jobs."""
  with suppress(ApiException):
    for pod in list_job_pods(core_v1_client, job_name, namespace):
      with suppress(ApiException):
        logs = core_v1_client.read_namespaced_pod_log(
          pod.metadata.name, namespace, tail_lines=100
        )
        logging.info("Pod %s logs:\n%s", pod.metadata.name, logs)


def _get_cluster_info() -> tuple[str, str, str] | None:
  """Extract project, location, and cluster name from kubeconfig context.

  Returns:
      (project, location, cluster) tuple, or None if not determinable.
  """
  try:
    _, active_context = config.kube_config.list_kube_config_contexts()
    context_name = active_context.get("name", "")
    if context_name.startswith("gke_"):
      parts = context_name.split("_")
      if len(parts) >= 4:
        return parts[1], parts[2], "_".join(parts[3:])
  except (config.ConfigException, KeyError, TypeError) as e:
    logging.warning(
      "Could not determine cluster context from kubeconfig: %s", e
    )
  return None


def _build_pool_labels(pool: container_v1.NodePool, selector: dict) -> dict:
  """Build a synthetic label dict from a NodePool for matching against a pod selector."""
  pool_config = pool.config
  pool_labels = dict(pool_config.labels) if pool_config.labels else {}

  # Spot VM mapping
  if pool_config.spot:
    pool_labels[_LABEL_SPOT] = "true"

  # Map accelerator type from pool config
  if pool_config.accelerators:
    accel_type = pool_config.accelerators[0].accelerator_type
    if accel_type.startswith("tpu-"):
      pool_labels[_LABEL_TPU_ACCELERATOR] = accel_type
    else:
      pool_labels[_LABEL_GPU_ACCELERATOR] = accel_type

  # TPU topology from placement policy
  tpu_topology = pool.placement_policy.tpu_topology
  if tpu_topology:
    pool_labels[_LABEL_TPU_TOPOLOGY] = tpu_topology

  machine_type = pool_config.machine_type or ""

  # Check resource labels for TPU type (common in v5e/v5litepod)
  resource_labels = (
    dict(pool_config.resource_labels) if pool_config.resource_labels else {}
  )
  if _RESOURCE_LABEL_TPU_TYPE in resource_labels:
    pool_labels[_LABEL_TPU_ACCELERATOR] = resource_labels[
      _RESOURCE_LABEL_TPU_TYPE
    ]

  if machine_type.startswith("ct") and not pool_labels.get(_LABEL_TPU_TOPOLOGY):
    # We roughly map TPU topology presence for preflight
    pool_labels[_LABEL_TPU_TOPOLOGY] = selector.get(_LABEL_TPU_TOPOLOGY, "")

  # Infer accelerator count from machine type using registry.
  # This is robust because it uses the same source of truth as the Pod spec generation.
  for tpu_spec in accelerators.TPUS.values():
    for chips, topo_spec in tpu_spec.topologies.items():
      if topo_spec.machine_type == machine_type:
        pool_labels[_LABEL_ACCELERATOR_COUNT] = str(
          chips // topo_spec.num_nodes
        )
        break

  return pool_labels


@functools.lru_cache(maxsize=16)
def _check_node_pool_exists_cached(selector_items) -> bool:
  """Verify that a GKE NodePool matches the pod node selector.

  Uses the google-cloud-container client library to list node pools.

  Note: This caches results for the process lifetime. If a user creates a new
  node pool in another terminal (e.g. `kinetic pool add`) during a long-running
  session, this may return stale results. This is acceptable for our current
  scale-to-zero model with ephemeral sessions.
  """
  selector = dict(selector_items)
  try:
    cluster_info = _get_cluster_info()
    if cluster_info is None:
      logging.warning(
        "Could not determine GKE cluster from kubeconfig. "
        "Skipping node pool validation."
      )
      return True

    project, location, cluster_name = cluster_info
    gke_client = container_v1.ClusterManagerClient()
    parent = f"projects/{project}/locations/{location}/clusters/{cluster_name}"
    response = gke_client.list_node_pools(parent=parent)

    for pool in response.node_pools:
      pool_labels = _build_pool_labels(pool, selector)
      if all(pool_labels.get(k) == str(v) for k, v in selector.items()):
        return True
    return False
  except Exception as e:
    # Degrade gracefully, but inform the user that the check failed.
    logging.warning(
      "Could not verify node pool existence. "
      "Proceeding with assumption that it exists. Error: %s",
      e,
    )
    return True


def _validate_node_pool_exists(selector: dict) -> bool:
  if not selector:
    return True
  return _check_node_pool_exists_cached(tuple(sorted(selector.items())))


def validate_preflight(accelerator):
  """Check if any nodes match the required accelerator selector.

  Args:
      accelerator: Accelerator string (e.g., 'l4', 'v3-8')
  """
  accel_config = parse_accelerator(accelerator)
  node_selector = accel_config.get("node_selector")

  if not node_selector:
    return  # CPU or no selector required

  core_v1_client = core_v1()
  try:
    # Construct label selector string: "key1=val1,key2=val2"
    label_selector = ",".join([f"{k}={v}" for k, v in node_selector.items()])
    nodes = core_v1_client.list_node(label_selector=label_selector)

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


_IMAGE_PULL_ERROR_REASONS = frozenset(
  {
    "ImagePullBackOff",
    "ErrImagePull",
    "ErrImageNeverPull",
    "InvalidImageName",
  }
)


def _check_image_pull_errors(pod) -> None:
  """Raise a clear error if any container is stuck on an image pull failure."""
  all_statuses = list(pod.status.init_container_statuses or []) + list(
    pod.status.container_statuses or []
  )
  for cs in all_statuses:
    waiting = cs.state.waiting if cs.state else None
    if waiting and waiting.reason in _IMAGE_PULL_ERROR_REASONS:
      image = cs.image or "unknown"
      detail = waiting.message or waiting.reason
      raise RuntimeError(
        f"Container image pull failed for '{image}': {detail}\n"
        f"If using prebuilt images, either:\n"
        f"  1. If using custom prebuilt images, ensure you have built and\n"
        f"     pushed the image (kinetic build-base --repo <repo>) and set\n"
        f"     KINETIC_BASE_IMAGE_REPO=<repo>.\n"
        f"  2. Use bundled mode: @kinetic.run(..., container_image='bundled')\n"
        f"  3. If using official kinetic images, report the issue at:\n"
        f"     https://github.com/keras-team/kinetic/issues"
      )


def check_pod_scheduling(core_v1_client, job_name, namespace, logged_pending):
  """Check for pod scheduling and image pull issues, raising helpful errors."""
  with suppress(ApiException):
    pods = core_v1_client.list_namespaced_pod(
      namespace, label_selector=f"job-name={job_name}"
    )
    for pod in pods.items:
      pod_name = pod.metadata.name
      # Check for image pull failures (can occur in any phase).
      _check_image_pull_errors(pod)

      if pod.status.phase == "Pending":
        for condition in pod.status.conditions or []:
          if condition.type == "PodScheduled" and condition.status == "False":
            msg = condition.message or ""

            is_insufficient = (
              f"Insufficient {_RESOURCE_GPU}" in msg
              or f"Insufficient {_RESOURCE_TPU}" in msg
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
