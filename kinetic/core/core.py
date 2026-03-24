import functools
import os

from kinetic.backend.execution import (
  GKEBackend,
  JobContext,
  PathwaysBackend,
  execute_remote,
)
from kinetic.constants import DEFAULT_CLUSTER_NAME
from kinetic.core import accelerators
from kinetic.data import Data


def run(
  accelerator="v6e-8",
  container_image=None,
  zone=None,
  project=None,
  capture_env_vars=None,
  cluster=None,
  backend=None,
  namespace=None,
  volumes=None,
):
  """Execute function on remote TPU/GPU.

  Args:
    accelerator: TPU/GPU type (e.g., 'v3-8', 'v5litepod-4', 'l4', 'a100')
    container_image: Custom container image URI (optional)
    zone: GCP zone (default: from KERAS_REMOTE_ZONE or 'us-central1-a')
    project: GCP project (default: from KERAS_REMOTE_PROJECT)
    capture_env_vars: List of environment variable names or patterns (ending in *)
      to propagate to the remote environment. Defaults to None.
    cluster: GKE cluster name (default: from KERAS_REMOTE_CLUSTER)
    backend: Backend to use ('gke' or 'pathways')
    namespace: Kubernetes namespace (default: None, resolved via
      KERAS_REMOTE_NAMESPACE env var or 'default')
    volumes: Dict mapping absolute mount paths to Data objects, e.g.
      ``{"/data": Data("./dataset/")}``. Data is downloaded to these
      paths on the pod before function execution.
  """
  # Validate volumes
  if volumes is not None:
    if not isinstance(volumes, dict):
      raise TypeError(f"volumes must be a dict, got {type(volumes).__name__}")
    for mount_path, data_obj in volumes.items():
      if not isinstance(mount_path, str) or not mount_path.startswith("/"):
        raise ValueError(
          f"Volume mount path must be an absolute path "
          f"(start with '/'), got: {mount_path!r}"
        )
      if not isinstance(data_obj, Data):
        raise TypeError(
          f"Volume value for {mount_path!r} must be a Data "
          f"instance, got {type(data_obj).__name__}"
        )

  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      # Capture environment variables
      env_vars = {}
      if capture_env_vars:
        for pattern in capture_env_vars:
          if pattern.endswith("*"):
            prefix = pattern[:-1]
            env_vars.update(
              {k: v for k, v in os.environ.items() if k.startswith(prefix)}
            )
          elif pattern in os.environ:
            env_vars[pattern] = os.environ[pattern]

      # Resolve backend
      resolved_backend = backend
      if resolved_backend is None:
        try:
          accel_config = accelerators.parse_accelerator(accelerator)
          # Use Pathways for multi-host TPUs (if supported) or simplified logic
          # For now, let's default to GKE unless explicit or strictly needed
          if (
            isinstance(accel_config, accelerators.TpuConfig)
            and accel_config.num_nodes > 1
          ):
            resolved_backend = "pathways"
          else:
            resolved_backend = "gke"
        except ValueError:
          resolved_backend = "gke"

      if resolved_backend == "gke":
        return _execute_on_gke(
          func,
          args,
          kwargs,
          accelerator,
          container_image,
          zone,
          project,
          cluster,
          namespace,
          env_vars,
          volumes,
        )
      elif resolved_backend == "pathways":
        return _execute_on_pathways(
          func,
          args,
          kwargs,
          accelerator,
          container_image,
          zone,
          project,
          cluster,
          namespace,
          env_vars,
          volumes,
        )
      else:
        raise ValueError(
          f"Unknown backend: {resolved_backend}. Use 'gke', 'pathways', or None for auto-detection"
        )

    return wrapper

  return decorator


def _execute_on_gke(
  func,
  args,
  kwargs,
  accelerator,
  container_image,
  zone,
  project,
  cluster,
  namespace,
  env_vars,
  volumes,
):
  """Execute function on GKE cluster with GPU/TPU nodes."""
  # Get GKE-specific defaults
  if not cluster:
    cluster = os.environ.get("KERAS_REMOTE_CLUSTER", DEFAULT_CLUSTER_NAME)
  if not namespace:
    namespace = os.environ.get("KERAS_REMOTE_NAMESPACE", "default")

  ctx = JobContext.from_params(
    func,
    args,
    kwargs,
    accelerator,
    container_image,
    zone,
    project,
    env_vars,
    cluster_name=cluster,
    volumes=volumes,
  )
  return execute_remote(ctx, GKEBackend(cluster=cluster, namespace=namespace))


def _execute_on_pathways(
  func,
  args,
  kwargs,
  accelerator,
  container_image,
  zone,
  project,
  cluster,
  namespace,
  env_vars,
  volumes,
):
  """Execute function on GKE cluster via ML Pathways."""
  if not cluster:
    cluster = os.environ.get("KERAS_REMOTE_CLUSTER", DEFAULT_CLUSTER_NAME)
  if not namespace:
    namespace = os.environ.get("KERAS_REMOTE_NAMESPACE", "default")

  ctx = JobContext.from_params(
    func,
    args,
    kwargs,
    accelerator,
    container_image,
    zone,
    project,
    env_vars,
    cluster_name=cluster,
    volumes=volumes,
  )
  return execute_remote(
    ctx, PathwaysBackend(cluster=cluster, namespace=namespace)
  )
