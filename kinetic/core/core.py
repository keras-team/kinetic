from __future__ import annotations

import functools
import os
from typing import Any, Callable

from kinetic.backend.execution import (
  GKEBackend,
  JobContext,
  PathwaysBackend,
  submit_remote,
)
from kinetic.constants import DEFAULT_CLUSTER_NAME, get_default_namespace
from kinetic.core import accelerators
from kinetic.data import Data
from kinetic.jobs import JobHandle


def _validate_volumes(volumes):
  """Validate the optional volumes mapping."""
  if volumes is None:
    return
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


def _capture_env(capture_env_vars):
  """Capture requested environment variables for remote execution."""
  env_vars = {}
  if not capture_env_vars:
    return env_vars

  for pattern in capture_env_vars:
    if pattern.endswith("*"):
      prefix = pattern[:-1]
      env_vars.update(
        {k: v for k, v in os.environ.items() if k.startswith(prefix)}
      )
    elif pattern in os.environ:
      env_vars[pattern] = os.environ[pattern]
  return env_vars


def _resolve_backend_name(accelerator, backend, spot=False):
  """Resolve the backend from explicit config or accelerator type."""
  if backend is not None:
    return backend

  try:
    accel_config = accelerators.parse_accelerator(accelerator, spot=spot)
    if (
      isinstance(accel_config, accelerators.TpuConfig)
      and accel_config.num_nodes > 1
    ):
      return "pathways"
  except ValueError:
    pass
  return "gke"


def _make_decorator(
  accelerator,
  container_image,
  base_image_repo,
  zone,
  project,
  capture_env_vars,
  cluster,
  backend,
  namespace,
  volumes,
  spot,
  sync,
  output_dir,
):
  """Build a decorator that submits the wrapped function for remote execution.

  Args:
    sync: If True, block on result (`run()` semantics).
      If False, return a `JobHandle` immediately (`submit()` semantics).
  """
  _validate_volumes(volumes)

  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      env_vars = _capture_env(capture_env_vars)
      resolved_backend = _resolve_backend_name(accelerator, backend, spot=spot)

      if resolved_backend not in ("gke", "pathways"):
        raise ValueError(
          f"Unknown backend: {resolved_backend}. "
          "Use 'gke', 'pathways', or None for auto-detection"
        )

      resolved_cluster = cluster or os.environ.get(
        "KINETIC_CLUSTER", DEFAULT_CLUSTER_NAME
      )
      resolved_namespace = get_default_namespace(namespace)

      ctx = JobContext.from_params(
        func,
        args,
        kwargs,
        accelerator,
        container_image,
        zone,
        project,
        env_vars,
        cluster_name=resolved_cluster,
        volumes=volumes,
        spot=spot,
        output_dir=output_dir,
        base_image_repo=base_image_repo,
      )

      if resolved_backend == "pathways":
        backend_inst = PathwaysBackend(
          cluster=resolved_cluster, namespace=resolved_namespace
        )
      else:
        backend_inst = GKEBackend(
          cluster=resolved_cluster, namespace=resolved_namespace
        )

      handle = submit_remote(ctx, backend_inst)
      return handle.result(stream_logs=True) if sync else handle

    return wrapper

  return decorator


def run(
  accelerator: str = "v5e-1",
  container_image: str | None = None,
  base_image_repo: str | None = None,
  zone: str | None = None,
  project: str | None = None,
  capture_env_vars: list[str] | None = None,
  cluster: str | None = None,
  backend: str | None = None,
  namespace: str | None = None,
  volumes: dict[str, Data] | None = None,
  spot: bool = False,
  output_dir: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
  """Execute function on remote TPU/GPU.

  Args:
    accelerator: TPU/GPU type (e.g., 'v3-8', 'v5litepod-4', 'l4', 'a100')
    container_image: Controls the container image used for execution.
      `None` or `"bundled"` (default) builds a custom image with all
      dependencies baked in via Cloud Build.  `"prebuilt"` uses a
      prebuilt base image and installs user requirements at pod startup
      via `uv pip install`.  Any other string is treated as a custom
      container image URI.
    base_image_repo: Docker Hub repository for prebuilt base images
      (e.g., `"mycompany/kinetic"`). Defaults to `KINETIC_BASE_IMAGE_REPO`
      env var, then `"kinetic"`. Only used when `container_image` is
      `"prebuilt"`.
    zone: GCP zone (default: from KINETIC_ZONE or 'us-central1-a')
    project: GCP project (default: from KINETIC_PROJECT)
    capture_env_vars: List of environment variable names or patterns (ending in `*`)
      to propagate to the remote environment. Defaults to None.
    cluster: GKE cluster name (default: from KINETIC_CLUSTER)
    backend: Backend to use ('gke' or 'pathways')
    namespace: Kubernetes namespace (default: None, resolved via
      KINETIC_NAMESPACE env var or 'default')
    volumes: Dict mapping absolute mount paths to Data objects, e.g.
      `{"/data": Data("./dataset/")}`. Data is downloaded to these
      paths on the pod before function execution.
    spot: If True, use preemptible/spot VMs for the job.
  """
  return _make_decorator(
    accelerator,
    container_image,
    base_image_repo,
    zone,
    project,
    capture_env_vars,
    cluster,
    backend,
    namespace,
    volumes,
    spot,
    sync=True,
    output_dir=output_dir,
  )


def submit(
  accelerator: str = "v5e-1",
  container_image: str | None = None,
  base_image_repo: str | None = None,
  zone: str | None = None,
  project: str | None = None,
  capture_env_vars: list[str] | None = None,
  cluster: str | None = None,
  backend: str | None = None,
  namespace: str | None = None,
  volumes: dict[str, Data] | None = None,
  spot: bool = False,
  output_dir: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., JobHandle]]:
  """Submit function for remote execution, returning a `JobHandle`.

  Same parameters as `run()`.  Blocks through container build and
  artifact upload, but returns immediately after k8s submission.
  Use the returned `JobHandle` to observe, collect, or cancel.

  Returns:
    A decorator whose wrapper returns a `JobHandle`.
  """
  return _make_decorator(
    accelerator,
    container_image,
    base_image_repo,
    zone,
    project,
    capture_env_vars,
    cluster,
    backend,
    namespace,
    volumes,
    spot,
    sync=False,
    output_dir=output_dir,
  )
