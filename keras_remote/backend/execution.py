"""Unified remote execution module for GKE backend.

This module consolidates the common execution logic shared between different
backend implementations, reducing code duplication and improving maintainability.
"""

import inspect
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import cloudpickle
from absl import logging
from google.api_core import exceptions as google_exceptions

from keras_remote.backend import gke_client, pathways_client
from keras_remote.constants import get_default_zone, zone_to_region
from keras_remote.credentials import ensure_credentials
from keras_remote.infra import container_builder
from keras_remote.infra.infra import get_default_project
from keras_remote.utils import packager, storage


@dataclass
class JobContext:
  """Encapsulates all state for a remote job execution."""

  # Function and arguments
  func: Callable
  args: tuple
  kwargs: dict
  env_vars: dict

  # Configuration
  accelerator: str
  container_image: Optional[str]
  zone: str
  project: str

  # Generated identifiers
  job_id: str = field(default_factory=lambda: f"job-{uuid.uuid4().hex[:8]}")

  # Derived values (computed in __post_init__)
  bucket_name: str = field(init=False)
  region: str = field(init=False)
  display_name: str = field(init=False)

  # Artifact paths (set during prepare phase)
  payload_path: Optional[str] = None
  context_path: Optional[str] = None
  requirements_path: Optional[str] = None
  image_uri: Optional[str] = None

  def __post_init__(self):
    self.bucket_name = f"{self.project}-keras-remote-jobs"
    self.region = zone_to_region(self.zone)
    self.display_name = f"keras-remote-{self.func.__name__}-{self.job_id}"

  @classmethod
  def from_params(
    cls,
    func: Callable,
    args: tuple,
    kwargs: dict,
    accelerator: str,
    container_image: Optional[str],
    zone: Optional[str],
    project: Optional[str],
    env_vars: dict,
  ) -> "JobContext":
    """Factory method with default resolution for zone/project."""
    if not zone:
      zone = get_default_zone()
    if not project:
      project = get_default_project()
      if not project:
        raise ValueError(
          "project must be specified or set KERAS_REMOTE_PROJECT"
          " (or GOOGLE_CLOUD_PROJECT) environment variable"
        )

    return cls(
      func=func,
      args=args,
      kwargs=kwargs,
      env_vars=env_vars,
      accelerator=accelerator,
      container_image=container_image,
      zone=zone,
      project=project,
    )


class BaseK8sBackend:
  """Base class for Kubernetes-based backends."""

  def __init__(self, cluster: Optional[str] = None, namespace: str = "default"):
    self.cluster = cluster
    self.namespace = namespace

  def validate_preflight(self, ctx: JobContext) -> None:
    """Perform preflight checks before building container or uploading artifacts."""
    pass

  def submit_job(self, ctx: JobContext) -> Any:
    """Submit a job to the backend. Returns backend-specific job handle."""
    raise NotImplementedError

  def wait_for_job(self, job: Any, ctx: JobContext) -> None:
    """Wait for job completion. Raises RuntimeError if job fails."""
    raise NotImplementedError

  def cleanup_job(self, job: Any, ctx: JobContext) -> None:
    """Optional cleanup after job completion."""
    raise NotImplementedError


class GKEBackend(BaseK8sBackend):
  """Backend adapter for standard GKE Jobs."""

  def validate_preflight(self, ctx: JobContext) -> None:
    """Check if the required node pool exists for the accelerator."""
    gke_client.validate_preflight(
      accelerator=ctx.accelerator,
      project=ctx.project,
      cluster=self.cluster,
      zone=ctx.zone,
      namespace=self.namespace,
    )

  def submit_job(self, ctx: JobContext) -> Any:
    """Submit job to GKE cluster."""
    return gke_client.submit_k8s_job(
      display_name=ctx.display_name,
      container_uri=ctx.image_uri,
      accelerator=ctx.accelerator,
      project=ctx.project,
      job_id=ctx.job_id,
      bucket_name=ctx.bucket_name,
      namespace=self.namespace,
    )

  def wait_for_job(self, job: Any, ctx: JobContext) -> None:
    """Wait for GKE job completion."""
    gke_client.wait_for_job(job, namespace=self.namespace)

  def cleanup_job(self, job: Any, ctx: JobContext) -> None:
    """Clean up K8s job resources."""
    job_name = job.metadata.name
    gke_client.cleanup_job(job_name, namespace=self.namespace)


class PathwaysBackend(BaseK8sBackend):
  """Backend adapter for ML Pathways using LeaderWorkerSet."""

  def validate_preflight(self, ctx: JobContext) -> None:
    """Preflight checks for Pathways (currently same as GKE)."""
    # Pathways also runs on GKE nodes with specific labels
    gke_client.validate_preflight(
      accelerator=ctx.accelerator,
      project=ctx.project,
      cluster=self.cluster,
      zone=ctx.zone,
      namespace=self.namespace,
    )

  def submit_job(self, ctx: JobContext) -> Any:
    """Submit LWS job to GKE cluster."""
    return pathways_client.submit_pathways_job(
      display_name=ctx.display_name,
      container_uri=ctx.image_uri,
      accelerator=ctx.accelerator,
      project=ctx.project,
      job_id=ctx.job_id,
      bucket_name=ctx.bucket_name,
      namespace=self.namespace,
    )

  def wait_for_job(self, job: Any, ctx: JobContext) -> None:
    """Wait for Pathways LWS completion."""
    pathways_client.wait_for_job(ctx.job_id, namespace=self.namespace)

  def cleanup_job(self, job: Any, ctx: JobContext) -> None:
    """Clean up LWS resources."""
    job_name = pathways_client._get_job_name(ctx.job_id)
    pathways_client.cleanup_job(job_name, namespace=self.namespace)


def _find_requirements(start_dir: str) -> Optional[str]:
  """Search up directory tree for requirements.txt."""
  search_dir = start_dir
  while search_dir != "/":
    req_path = os.path.join(search_dir, "requirements.txt")
    if os.path.exists(req_path):
      return req_path
    parent_dir = os.path.dirname(search_dir)
    if parent_dir == search_dir:
      break
    search_dir = parent_dir
  return None


def _prepare_artifacts(
  ctx: JobContext, tmpdir: str, caller_frame_depth: int = 3
) -> None:
  """Phase 1: Package function payload and working directory context."""
  logging.info("Packaging function and context...")

  # Get caller directory
  frame = inspect.stack()[caller_frame_depth]
  module = inspect.getmodule(frame[0])
  if module:
    caller_path = os.path.dirname(os.path.abspath(module.__file__))
  else:
    caller_path = os.getcwd()

  # Serialize function + args
  ctx.payload_path = os.path.join(tmpdir, "payload.pkl")
  packager.save_payload(
    ctx.func, ctx.args, ctx.kwargs, ctx.env_vars, ctx.payload_path
  )
  logging.info("Payload serialized to %s", ctx.payload_path)

  # Zip working directory
  ctx.context_path = os.path.join(tmpdir, "context.zip")
  packager.zip_working_dir(caller_path, ctx.context_path)
  logging.info("Context packaged to %s", ctx.context_path)

  # Find requirements.txt
  ctx.requirements_path = _find_requirements(caller_path)
  if ctx.requirements_path:
    logging.info("Found requirements.txt: %s", ctx.requirements_path)
  else:
    logging.info("No requirements.txt found")


def _build_container(ctx: JobContext) -> None:
  """Phase 2: Build or get cached container image."""
  if ctx.container_image:
    ctx.image_uri = ctx.container_image
    logging.info("Using custom container: %s", ctx.image_uri)
  else:
    import sys

    logging.info("Building container image...")
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    ctx.image_uri = container_builder.get_or_build_container(
      base_image=f"python:{py_version}-slim",
      requirements_path=ctx.requirements_path,
      accelerator_type=ctx.accelerator,
      project=ctx.project,
      zone=ctx.zone,
    )


def _upload_artifacts(ctx: JobContext) -> None:
  """Phase 3: Upload artifacts to Cloud Storage."""
  logging.info("Uploading artifacts to Cloud Storage (job: %s)...", ctx.job_id)
  storage.upload_artifacts(
    bucket_name=ctx.bucket_name,
    job_id=ctx.job_id,
    payload_path=ctx.payload_path,
    context_path=ctx.context_path,
    project=ctx.project,
  )


def _download_result(ctx: JobContext) -> dict:
  """Phase 6: Download and deserialize result from Cloud Storage."""
  logging.info("Downloading result...")
  result_path = storage.download_result(
    ctx.bucket_name, ctx.job_id, project=ctx.project
  )

  with open(result_path, "rb") as f:
    return cloudpickle.load(f)


def _cleanup_and_return(ctx: JobContext, result_payload: dict) -> Any:
  """Phase 7: Cleanup Cloud Storage artifacts and handle result."""
  logging.info("Cleaning up artifacts...")
  storage.cleanup_artifacts(ctx.bucket_name, ctx.job_id, project=ctx.project)

  if result_payload["success"]:
    logging.info("Remote execution completed successfully")
    return result_payload["result"]
  else:
    logging.error("Remote execution failed:\n%s", result_payload["traceback"])
    raise result_payload["exception"]


def execute_remote(ctx: JobContext, backend: BaseK8sBackend) -> Any:
  """Execute a function remotely using the specified backend.

  This is the unified executor that handles all common phases
  and delegates backend-specific operations to the backend client.

  Args:
      ctx: Job context with function and configuration
      backend: Backend instance (GKEBackend or PathwaysBackend)

  Returns:
      The result of the remote function execution

  Raises:
      Exception: Re-raised from remote execution if it failed
  """
  ensure_credentials(
    project=ctx.project,
    zone=ctx.zone,
    cluster=backend.cluster,
  )

  # Preflight check
  backend.validate_preflight(ctx)

  with tempfile.TemporaryDirectory() as tmpdir:
    # Phase 1: Package artifacts
    _prepare_artifacts(ctx, tmpdir)

    # Phase 2: Build or get cached container image
    _build_container(ctx)

    # Phase 3: Upload artifacts to Cloud Storage
    _upload_artifacts(ctx)

    # Phase 4: Submit job (backend-specific)
    logging.info("Submitting job to %s...", backend.__class__.__name__)
    job = backend.submit_job(ctx)

    # Phase 5: Wait for completion (with cleanup on failure)
    job_error = None
    try:
      backend.wait_for_job(job, ctx)
    except RuntimeError as e:
      job_error = e
    finally:
      backend.cleanup_job(job, ctx)

    # Phase 6: Download and deserialize result
    # Try even if the job failed — the runner may have captured a user
    # exception and uploaded the result before exiting with non-zero.
    if job_error is not None:
      try:
        result_payload = _download_result(ctx)
      except google_exceptions.NotFound:
        # Result wasn't uploaded (infrastructure failure), surface the
        # original job error.
        raise job_error from None
    else:
      result_payload = _download_result(ctx)

    # Phase 7: Cleanup and return/raise
    return _cleanup_and_return(ctx, result_payload)
