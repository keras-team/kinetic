"""Async job handles and detached job operations for Kinetic.

Provides `JobHandle` for observing, collecting, and cleaning up
remote jobs submitted via `kinetic.submit()`.  Includes `attach()`
for cross-session reattachment and `list_jobs()` for discovery.
"""

import contextlib
import time
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from typing import Any

import cloudpickle
from absl import logging
from google.api_core import exceptions as google_exceptions
from kubernetes import client

from kinetic.backend import gke_client, pathways_client
from kinetic.backend.log_streaming import LogStreamer
from kinetic.constants import (
  build_bucket_name,
  get_default_cluster_name,
  get_default_namespace,
  get_default_zone,
  get_required_project,
)
from kinetic.credentials import ensure_credentials
from kinetic.job_status import JobStatus  # re-export
from kinetic.utils import storage

_BACKEND_CLIENTS = {
  "gke": gke_client,
  "pathways": pathways_client,
}

_RESULT_POLL_INTERVAL_SECONDS = 5
_RESULT_DOWNLOAD_BACKOFF_SECONDS = (0, 1, 2, 4, 8, 16)
_TERMINAL_STATUSES = frozenset(
  {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.NOT_FOUND}
)


def _utcnow_iso() -> str:
  """Return an ISO 8601 UTC timestamp without fractional seconds."""
  return (
    datetime.now(timezone.utc)
    .replace(microsecond=0)
    .isoformat()
    .replace("+00:00", "Z")
  )


def attach_remote_traceback(
  exception: BaseException, remote_traceback: str | None
) -> BaseException:
  """Attach the remote traceback string to an exception when available."""
  if not remote_traceback or not hasattr(exception, "add_note"):
    return exception
  exception.add_note(f"Remote traceback:\n{remote_traceback}")
  return exception


@dataclass
class JobHandle:
  """Durable description of a submitted remote job.

  All fields are JSON-serializable strings.  No `func` object or
  closure state is stored — only the metadata needed to observe,
  collect, and clean up the job from any machine.
  """

  job_id: str
  backend: str
  project: str
  cluster_name: str
  zone: str
  namespace: str
  bucket_name: str
  k8s_name: str
  image_uri: str
  accelerator: str
  func_name: str
  display_name: str
  created_at: str

  # Optional group membership (set for collection children, None otherwise).
  group_id: str | None = None
  group_kind: str | None = None
  group_index: int | None = None

  # ------------------------------------------------------------------
  # Serialisation helpers
  # ------------------------------------------------------------------

  @classmethod
  def from_job_context(
    cls,
    ctx,
    backend_name: str,
    namespace: str,
    k8s_name: str,
  ) -> "JobHandle":
    """Build a `JobHandle` from a live `JobContext`."""
    return cls(
      job_id=ctx.job_id,
      backend=backend_name,
      project=ctx.project,
      cluster_name=ctx.cluster_name,
      zone=ctx.zone,
      namespace=namespace,
      bucket_name=ctx.bucket_name,
      k8s_name=k8s_name,
      image_uri=ctx.image_uri or "",
      accelerator=ctx.accelerator,
      func_name=ctx.func.__name__,
      display_name=ctx.display_name,
      created_at=_utcnow_iso(),
    )

  @classmethod
  def from_dict(cls, d: dict[str, Any]) -> "JobHandle":
    """Reconstruct a `JobHandle` from a plain dict.

    Unknown keys are silently ignored so that handles persisted by a
    future version (with extra fields) can still be loaded.
    """
    return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

  def to_dict(self) -> dict[str, str]:
    """Serialize the handle to a JSON-safe payload."""
    return {
      f.name: getattr(self, f.name)
      for f in fields(self)
      if getattr(self, f.name) is not None
    }

  # ------------------------------------------------------------------
  # Internal helpers
  # ------------------------------------------------------------------

  @property
  def _client(self):
    """Return the backend client module for this handle's backend."""
    try:
      return _BACKEND_CLIENTS[self.backend]
    except KeyError:
      raise ValueError(f"Unknown backend: {self.backend}") from None

  def _ensure_credentials(self) -> None:
    ensure_credentials(
      project=self.project, zone=self.zone, cluster=self.cluster_name
    )

  def _get_status(self) -> JobStatus:
    """Return the backend job status."""
    self._ensure_credentials()
    return self._client.get_job_status(self.k8s_name, namespace=self.namespace)

  def _get_pod_name(self) -> str | None:
    """Return the pod name used for log retrieval, if it exists."""
    self._ensure_credentials()
    return self._client.get_job_pod_name(
      self.k8s_name, namespace=self.namespace
    )

  def _get_logs(self, tail_lines: int | None = None) -> str:
    """Return log text for this job."""
    self._ensure_credentials()
    return self._client.get_job_logs(
      self.k8s_name,
      namespace=self.namespace,
      tail_lines=tail_lines,
    )

  def _cleanup_k8s_resource(
    self,
    timeout: float = 180,
    poll_interval: float = 2,
  ) -> None:
    """Delete the backend-specific Kubernetes resource if it exists."""
    self._ensure_credentials()
    self._client.cleanup_job(
      self.k8s_name,
      namespace=self.namespace,
      timeout=timeout,
      poll_interval=poll_interval,
    )

  def _download_result_payload(self) -> dict[str, Any]:
    """Download and deserialize the remote result payload."""
    result_path = storage.download_result(
      self.bucket_name,
      self.job_id,
      project=self.project,
    )
    with open(result_path, "rb") as f:
      return cloudpickle.load(f)

  def _download_result_payload_with_backoff(
    self, deadline: float | None
  ) -> dict[str, Any]:
    """Retry result download to handle post-exit GCS propagation lag."""
    last_error = None
    for delay in _RESULT_DOWNLOAD_BACKOFF_SECONDS:
      if delay:
        if deadline is not None and time.monotonic() + delay > deadline:
          break
        time.sleep(delay)
      try:
        return self._download_result_payload()
      except google_exceptions.NotFound as error:
        last_error = error
    if last_error is None:
      raise RuntimeError("result payload download retries were not attempted")
    raise last_error

  def _missing_result_error(self, status: JobStatus) -> RuntimeError:
    """Return a clear failure for terminal jobs without a result payload."""
    result_uri = f"gs://{self.bucket_name}/{self.job_id}/result.pkl"
    if status == JobStatus.NOT_FOUND:
      return RuntimeError(
        "Job resource was not found and no result payload exists at "
        f"{result_uri}"
      )
    if status == JobStatus.FAILED:
      return RuntimeError(
        f"Job failed but no result payload was found at {result_uri}"
      )
    return RuntimeError(
      f"Job completed but no result payload was found at {result_uri}"
    )

  def _stream_logs(self) -> None:
    """Stream logs to stdout via LogStreamer (blocking)."""
    self._ensure_credentials()
    core_v1 = client.CoreV1Api()
    pod_name = self._get_pod_name()
    if pod_name is None:
      raise RuntimeError(
        f"No pod found for job {self.job_id} — "
        "it may have been deleted or has not started yet."
      )
    with LogStreamer(core_v1, self.namespace) as streamer:
      streamer.start(pod_name)
      if streamer._thread is not None:
        streamer._thread.join()

  # ------------------------------------------------------------------
  # Observation & collection methods
  # ------------------------------------------------------------------

  def status(self) -> JobStatus:
    """Return the current execution status of the job."""
    return self._get_status()

  def logs(self, follow: bool = False) -> str | None:
    """Return logs or stream them to stdout until the job terminates."""
    if not follow:
      return self._get_logs()
    self._stream_logs()
    return None

  def tail(self, n: int = 100) -> str:
    """Return the last n log lines from the active pod."""
    return self._get_logs(tail_lines=n)

  def result(
    self,
    timeout: float | None = None,
    cleanup: bool = True,
    cleanup_timeout: float = 180,
    cleanup_poll_interval: float = 2,
    stream_logs: bool = False,
  ) -> Any:
    """Wait for the job result and return it or re-raise the user exception.

    Args:
      timeout: Maximum seconds to wait.  `None` means wait forever.
        If reached, `TimeoutError` is raised but the job keeps
        running and the handle remains valid.
      cleanup: When *True* (default), delete the k8s resource and
        GCS artifacts after a result payload is successfully
        downloaded.  Matches `run()` semantics.
      cleanup_timeout: Maximum seconds to wait for the k8s resource
        deletion to be confirmed.
      cleanup_poll_interval: Seconds between deletion-confirmation
        polls.
      stream_logs: When *True*, stream live pod logs to the terminal
        while waiting for the job to complete.

    Returns:
      The function's return value.

    Raises:
      TimeoutError: If *timeout* is exceeded.
      RuntimeError: If the job failed without uploading a result.
      Exception: Re-raised from the remote function on user failure.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    observed_status = None
    streamer_ctx = None

    if stream_logs:
      self._ensure_credentials()
      streamer_ctx = LogStreamer(client.CoreV1Api(), self.namespace)

    with streamer_ctx if streamer_ctx is not None else contextlib.nullcontext():
      while True:
        observed_status = self.status()
        if observed_status in _TERMINAL_STATUSES:
          break
        if deadline is not None and time.monotonic() >= deadline:
          raise TimeoutError(
            f"Timed out waiting for job {self.job_id} after {timeout}s"
          )
        if (
          streamer_ctx is not None
          and streamer_ctx._thread is None
          and observed_status == JobStatus.RUNNING
        ):
          pod_name = self._get_pod_name()
          if pod_name is not None:
            streamer_ctx.start(pod_name)
        time.sleep(_RESULT_POLL_INTERVAL_SECONDS)

    result_payload = None
    try:
      try:
        result_payload = self._download_result_payload_with_backoff(deadline)
      except google_exceptions.NotFound:
        raise self._missing_result_error(observed_status) from None

      if result_payload["success"]:
        return result_payload["result"]
      raise attach_remote_traceback(
        result_payload["exception"],
        result_payload.get("traceback"),
      )
    finally:
      if cleanup:
        try:
          self.cleanup(
            k8s=True,
            gcs=result_payload is not None,
            cleanup_timeout=cleanup_timeout,
            cleanup_poll_interval=cleanup_poll_interval,
          )
        except Exception:
          logging.warning(
            "Failed to clean up job %s after result collection",
            self.job_id,
          )

  def cancel(
    self,
    cleanup_timeout: float = 180,
    cleanup_poll_interval: float = 2,
  ) -> None:
    """Cancel the running job by deleting its Kubernetes resource."""
    self.cleanup(
      k8s=True,
      gcs=False,
      cleanup_timeout=cleanup_timeout,
      cleanup_poll_interval=cleanup_poll_interval,
    )

  def cleanup(
    self,
    k8s: bool = True,
    gcs: bool = True,
    cleanup_timeout: float = 180,
    cleanup_poll_interval: float = 2,
  ) -> None:
    """Clean up Kubernetes resources and/or uploaded GCS artifacts.

    Args:
      k8s: Delete the Kubernetes job/LWS resource.
      gcs: Delete uploaded GCS artifacts.
      cleanup_timeout: Maximum seconds to wait for the k8s resource
        deletion to be confirmed.
      cleanup_poll_interval: Seconds between deletion-confirmation
        polls.
    """
    if k8s:
      self._cleanup_k8s_resource(
        timeout=cleanup_timeout,
        poll_interval=cleanup_poll_interval,
      )
    if gcs:
      storage.cleanup_artifacts(
        self.bucket_name,
        self.job_id,
        project=self.project,
      )


# ------------------------------------------------------------------
# Top-level convenience functions
# ------------------------------------------------------------------


def attach(
  job_id: str,
  project: str | None = None,
  cluster: str | None = None,
) -> JobHandle:
  """Reconstruct a persisted handle from GCS.

  Args:
    job_id: The job identifier (e.g. `"job-a1b2c3d4"`).
    project: GCP project (uses default when *None*).
    cluster: GKE cluster name (uses default when *None*).

  Returns:
    A hydrated `JobHandle` ready for `status()`, `result()`, etc.
  """
  project = get_required_project(project)
  cluster_name = cluster or get_default_cluster_name()
  bucket_name = build_bucket_name(project, cluster_name)
  payload = storage.download_handle(
    bucket_name,
    job_id,
    project=project,
  )
  return JobHandle.from_dict(payload)


def list_jobs(
  project: str | None = None,
  zone: str | None = None,
  cluster: str | None = None,
  namespace: str | None = None,
) -> list[JobHandle]:
  """List live jobs by hydrating durable handles from discovered k8s jobs.

  Queries Kubernetes for GKE Jobs and Pathways LWS resources that
  carry the `app=kinetic` / `app=kinetic-pathways` labels, then
  downloads each job's `handle.json` from GCS.  Jobs whose
  `handle.json` is missing are skipped with a warning.
  """
  project = get_required_project(project)
  zone = zone or get_default_zone()
  cluster_name = cluster or get_default_cluster_name()
  namespace = get_default_namespace(namespace)
  bucket_name = build_bucket_name(project, cluster_name)

  ensure_credentials(
    project=project,
    zone=zone,
    cluster=cluster_name,
  )

  discovered: list[dict[str, str]] = []
  try:
    discovered.extend(gke_client.list_jobs(namespace=namespace))
  except Exception:
    logging.warning("Failed to list GKE jobs")
  try:
    discovered.extend(pathways_client.list_jobs(namespace=namespace))
  except Exception:
    logging.warning("Failed to list Pathways jobs")

  handles: list[JobHandle] = []
  for item in discovered:
    job_id = item["job_id"]
    try:
      payload = storage.download_handle(
        bucket_name,
        job_id,
        project=project,
      )
      handles.append(JobHandle.from_dict(payload))
    except (ValueError, TypeError, KeyError, google_exceptions.NotFound):
      logging.warning(
        "Skipping discovered job %s because its handle could not be loaded",
        job_id,
      )

  return sorted(handles, key=lambda handle: handle.created_at, reverse=True)
