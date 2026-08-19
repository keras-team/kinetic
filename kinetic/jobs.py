"""Async job handles and detached job operations for Kinetic.

Provides `JobHandle` for observing, collecting, and cleaning up
remote jobs submitted via `func.run_async()`.  Includes `attach()`
for cross-session reattachment and `list_jobs()` for discovery.
"""

import contextlib
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from typing import Any

import cloudpickle
from absl import logging
from google.api_core import exceptions as google_exceptions
from kubernetes import client

from kinetic.backend import gke_client, k8s_utils, pathways_client
from kinetic.backend.log_streaming import LogStreamer
from kinetic.cli.profiles import resolve_infra
from kinetic.constants import build_bucket_name
from kinetic.credentials import ensure_credentials
from kinetic.debug import (
  DEBUGPY_PORT,
  print_attach_instructions,
  start_port_forward,
  wait_for_debug_server,
)
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


def _attach_note(exception: BaseException, note: str) -> BaseException:
  """Attach a diagnostic note to an exception when the runtime supports it.

  The hasattr guard is pure defensiveness: requires-python is >=3.11, where
  ``add_note`` always exists. There is deliberately no fallback that edits
  ``exception.args`` — mutating args can corrupt exceptions whose
  reconstruction or ``__str__`` depends on their shape.
  """
  if note and hasattr(exception, "add_note"):
    exception.add_note(note)
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

  # Debug mode — when True, the pod runs a debugpy server.
  debug: bool = False

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
      debug=ctx.debug,
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

  def _result_uri(self) -> str:
    """Return the GCS URI of this job's result payload."""
    return f"gs://{self.bucket_name}/{self.job_id}/result.pkl"

  def _blob_uri(self, blob_name: str) -> str:
    """Return the GCS URI of a blob inside this job's bucket."""
    return f"gs://{self.bucket_name}/{blob_name}"

  @property
  def _is_multi_host(self) -> bool:
    """Whether this job can run more than one pod.

    Only the Pathways (LeaderWorkerSet) backend does, so it is the only
    one that can leave per-host result payloads behind — every other
    backend runs a single pod and is spared the extra GCS listing.
    """
    return self.backend == "pathways"

  def _download_result_payload(self) -> dict[str, Any]:
    """Download and deserialize the remote result payload.

    Raises:
      RuntimeError: If the downloaded payload cannot be deserialized in
        this client (missing modules, renamed classes, version skew).
        The GCS artifacts are left in place so the result stays
        recoverable from an environment that can import its types.
    """
    result_path = storage.download_result(
      self.bucket_name,
      self.job_id,
      project=self.project,
    )
    try:
      with open(result_path, "rb") as f:
        return cloudpickle.load(f)
    except Exception as e:
      raise RuntimeError(
        f"Could not deserialize the result of job {self.job_id}: "
        f"{type(e).__name__}: {e}. The result references types from your "
        "project or environment that this client cannot import — reattach "
        "from the project directory (or an environment with the same "
        "packages). Artifacts were NOT deleted: "
        f"{self._result_uri()}"
      ) from e
    finally:
      try:
        os.remove(result_path)
      except OSError as e:
        logging.warning(
          "Failed to remove temporary result file %s: %s", result_path, e
        )

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

  def _download_worker_payload(self, blob_name: str) -> dict[str, Any] | None:
    """Download and deserialize one per-host result payload.

    Diagnostics only: every failure degrades to None with a warning
    rather than replacing the error the caller is already reporting.
    """
    try:
      local_path = storage.download_worker_result(
        self.bucket_name, blob_name, project=self.project
      )
    except Exception as e:
      logging.warning(
        "Could not download per-host result %s: %s",
        self._blob_uri(blob_name),
        e,
      )
      return None
    try:
      with open(local_path, "rb") as f:
        payload = cloudpickle.load(f)
    except Exception as e:
      logging.warning(
        "Could not deserialize per-host result %s: %s",
        self._blob_uri(blob_name),
        e,
      )
      return None
    finally:
      try:
        os.remove(local_path)
      except OSError as e:
        logging.warning(
          "Failed to remove temporary result file %s: %s", local_path, e
        )
    return payload if isinstance(payload, dict) else None

  def _list_worker_results(self) -> list[tuple[int, str]]:
    """Return the per-host result blobs this job left behind.

    Returns:
      `(host_index, blob_name)` pairs ordered by host index.  Empty for
      single-pod backends, for jobs whose non-leader hosts all completed,
      and whenever the listing itself could not be performed.
    """
    if not self._is_multi_host:
      return []
    try:
      return storage.list_worker_results(
        self.bucket_name, self.job_id, project=self.project
      )
    except Exception as e:
      logging.warning(
        "Could not list per-host result payloads for job %s: %s",
        self.job_id,
        e,
      )
      return []

  def _worker_failure_error(self) -> BaseException | None:
    """Return the exception reported by the lowest-indexed failing host.

    Non-leader hosts upload a failure payload each, so the exception
    that surfaces locally is decided by host index rather than by
    whichever pod wrote to GCS last.

    Only the payload that is actually re-raised is downloaded.  A
    non-leader writes its blob solely to report a failure, so the later
    entries need no download to be named — which matters on a slice
    where one collective timeout leaves a payload on every host.

    Returns:
      The remote exception to re-raise, or None when no host reported
      one (single-pod backends, an unreachable bucket, or payloads this
      client cannot deserialize).
    """
    entries = self._list_worker_results()
    for position, (host_index, blob_name) in enumerate(entries):
      payload = self._download_worker_payload(blob_name)
      # A non-leader writes this blob only to report a failure, so a
      # payload claiming success is not something the runner produces.
      # Skip it rather than report a success as the job's failure.
      if payload is None or payload.get("success"):
        continue
      exception = self._remote_failure(payload, self._blob_uri(blob_name))
      note = (
        f"Reported by host {host_index} of multi-host job {self.job_id} "
        f"({self._blob_uri(blob_name)})."
      )
      others = [str(index) for index, _ in entries[position + 1 :]]
      if others:
        note += (
          f" Other hosts that also reported a failure: {', '.join(others)}."
        )
      return _attach_note(exception, note)
    return None

  def _missing_result_error(self, status: JobStatus) -> RuntimeError:
    """Return a clear failure for terminal jobs without a result payload."""
    result_uri = self._result_uri()
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

  def _remote_failure(
    self, result_payload: dict[str, Any], result_uri: str | None = None
  ) -> BaseException:
    """Return the exception to raise for a non-successful result payload.

    Args:
      result_payload: The deserialized payload written by a remote host.
      result_uri: GCS URI the payload came from, named in the messages.
        Defaults to the job's leader payload.
    """
    result_uri = result_uri or self._result_uri()
    exception = result_payload.get("exception")
    if not isinstance(exception, BaseException):
      exception = RuntimeError(
        f"Job {self.job_id} failed but its result payload carried no usable "
        f"exception object (got {type(exception).__name__}: {exception!r}). "
        f"Artifacts were kept for inspection: {result_uri}"
      )
    if result_payload.get("serialization_failed"):
      result_repr = result_payload.get("result_repr") or "<unavailable>"
      _attach_note(
        exception,
        "The remote function completed but its return value could not be "
        f"serialized, so it cannot be retrieved. repr(result):\n{result_repr}\n"
        f"Artifacts were kept for inspection: {result_uri}",
      )
    phase = result_payload.get("phase")
    if phase:
      _attach_note(exception, f"Failed during kinetic phase: {phase}")
    return attach_remote_traceback(exception, result_payload.get("traceback"))

  def _false_success_error(self) -> RuntimeError:
    """Return the failure for a FAILED job whose payload claims success.

    In multi-host (Pathways) jobs the leader can finish cleanly and
    upload a success payload even though a worker pod failed, so the
    computation as a whole cannot be trusted.  Returning the leader's
    value would silently hide the failure; surface it instead, with pod
    failure details when they can still be collected.

    This is the fallback for when no host left a failure payload behind
    — a host killed outright by the kubelet (OOM, preemption, node
    eviction) never gets to write one.  When one exists,
    `_worker_failure_error` re-raises that host's exception instead.
    """
    msg = (
      f"Job {self.job_id} finished with status FAILED, but its result "
      "payload claims success. The main process likely uploaded its "
      "result while another part of the job failed (for multi-host "
      "jobs, a worker pod), so the result may be incomplete or wrong "
      "and was not returned. Artifacts were kept for inspection: "
      f"{self._result_uri()}"
    )
    details = ""
    try:
      self._ensure_credentials()
      details = k8s_utils.collect_pod_failure_details(
        k8s_utils.core_v1(), self.k8s_name, self.namespace
      )
    except Exception as e:
      logging.warning(
        "Could not collect pod failure details for job %s: %s",
        self.job_id,
        e,
      )
    if details:
      msg += f"\n{details}"
    return RuntimeError(msg)

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

  def debug_attach(
    self,
    local_port: int = DEBUGPY_PORT,
    working_dir: str | os.PathLike[str] | None = None,
  ) -> subprocess.Popen:
    """Wait for debugpy, start port-forward, and print VS Code config.

    Returns the port-forward subprocess so the caller can manage its
    lifecycle (e.g. terminate it after ``result()`` completes).

    Args:
      local_port: Local port to forward debugpy traffic to.
      working_dir: Local working directory for VS Code path mappings.
          The pod mirrors this path, so the printed mapping is the
          identity. If None, no pathMappings entry is printed.

    Returns:
      The ``subprocess.Popen`` handle for the kubectl port-forward
      process. The caller should call
      ``kinetic.debug.cleanup_port_forward(proc)`` when done.
    """
    self._ensure_credentials()

    # Wait for pod Running + debugpy ready sentinel file
    # before starting port-forward
    wait_for_debug_server(self)

    # Start kubectl port-forward
    pod_name = self._get_pod_name()
    if pod_name is None:
      raise RuntimeError(
        f"No pod found for job {self.job_id} — "
        "it may have been deleted or has not started yet."
      )
    pf_proc = start_port_forward(
      pod_name, self.namespace, local_port, DEBUGPY_PORT
    )

    # Print VS Code attach config
    print_attach_instructions(local_port, working_dir)

    return pf_proc

  def result(
    self,
    timeout: float | None = None,
    cleanup: bool | None = None,
    cleanup_timeout: float = 180,
    cleanup_poll_interval: float = 2,
    stream_logs: bool | None = None,
    on_status_change: Callable[[JobStatus], None] | None = None,
  ) -> Any:
    """Wait for the job result and return it or re-raise the user exception.

    Args:
      timeout: Maximum seconds to wait.  `None` means wait forever.
        If reached, `TimeoutError` is raised but the job keeps
        running and the handle remains valid.
      cleanup: When *True*, delete the k8s resource once the job is
        terminal.  GCS artifacts are only deleted when the job actually
        succeeded and its result was retrievable — a failed job, a
        missing result payload, a result that could not be serialized
        remotely, and a result that could not be deserialized locally
        all keep their artifacts for post-mortem debugging.  Defaults
        to *True* for normal jobs and *False* for debug jobs.
      cleanup_timeout: Maximum seconds to wait for the k8s resource
        deletion to be confirmed.
      cleanup_poll_interval: Seconds between deletion-confirmation
        polls.
      stream_logs: When *True*, stream live pod logs to the terminal
        while waiting for the job to complete.  Defaults to *False*
        for debug jobs to avoid Rich panel conflicts.
      on_status_change: Optional callback invoked with the new
        `JobStatus` each time the polled status differs from the
        previous one, including the first observation and the final
        terminal status.  Exceptions raised by the callback are
        logged and swallowed so they never break result collection.

    Returns:
      The function's return value.

    Raises:
      TimeoutError: If *timeout* is exceeded.
      RuntimeError: If the job failed without uploading a result, if
        the downloaded result payload cannot be deserialized locally,
        or if a job whose status is FAILED uploaded a success payload
        and no host recorded an exception explaining the failure.
      Exception: Re-raised from the remote function on user failure.
        For a multi-host job this is the leader's exception when the
        leader itself failed, and otherwise the exception from the
        lowest-indexed host that did.
    """
    if cleanup is None:
      cleanup = not self.debug
    if stream_logs is None:
      stream_logs = False

    deadline = None if timeout is None else time.monotonic() + timeout
    observed_status = None
    previous_status = None
    streamer_ctx = None

    if stream_logs:
      self._ensure_credentials()
      streamer_ctx = LogStreamer(client.CoreV1Api(), self.namespace)

    with streamer_ctx if streamer_ctx is not None else contextlib.nullcontext():
      while True:
        observed_status = self.status()
        if on_status_change is not None and observed_status != previous_status:
          try:
            on_status_change(observed_status)
          except Exception as exc:
            logging.exception(
              "on_status_change callback raised for job %s at status %s: %s",
              self.job_id,
              observed_status.value,
              exc,
            )
        previous_status = observed_status
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
    collected = False
    try:
      try:
        result_payload = self._download_result_payload_with_backoff(deadline)
      except google_exceptions.NotFound:
        # The leader never wrote a payload.  A host that failed before
        # it did still explains why far better than "no result payload
        # was found", so prefer its exception.
        worker_failure = self._worker_failure_error()
        if worker_failure is not None:
          raise worker_failure from None
        raise self._missing_result_error(observed_status) from None

      if not isinstance(result_payload, dict):
        raise RuntimeError(
          f"Job {self.job_id} returned an invalid result payload "
          f"(expected dict, got {type(result_payload).__name__}). The "
          f"artifact may be corrupted; it was kept for inspection: "
          f"{self._result_uri()}"
        )
      succeeded = bool(result_payload.get("success"))
      serialization_failed = bool(result_payload.get("serialization_failed"))
      if (
        succeeded
        and not serialization_failed
        and observed_status == JobStatus.FAILED
      ):
        # Never return a value from a job whose observed status is
        # FAILED, even when the payload claims success (e.g. a Pathways
        # worker pod failed after the leader uploaded its result).
        # `collected` stays False so the artifacts are preserved.
        # The failing host's own exception is the useful error; pod exit
        # summaries are the fallback when no host recorded one.
        worker_failure = self._worker_failure_error()
        if worker_failure is not None:
          raise worker_failure
        raise self._false_success_error()
      collected = succeeded and not serialization_failed
      if collected:
        return result_payload["result"]
      raise self._remote_failure(result_payload)
    finally:
      if cleanup:
        try:
          self.cleanup(
            k8s=True,
            gcs=collected,
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
    project: GCP project. Falls back to KINETIC_PROJECT, then the active
      profile's project, then GOOGLE_CLOUD_PROJECT.
    cluster: GKE cluster name. Falls back to KINETIC_CLUSTER, then the
      active profile's cluster, then the built-in default.

  Returns:
    A hydrated `JobHandle` ready for `status()`, `result()`, etc.
  """
  infra = resolve_infra(project=project, cluster=cluster)
  bucket_name = build_bucket_name(infra["project"], infra["cluster"])
  payload = storage.download_handle(
    bucket_name,
    job_id,
    project=infra["project"],
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

  Each field falls back through KINETIC_* env vars, the active profile,
  and finally the built-in defaults — matching `kinetic.run`.
  """
  infra = resolve_infra(
    project=project, zone=zone, cluster=cluster, namespace=namespace
  )
  bucket_name = build_bucket_name(infra["project"], infra["cluster"])

  ensure_credentials(
    project=infra["project"],
    zone=infra["zone"],
    cluster=infra["cluster"],
  )

  discovered: list[dict[str, str]] = []
  try:
    discovered.extend(gke_client.list_jobs(namespace=infra["namespace"]))
  except Exception:
    logging.warning("Failed to list GKE jobs")
  try:
    discovered.extend(pathways_client.list_jobs(namespace=infra["namespace"]))
  except Exception:
    logging.warning("Failed to list Pathways jobs")

  handles: list[JobHandle] = []
  for item in discovered:
    job_id = item["job_id"]
    try:
      payload = storage.download_handle(
        bucket_name,
        job_id,
        project=infra["project"],
      )
      handles.append(JobHandle.from_dict(payload))
    except (ValueError, TypeError, KeyError, google_exceptions.NotFound):
      logging.warning(
        "Skipping discovered job %s because its handle could not be loaded",
        job_id,
      )

  return sorted(handles, key=lambda handle: handle.created_at, reverse=True)
