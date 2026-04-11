"""Async collection orchestration for Kinetic.

Provides `map()` for job-array-style fan-out, `BatchHandle` for
observing and collecting collection results, and `attach_batch()`
for cross-session reattachment.
"""

from __future__ import annotations

import collections
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator

from absl import logging

from kinetic.collections_helpers import (
  append_child_to_manifest,
  build_initial_manifest,
  call_with_input,
)
from kinetic.constants import (
  build_bucket_name,
  get_default_cluster_name,
  get_required_project,
)
from kinetic.job_status import JobStatus
from kinetic.jobs import _TERMINAL_STATUSES, JobHandle
from kinetic.utils import storage

_DEFAULT_MAX_CONCURRENT = 64
_STATUS_POLL_INTERVAL = 5.0


def _resolve_bucket(
  project: str | None, cluster: str | None
) -> tuple[str, str]:
  """Return `(resolved_project, bucket_name)`."""
  resolved_project = get_required_project(project)
  resolved_cluster = cluster or get_default_cluster_name()
  return resolved_project, build_bucket_name(resolved_project, resolved_cluster)


class BatchError(Exception):
  """Raised when a batch collection has failed children.

  Attributes:
    group_id: The collection's group identifier.
    failures: List of JobHandles for failed children.
    partial_results: List where successful positions contain the
      result and failed positions contain `None`.
  """

  def __init__(
    self,
    group_id: str,
    failures: list[JobHandle],
    partial_results: list[Any],
  ):
    self.group_id = group_id
    self.failures = failures
    self.partial_results = partial_results
    n_failed = len(failures)
    n_total = len(partial_results)
    super().__init__(f"Batch {group_id}: {n_failed} of {n_total} jobs failed")


@dataclass
class BatchHandle:
  """Handle for a collection of submitted jobs.

  Created by `kinetic.map()` or reconstructed by
  `kinetic.attach_batch()`.  Provides collection-level observation,
  result gathering, and cleanup.
  """

  group_id: str
  name: str | None
  tags: dict[str, str]
  jobs: list[JobHandle | None]

  # Bucket / project derived from eager resolution in map().
  _bucket_name: str = field(default="", repr=False, compare=False)
  _project: str = field(default="", repr=False, compare=False)

  # Internal state for background submission.
  _submission_complete: threading.Event = field(
    default_factory=threading.Event, repr=False, compare=False
  )
  _submission_error: BaseException | None = field(
    default=None, repr=False, compare=False
  )
  _lock: threading.Lock = field(
    default_factory=threading.Lock, repr=False, compare=False
  )

  # Per-index submission errors (index -> exception).
  _submission_errors: dict[int, Exception] = field(
    default_factory=dict, repr=False, compare=False
  )

  # ------------------------------------------------------------------
  # Observation
  # ------------------------------------------------------------------

  def statuses(self) -> list[tuple[int, JobStatus]]:
    """Return `(index, status)` for each submitted job."""
    return [
      (i, job.status()) for i, job in enumerate(self.jobs) if job is not None
    ]

  def status_counts(self) -> dict[str, int]:
    """Return a count of jobs in each status."""
    return dict(collections.Counter(s.value for _, s in self.statuses()))

  def _all_accounted_for(self, seen: set[int]) -> bool:
    """True when every job slot is either seen-terminal or a submission error."""
    if not self._submission_complete.is_set():
      return False
    with self._lock:
      total_submitted = sum(1 for j in self.jobs if j is not None)
    total_errors = len(self._submission_errors)
    return len(seen) >= total_submitted and (
      len(seen) + total_errors >= len(self.jobs)
    )

  # ------------------------------------------------------------------
  # Blocking helpers
  # ------------------------------------------------------------------

  def wait(self, *, timeout: float | None = None) -> None:
    """Block until all jobs reach a terminal state."""
    deadline = None if timeout is None else time.monotonic() + timeout

    # Wait for background submission to finish first.
    if not self._submission_complete.is_set():
      remaining = (
        None if deadline is None else max(0, deadline - time.monotonic())
      )
      if not self._submission_complete.wait(timeout=remaining):
        raise TimeoutError(
          f"Timed out waiting for submission to complete "
          f"for batch {self.group_id}"
        )

    if self._submission_error is not None:
      raise self._submission_error

    # Poll until every submitted job is terminal.
    while True:
      if all(
        job.status() in _TERMINAL_STATUSES
        for job in self.jobs
        if job is not None
      ):
        return
      if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError(
          f"Timed out waiting for batch {self.group_id} after {timeout}s"
        )
      time.sleep(_STATUS_POLL_INTERVAL)

  def as_completed(
    self,
    *,
    poll_interval: float = 5.0,
    timeout: float | None = None,
  ) -> Iterator[JobHandle]:
    """Yield jobs as they reach terminal states, in completion order.

    Unlike the simple approach of waiting for all submissions first,
    this streams results as soon as each job reaches a terminal state
    — even while more inputs are still being submitted.

    Args:
      poll_interval: Seconds between status polls.
      timeout: Maximum seconds to wait.  Raises `TimeoutError` if
        exceeded.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    seen: set[int] = set()

    while True:
      # Snapshot current jobs (slots may be filled by the submission thread).
      with self._lock:
        current_jobs = list(enumerate(self.jobs))

      newly_done = []
      for i, job in current_jobs:
        if i in seen or job is None:
          continue
        if job.status() in _TERMINAL_STATUSES:
          newly_done.append(i)

      for i in newly_done:
        seen.add(i)
        yield self.jobs[i]  # type: ignore[misc]

      if self._all_accounted_for(seen):
        break

      if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError(
          f"as_completed() timed out after {timeout}s for batch {self.group_id}"
        )

      if not newly_done:
        time.sleep(poll_interval)

  def results(
    self,
    *,
    timeout: float | None = None,
    ordered: bool = True,
    cleanup: bool = True,
    return_exceptions: bool = False,
  ) -> list[Any]:
    """Collect results from all jobs.

    Args:
      timeout: Maximum seconds to wait for all jobs.
      ordered: If *True*, return in input order.  If *False*,
        return in completion order.
      cleanup: If *True*, clean up each child's K8s and GCS
        resources (the group manifest is preserved). Note that
        cleaning up causes `failures()` to return an empty list
        as job statuses become `NOT_FOUND`.
      return_exceptions: If *True*, failed positions contain the
        exception object.  If *False*, raise `BatchError` on any
        failure.

    Returns:
      List of results (input order when *ordered=True*, completion
      order otherwise).
    """
    if ordered:
      results_list, failures = self._results_ordered(
        timeout=timeout, cleanup=cleanup, return_exceptions=return_exceptions
      )
    else:
      results_list, failures = self._results_completion_order(
        timeout=timeout, cleanup=cleanup, return_exceptions=return_exceptions
      )

    if failures and not return_exceptions:
      raise BatchError(
        group_id=self.group_id,
        failures=failures,
        partial_results=results_list,
      )

    return results_list

  def _results_ordered(
    self,
    *,
    timeout: float | None,
    cleanup: bool,
    return_exceptions: bool,
  ) -> tuple[list[Any], list[JobHandle]]:
    """Collect results in input order (waits for all jobs first)."""
    self.wait(timeout=timeout)
    failures: list[JobHandle] = []
    results_list: list[Any] = [None] * len(self.jobs)

    for i, job in enumerate(self.jobs):
      if job is None:
        if i in self._submission_errors:
          exc = self._submission_errors[i]
          if return_exceptions:
            results_list[i] = exc
          else:
            failures.append(None)  # type: ignore[arg-type]
        continue
      try:
        results_list[i] = job.result(cleanup=cleanup)
      except Exception as exc:
        if return_exceptions:
          results_list[i] = exc
        else:
          failures.append(job)

    return results_list, failures

  def _results_completion_order(
    self,
    *,
    timeout: float | None,
    cleanup: bool,
    return_exceptions: bool,
  ) -> tuple[list[Any], list[JobHandle]]:
    """Collect results in completion order, streaming as they arrive."""
    failures: list[JobHandle] = []
    results_list: list[Any] = []

    for job in self.as_completed(timeout=timeout):
      try:
        results_list.append(job.result(cleanup=cleanup))
      except Exception as exc:
        if return_exceptions:
          results_list.append(exc)
        else:
          failures.append(job)

    for idx in sorted(self._submission_errors):
      exc = self._submission_errors[idx]
      if return_exceptions:
        results_list.append(exc)
      else:
        failures.append(None)  # type: ignore[arg-type]

    return results_list, failures

  def failures(self) -> list[JobHandle]:
    """Return handles for jobs that failed.

    Only includes jobs whose status is `FAILED`.  Jobs that are
    `NOT_FOUND` (e.g. after cleanup) are excluded because the
    status is ambiguous — use `statuses()` for finer control.

    Note:
      If `results(cleanup=True)` was called (the default), child
      resources are deleted and their status becomes `NOT_FOUND`.
      In that case, this method will return an empty list.
    """
    return [
      job
      for job in self.jobs
      if job is not None and job.status() == JobStatus.FAILED
    ]

  def cancel(self) -> None:
    """Cancel all non-terminal jobs in the collection."""
    for job in self.jobs:
      if job is None:
        continue
      try:
        if job.status() not in _TERMINAL_STATUSES:
          job.cancel()
      except Exception:
        logging.warning("Failed to cancel job %s", job.job_id)

  def cleanup(self, *, k8s: bool = True, gcs: bool = True) -> None:
    """Clean up all jobs and optionally the group manifest.

    Args:
      k8s: Delete K8s resources for each child.
      gcs: Delete GCS artifacts for each child **and** the group
        manifest.
    """
    for job in self.jobs:
      if job is None:
        continue
      try:
        job.cleanup(k8s=k8s, gcs=gcs)
      except Exception:
        logging.warning("Failed to clean up job %s", job.job_id)

    if gcs:
      bucket = self._bucket_name
      project = self._project
      if not bucket and self.jobs:
        first = next((j for j in self.jobs if j is not None), None)
        if first is not None:
          bucket = first.bucket_name
          project = first.project
      if bucket:
        try:
          storage.cleanup_manifest(bucket, self.group_id, project=project)
        except Exception:
          logging.warning(
            "Failed to clean up manifest for group %s", self.group_id
          )


# ------------------------------------------------------------------
# Submission loop
# ------------------------------------------------------------------


def _cancel_active(handle: BatchHandle, active_indices: set[int]) -> None:
  """Best-effort cancel of all active jobs."""
  for idx in list(active_indices):
    job = handle.jobs[idx]
    if job is None:
      continue
    try:
      job.cancel()
    except Exception:
      logging.warning("Failed to cancel job at index %d", idx)


def _submission_loop(
  submit_fn,
  inputs: list,
  input_mode: str,
  manifest: dict,
  handle: BatchHandle,
  max_concurrent: int | None,
  retries: int,
  fail_fast: bool,
  cancel_running_on_fail: bool,
) -> None:
  """Core submission and retry loop.

  Mutates *handle.jobs* and *manifest* in place.  Runs in the calling
  thread (`max_concurrent=None` and `retries=0`) or in a daemon
  thread otherwise.
  """
  group_id = handle.group_id
  group_kind = manifest["group_kind"]
  bucket_name = handle._bucket_name
  project = handle._project

  attempt_counts = [0] * len(inputs)
  pending_indices = collections.deque(range(len(inputs)))
  active_indices: set[int] = set()
  stop_launching = False
  max_attempts = 1 + retries

  try:
    while pending_indices or active_indices:
      # Submit pending jobs up to concurrency limit
      while pending_indices and not stop_launching:
        if max_concurrent is not None and len(active_indices) >= max_concurrent:
          break

        idx = pending_indices.popleft()
        attempt_counts[idx] += 1

        try:
          job_handle = call_with_input(submit_fn, inputs[idx], input_mode)
        except Exception as exc:
          logging.error("Submission failed for index %d: %s", idx, exc)
          with handle._lock:
            handle._submission_errors[idx] = exc
          if fail_fast:
            stop_launching = True
            if cancel_running_on_fail:
              _cancel_active(handle, active_indices)
          continue

        # Inject group metadata and re-upload handle.
        job_handle.group_id = group_id
        job_handle.group_kind = group_kind
        job_handle.group_index = idx

        try:
          storage.upload_handle(
            job_handle.bucket_name,
            job_handle.job_id,
            job_handle.to_dict(),
            project=job_handle.project,
          )
        except Exception:
          logging.warning(
            "Failed to re-upload handle with group fields for %s",
            job_handle.job_id,
          )

        with handle._lock:
          handle.jobs[idx] = job_handle

        active_indices.add(idx)

        append_child_to_manifest(
          manifest, idx, job_handle.job_id, attempt_counts[idx]
        )
        try:
          storage.upload_manifest(
            bucket_name, group_id, manifest, project=project
          )
        except Exception:
          logging.warning(
            "Failed to update manifest after submitting index %d",
            idx,
          )

      # If fail_fast stopped launching, discard remaining pending jobs.
      if stop_launching:
        pending_indices.clear()

      # Nothing left to do
      if not active_indices:
        break

      # All jobs submitted, no retries, and fail_fast is off — no need
      # to poll.  The caller uses wait()/results() to observe terminal
      # states.  When fail_fast is on we must keep polling so that
      # runtime failures trigger sibling cancellation.
      if not pending_indices and retries == 0 and not fail_fast:
        break

      # Poll active jobs for terminal states
      newly_terminal: list[tuple[int, JobStatus]] = []
      for idx in list(active_indices):
        job = handle.jobs[idx]
        if job is None:
          continue
        try:
          status = job.status()
          if status in _TERMINAL_STATUSES:
            newly_terminal.append((idx, status))
        except Exception:
          logging.warning("Failed to poll status for index %d", idx)

      for idx, status in newly_terminal:
        active_indices.discard(idx)

        if status in (JobStatus.FAILED, JobStatus.NOT_FOUND):
          if attempt_counts[idx] < max_attempts:
            # Clean up previous attempt's K8s resources.
            try:
              handle.jobs[idx].cleanup(k8s=True, gcs=False)  # type: ignore[union-attr]
            except Exception:
              logging.warning(
                "Failed to clean up before retry for index %d",
                idx,
              )
            pending_indices.append(idx)
          else:
            if fail_fast:
              stop_launching = True
              if cancel_running_on_fail:
                _cancel_active(handle, active_indices)

      if active_indices or pending_indices:
        time.sleep(_STATUS_POLL_INTERVAL)

  except BaseException as exc:
    handle._submission_error = exc
    logging.error("Submission loop error: %s", exc)
  finally:
    handle._submission_complete.set()


def map(
  submit_fn,
  inputs,
  *,
  input_mode: str = "auto",
  max_concurrent: int | None = _DEFAULT_MAX_CONCURRENT,
  retries: int = 0,
  fail_fast: bool = False,
  cancel_running_on_fail: bool = False,
  name: str | None = None,
  tags: dict[str, str] | None = None,
  project: str | None = None,
  cluster: str | None = None,
) -> BatchHandle:
  """Launch many independent jobs over a set of inputs.

  `submit_fn` must be a function decorated with
  `@kinetic.submit(...)`.  Each input is dispatched according to
  `input_mode` and submitted as a separate remote job.

  Args:
    submit_fn: A `@kinetic.submit`-decorated callable.
    inputs: Iterable of inputs to fan out over.
    input_mode: How each input item is passed to *submit_fn*.
      `"auto"` (default) dispatches dicts as `**kwargs`,
      lists/tuples as `*args`, and scalars as a single positional
      argument.
    max_concurrent: Maximum number of concurrently active jobs.
      `None` submits all immediately.
    retries: Number of additional attempts after a job failure.
    fail_fast: Stop launching new jobs after the first failure.
    cancel_running_on_fail: Cancel running siblings on failure.
    name: Human-readable collection name.
    tags: Arbitrary key-value metadata.
    project: GCP project (uses default when *None*).
    cluster: GKE cluster name (uses default when *None*).

  Returns:
    A `BatchHandle` for observing, collecting, and cleaning up
    the collection.
  """
  if not callable(submit_fn):
    raise TypeError("submit_fn must be callable")

  if max_concurrent is not None and max_concurrent < 1:
    raise ValueError(
      f"max_concurrent must be a positive integer, got {max_concurrent}"
    )

  if retries < 0:
    raise ValueError(f"retries must be non-negative, got {retries}")

  if input_mode not in ("auto", "single", "args", "kwargs"):
    raise ValueError(f"Unknown input_mode: {input_mode!r}")

  inputs = list(inputs)
  if not inputs:
    raise ValueError("inputs must be non-empty")

  # Resolve bucket eagerly so the initial manifest can be written
  # before any jobs are submitted.
  resolved_project, bucket_name = _resolve_bucket(project, cluster)

  group_id = f"grp-{uuid.uuid4().hex[:8]}"
  group_kind = "map"
  fn_name = getattr(submit_fn, "__name__", str(submit_fn))

  manifest = build_initial_manifest(
    group_id, group_kind, name, tags, len(inputs), fn_name
  )

  # Write the initial manifest (empty children) before any jobs are
  # submitted so that crash recovery can distinguish "0 of N
  # submitted" from "collection never created".
  storage.upload_manifest(
    bucket_name, group_id, manifest, project=resolved_project
  )

  # Pre-allocate the jobs list with None placeholders.
  jobs: list[JobHandle | None] = [None] * len(inputs)

  handle = BatchHandle(
    group_id=group_id,
    name=name,
    tags=tags or {},
    jobs=jobs,
    _bucket_name=bucket_name,
    _project=resolved_project,
  )

  if max_concurrent is None and len(inputs) > 100:
    logging.warning(
      "Submitting %d jobs with max_concurrent=None. "
      "Consider setting max_concurrent to limit resource usage.",
      len(inputs),
    )

  if max_concurrent is None and retries == 0:
    # Simple path: submit all in calling thread.
    _submission_loop(
      submit_fn=submit_fn,
      inputs=inputs,
      input_mode=input_mode,
      manifest=manifest,
      handle=handle,
      max_concurrent=max_concurrent,
      retries=retries,
      fail_fast=fail_fast,
      cancel_running_on_fail=cancel_running_on_fail,
    )
  else:
    # Background thread for bounded concurrency or retries.
    thread = threading.Thread(
      target=_submission_loop,
      kwargs={
        "submit_fn": submit_fn,
        "inputs": inputs,
        "input_mode": input_mode,
        "manifest": manifest,
        "handle": handle,
        "max_concurrent": max_concurrent,
        "retries": retries,
        "fail_fast": fail_fast,
        "cancel_running_on_fail": cancel_running_on_fail,
      },
      daemon=False,
    )
    thread.start()

  return handle


# ------------------------------------------------------------------
# Public API — attach_batch
# ------------------------------------------------------------------


def attach_batch(
  group_id: str,
  project: str | None = None,
  cluster: str | None = None,
) -> BatchHandle:
  """Reattach to an existing batch collection by *group_id*.

  Downloads the group manifest from GCS, reconstructs `JobHandle`
  objects for each child, and returns a fully usable `BatchHandle`.

  Args:
    group_id: The collection identifier (e.g. `"grp-a1b2c3d4"`).
    project: GCP project (uses default when *None*).
    cluster: GKE cluster name (uses default when *None*).

  Returns:
    A hydrated `BatchHandle` ready for `results()`, etc.
  """
  resolved_project, bucket_name = _resolve_bucket(project, cluster)

  manifest = storage.download_manifest(
    bucket_name, group_id, project=resolved_project
  )

  children = manifest.get("children", [])
  total_expected = manifest.get("total_expected", len(children))

  # Preallocate to total_expected and slot each child by group_index
  # so that index alignment is preserved even when some handles are
  # missing or the batch was only partially submitted.
  jobs: list[JobHandle | None] = [None] * total_expected

  for child in children:
    idx = child["group_index"]
    if not isinstance(idx, int) or idx < 0 or idx >= total_expected:
      logging.warning(
        "Invalid child index %r (total_expected=%d); skipping",
        idx,
        total_expected,
      )
      continue
    try:
      payload = storage.download_handle(
        bucket_name, child["job_id"], project=resolved_project
      )
      jobs[idx] = JobHandle.from_dict(payload)
    except Exception:
      logging.warning(
        "Could not load handle for child job %s (index %d); skipping",
        child["job_id"],
        idx,
      )

  if len(children) < total_expected:
    logging.warning(
      "Batch %s was partially submitted: %d of %d expected jobs",
      group_id,
      len(children),
      total_expected,
    )

  handle = BatchHandle(
    group_id=manifest["group_id"],
    name=manifest.get("group_name"),
    tags=manifest.get("tags", {}),
    jobs=jobs,
    _bucket_name=bucket_name,
    _project=resolved_project,
  )
  # Submission is already complete for reattached handles — the
  # reattached handle has no submission thread.
  handle._submission_complete.set()

  return handle
