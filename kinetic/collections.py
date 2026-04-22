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
from google.api_core import exceptions as google_exceptions

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
_MANIFEST_POLL_INTERVAL = 10.0


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

  # Cached failure list populated by results() so that failures()
  # remains accurate after cleanup deletes K8s resources.
  _cached_failures: list[JobHandle] | None = field(
    default=None, repr=False, compare=False
  )

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
        break
      if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError(
          f"Timed out waiting for batch {self.group_id} after {timeout}s"
        )
      time.sleep(_STATUS_POLL_INTERVAL)

    if self._submission_errors:
      logging.warning(
        "Batch %s: %d input(s) failed at submission time. "
        "Use handle.submission_failures to inspect.",
        self.group_id,
        len(self._submission_errors),
      )

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
          failures.append(None)  # type: ignore[arg-type]
        continue
      try:
        results_list[i] = job.result(cleanup=cleanup)
      except Exception as exc:
        if return_exceptions:
          results_list[i] = exc
        failures.append(job)

    with self._lock:
      self._cached_failures = [f for f in failures if f is not None]
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
        failures.append(job)

    for idx in sorted(self._submission_errors):
      exc = self._submission_errors[idx]
      if return_exceptions:
        results_list.append(exc)
      failures.append(None)  # type: ignore[arg-type]

    with self._lock:
      self._cached_failures = [f for f in failures if f is not None]
    return results_list, failures

  def failures(self) -> list[JobHandle]:
    """Return handles for jobs that failed.

    Only includes jobs whose status is `FAILED`.  Jobs that are
    `NOT_FOUND` (e.g. after cleanup) are excluded because the
    status is ambiguous — use `statuses()` for finer control.

    After `results()` has been called, this returns the cached
    failure list from that collection pass, so it remains accurate
    even if cleanup has deleted K8s resources.

    See Also:
      `submission_failures`: returns per-input errors for inputs
      that failed at submission time (`jobs[idx]` is `None`).
    """
    with self._lock:
      if self._cached_failures is not None:
        return list(self._cached_failures)
    return [
      job
      for job in self.jobs
      if job is not None and job.status() == JobStatus.FAILED
    ]

  @property
  def submission_failures(self) -> dict[int, Exception]:
    """Return a copy of per-input submission errors (index -> exception).

    These are inputs where the submission itself failed (e.g. validation
    error, network error).  The corresponding `jobs[idx]` slot is
    `None`.  These errors are included in `results()` output but are
    **not** reflected by `failures()` which only inspects live job
    statuses.
    """
    with self._lock:
      return dict(self._submission_errors)

  def cancel(self) -> None:
    """Cancel all non-terminal jobs in the collection."""
    for job in self.jobs:
      if job is None:
        continue
      try:
        if job.status() not in _TERMINAL_STATUSES:
          job.cancel()
      except RuntimeError:
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
      except (RuntimeError, google_exceptions.GoogleAPIError):
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
        except google_exceptions.GoogleAPIError:
          logging.warning(
            "Failed to clean up manifest for group %s", self.group_id
          )


def _load_child_handle(
  bucket_name: str,
  child: dict,
  total_expected: int,
  project: str,
) -> tuple[int, JobHandle] | None:
  """Download and reconstruct a single child handle.

  Returns `(group_index, handle)` on success, or `None` if the
  child has an invalid index or the download fails.
  """
  idx = child["group_index"]
  if not isinstance(idx, int) or idx < 0 or idx >= total_expected:
    logging.warning(
      "Invalid child index %r (total_expected=%d); skipping",
      idx,
      total_expected,
    )
    return None
  try:
    payload = storage.download_handle(
      bucket_name, child["job_id"], project=project
    )
    return idx, JobHandle.from_dict(payload)
  except (google_exceptions.GoogleAPIError, KeyError, ValueError):
    logging.warning(
      "Could not load handle for child job %s (index %d); skipping",
      child["job_id"],
      idx,
    )
    return None


def _manifest_poll_loop(
  handle: BatchHandle,
  bucket_name: str,
  group_id: str,
  project: str,
  total_expected: int,
  poll_interval: float,
  timeout: float | None,
) -> None:
  """Poll GCS manifest until all children appear, then set `_submission_complete`.

  Used by `attach_batch()` when the manifest shows fewer children
  than `total_expected`, indicating the original `map()` is still
  submitting.
  """
  deadline = None if timeout is None else time.monotonic() + timeout

  try:
    while True:
      if deadline is not None and time.monotonic() >= deadline:
        logging.warning(
          "Timed out polling manifest for batch %s (%d/%d children)",
          group_id,
          sum(1 for j in handle.jobs if j is not None),
          total_expected,
        )
        break

      time.sleep(poll_interval)

      try:
        manifest = storage.download_manifest(
          bucket_name, group_id, project=project
        )
      except google_exceptions.GoogleAPIError:
        logging.warning("Failed to poll manifest for batch %s", group_id)
        continue

      for child in manifest.get("children", []):
        idx = child.get("group_index")
        if not isinstance(idx, int) or idx < 0 or idx >= total_expected:
          continue
        with handle._lock:
          if handle.jobs[idx] is not None:
            continue
        result = _load_child_handle(bucket_name, child, total_expected, project)
        if result is not None:
          loaded_idx, job_handle = result
          with handle._lock:
            handle.jobs[loaded_idx] = job_handle

      loaded = sum(1 for j in handle.jobs if j is not None)
      if loaded >= total_expected:
        break
  finally:
    handle._submission_complete.set()


def _cancel_active(handle: BatchHandle, active_indices: set[int]) -> None:
  """Best-effort cancel of all active jobs."""
  for idx in list(active_indices):
    job = handle.jobs[idx]
    if job is None:
      continue
    try:
      job.cancel()
    except RuntimeError:
      logging.warning("Failed to cancel job at index %d", idx)


@dataclass
class _SubmissionState:
  """Groups the mutable state tracked by the submission loop.

  Provides named predicates so the main loop reads as a clear
  sequence of phases rather than a tangle of flags and counters.
  """

  handle: BatchHandle
  manifest: dict
  submit_fn: Any
  inputs: list
  input_mode: str
  max_concurrent: int | None
  max_attempts: int
  fail_fast: bool
  cancel_running_on_fail: bool

  attempt_counts: list[int] = field(init=False)
  pending: collections.deque = field(init=False)
  active: set[int] = field(default_factory=set, init=False)
  stop_launching: bool = field(default=False, init=False)

  def __post_init__(self):
    self.attempt_counts = [0] * len(self.inputs)
    self.pending = collections.deque(range(len(self.inputs)))

  @property
  def has_work(self) -> bool:
    """True while jobs remain to be submitted or are still running."""
    return bool(self.pending) or bool(self.active)

  def can_submit_more(self) -> bool:
    """True when the next pending job is allowed to launch."""
    if not self.pending or self.stop_launching:
      return False
    return self.max_concurrent is None or len(self.active) < self.max_concurrent

  def needs_active_polling(self) -> bool:
    """True when the loop must poll active jobs itself.

    When all jobs are submitted with no retries and `fail_fast`
    is off, the caller uses `wait()`/`results()` to observe
    terminal states, so the submission loop can exit early.
    """
    if not self.active:
      return False
    return bool(self.pending) or self.max_attempts > 1 or self.fail_fast

  def trigger_fail_fast(self) -> None:
    """Stop launching new jobs and optionally cancel siblings."""
    self.stop_launching = True
    if self.cancel_running_on_fail:
      _cancel_active(self.handle, self.active)


def _submit_available(state: _SubmissionState) -> None:
  """Submit pending jobs up to the concurrency limit.

  On per-input errors the exception is recorded in
  `handle._submission_errors` and, when `fail_fast` is set,
  `trigger_fail_fast` is called.
  """
  handle = state.handle

  while state.can_submit_more():
    idx = state.pending.popleft()
    state.attempt_counts[idx] += 1

    # attempt submission
    try:
      job_handle = call_with_input(
        state.submit_fn, state.inputs[idx], state.input_mode
      )
    except Exception as exc:
      logging.error("Submission failed for index %d: %s", idx, exc)
      with handle._lock:
        handle._submission_errors[idx] = exc
      if state.fail_fast:
        state.trigger_fail_fast()
      continue

    # tag with group metadata and persist
    job_handle.group_id = handle.group_id
    job_handle.group_kind = state.manifest["group_kind"]
    job_handle.group_index = idx

    try:
      storage.upload_handle(
        job_handle.bucket_name,
        job_handle.job_id,
        job_handle.to_dict(),
        project=job_handle.project,
      )
    except google_exceptions.GoogleAPIError:
      logging.warning(
        "Failed to re-upload handle with group fields for %s",
        job_handle.job_id,
      )

    # register in handle and manifest
    with handle._lock:
      handle.jobs[idx] = job_handle
    state.active.add(idx)

    append_child_to_manifest(
      state.manifest, idx, job_handle.job_id, state.attempt_counts[idx]
    )
    try:
      storage.upload_manifest(
        handle._bucket_name,
        handle.group_id,
        state.manifest,
        project=handle._project,
      )
    except google_exceptions.GoogleAPIError:
      logging.warning(
        "Failed to update manifest after submitting index %d", idx
      )

  if state.stop_launching:
    state.pending.clear()


def _poll_and_handle_terminal(state: _SubmissionState) -> None:
  """Poll active jobs for terminal states; retry or trigger fail_fast."""
  handle = state.handle

  # Collect all newly-terminal jobs in one pass.
  newly_terminal: list[tuple[int, JobStatus, JobHandle]] = []
  for idx in list(state.active):
    job = handle.jobs[idx]
    if job is None:
      continue
    try:
      status = job.status()
      if status in _TERMINAL_STATUSES:
        newly_terminal.append((idx, status, job))
    except (RuntimeError, google_exceptions.GoogleAPIError):
      logging.warning("Failed to poll status for index %d", idx)

  for idx, status, job in newly_terminal:
    state.active.discard(idx)

    if status not in (JobStatus.FAILED, JobStatus.NOT_FOUND):
      continue

    if state.attempt_counts[idx] < state.max_attempts:
      # Retry: clean up previous attempt's K8s resources and re-queue.
      try:
        job.cleanup(k8s=True, gcs=False)
      except RuntimeError:
        logging.warning("Failed to clean up before retry for index %d", idx)
      state.pending.append(idx)
    elif state.fail_fast:
      state.trigger_fail_fast()


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
  thread (`max_concurrent=None` and `retries=0`) or in a background
  thread otherwise.

  Each iteration follows three phases:

  1. **Submit** — launch pending jobs up to the concurrency limit.
  2. **Poll**  — check active jobs for terminal states, retry or
     trigger `fail_fast` as needed.
  3. **Sleep** — back off before the next poll cycle.
  """
  state = _SubmissionState(
    handle=handle,
    manifest=manifest,
    submit_fn=submit_fn,
    inputs=inputs,
    input_mode=input_mode,
    max_concurrent=max_concurrent,
    max_attempts=1 + retries,
    fail_fast=fail_fast,
    cancel_running_on_fail=cancel_running_on_fail,
  )

  try:
    while state.has_work:
      _submit_available(state)

      if not state.needs_active_polling():
        break

      _poll_and_handle_terminal(state)

      if state.has_work:
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


def attach_batch(
  group_id: str,
  project: str | None = None,
  cluster: str | None = None,
  poll_interval: float = _MANIFEST_POLL_INTERVAL,
  poll_timeout: float | None = None,
) -> BatchHandle:
  """Reattach to an existing batch collection by *group_id*.

  Downloads the group manifest from GCS, reconstructs `JobHandle`
  objects for each child, and returns a fully usable `BatchHandle`.

  If the manifest has fewer children than `total_expected` (i.e.
  the original `map()` is still submitting), the returned handle
  polls the manifest in a background thread until all children
  appear or *poll_timeout* is reached.

  Args:
    group_id: The collection identifier (e.g. `"grp-a1b2c3d4"`).
    project: GCP project (uses default when *None*).
    cluster: GKE cluster name (uses default when *None*).
    poll_interval: Seconds between manifest polls when the batch
      is partially submitted.
    poll_timeout: Maximum seconds to poll for remaining children.
      `None` means poll indefinitely.

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
    result = _load_child_handle(
      bucket_name, child, total_expected, resolved_project
    )
    if result is not None:
      idx, job_handle = result
      jobs[idx] = job_handle

  handle = BatchHandle(
    group_id=manifest["group_id"],
    name=manifest.get("group_name"),
    tags=manifest.get("tags", {}),
    jobs=jobs,
    _bucket_name=bucket_name,
    _project=resolved_project,
  )

  loaded = sum(1 for j in jobs if j is not None)
  if loaded >= total_expected:
    # All children present — mark complete immediately.
    handle._submission_complete.set()
  else:
    logging.warning(
      "Batch %s was partially submitted: %d of %d expected jobs. "
      "Polling manifest for remaining children.",
      group_id,
      loaded,
      total_expected,
    )
    thread = threading.Thread(
      target=_manifest_poll_loop,
      kwargs={
        "handle": handle,
        "bucket_name": bucket_name,
        "group_id": group_id,
        "project": resolved_project,
        "total_expected": total_expected,
        "poll_interval": poll_interval,
        "timeout": poll_timeout,
      },
      daemon=True,
    )
    thread.start()

  return handle
