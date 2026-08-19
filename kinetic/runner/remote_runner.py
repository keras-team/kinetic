#!/usr/bin/env python3
"""Remote execution entrypoint for kinetic.

This script runs on the remote TPU/GPU and executes the user's function.
Artifacts are downloaded from and uploaded to Cloud Storage (GCS).
"""

import argparse
import atexit
import collections
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
import zipfile

import cloudpickle
from absl import logging
from google.cloud import exceptions as cloud_exceptions
from google.cloud import storage
from google.cloud.storage import transfer_manager

_DOWNLOAD_BATCH_SIZE = 10000

# Reserved path inside context.zip carrying the client's packaging plan: a
# small JSON dict newer clients write describing how to reconstruct their
# environment on the pod — project-relative sys.path entries to insert
# ("sys_path_rel") and the extracted directory to chdir into
# ("client_cwd_rel"). Archives without it (older clients) get the legacy
# behavior: workspace root on sys.path only, no chdir.
_PLAN_ARCHIVE_PATH = os.path.join(".kinetic", "plan.json")

# ZipInfo.create_system value meaning "Unix"; only then do the high bits of
# external_attr hold a POSIX mode.
_ZIP_CREATE_SYSTEM_UNIX = 3

# Cap on the repr stored in a result payload when the real result could
# not be serialized.
_RESULT_REPR_LIMIT = 20000

# Sentinel blob name written by the leader once it has finished
# waiting for a debugger client and is about to call the user
# function. Workers poll for this to stay in sync with the leader.
_LEADER_READY_SENTINEL = ".leader_ready"

# Extra seconds workers wait beyond the leader's attach timeout,
# to cover GCS write latency and any processing after the leader's
# wait_for_client() returns.
_WORKER_WAIT_BUFFER_SECONDS = 60

# Environment variables carrying this pod's process index within a
# multi-host job, in priority order. The Pathways/LWS pod spec sets
# TPU_WORKER_ID from LWS_WORKER_INDEX; LWS_WORKER_INDEX itself is read as
# a fallback because the "$(VAR)" substitution only expands when the
# webhook-injected variable precedes it in the container's env list.
# Single-host jobs set neither and are always their own leader.
_HOST_INDEX_ENV_VARS = ("TPU_WORKER_ID", "LWS_WORKER_INDEX")

# Blob name a non-leader host writes its failure payload to, beside the
# leader's result.pkl. Kept in sync with
# ``kinetic.utils.storage.worker_result_blob_name`` on the client side —
# this module ships standalone inside the container and cannot import it.
_WORKER_RESULT_TEMPLATE = "result-worker-{index}.pkl"


def _verify_sha256(path, expected_hash, name):
  """Verify the SHA-256 hash of a downloaded file."""
  hasher = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(65536), b""):
      hasher.update(chunk)
  actual_hash = hasher.hexdigest()
  if actual_hash != expected_hash:
    raise RuntimeError(
      f"Security verification failed: {name} SHA-256 hash mismatch. "
      f"Expected {expected_hash}, got {actual_hash}. "
      f"The file may have been tampered with."
    )


def _host_index():
  """Return this pod's process index within the job (0 for the leader).

  Every pod of a multi-host job is launched with the same command, so
  the index is the only thing distinguishing them. A missing variable
  means a single-host job, whose one pod is its own leader; a variable
  set to something that is not a host index is reported and ignored
  rather than silently demoting the leader.
  """
  for name in _HOST_INDEX_ENV_VARS:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
      continue
    try:
      index = int(raw)
    except ValueError:
      logging.warning(
        "Ignoring %s=%r: expected an integer host index.", name, raw
      )
      continue
    if index < 0:
      logging.warning(
        "Ignoring %s=%r: host index cannot be negative.", name, raw
      )
      continue
    return index
  return 0


def _worker_result_uri(result_gcs, host_index):
  """Return the per-host failure payload URI beside *result_gcs*.

  Args:
      result_gcs: The job-wide result URI every pod is given.
      host_index: This pod's process index (never 0 — the leader writes
          *result_gcs* itself).

  Returns:
      A GCS URI in the same job prefix, e.g.
      ``gs://bucket/job-a1/result-worker-3.pkl``.
  """
  prefix, separator, _ = result_gcs.rpartition("/")
  name = _WORKER_RESULT_TEMPLATE.format(index=host_index)
  return f"{prefix}{separator}{name}" if separator else name


def main():
  """Main entry point for remote execution.

  Usage: python remote_runner.py <context_gcs> <payload_gcs> <result_gcs> [requirements_gcs]
  """
  parser = argparse.ArgumentParser(description="Kinetic remote runner.")
  parser.add_argument("positional", nargs="*", help="Legacy positional args")
  parser.add_argument("--context-gcs", help="GCS URI for context.zip")
  parser.add_argument("--payload-gcs", help="GCS URI for payload.pkl")
  parser.add_argument("--result-gcs", help="GCS URI for result.pkl")
  parser.add_argument("--requirements-gcs", help="GCS URI for requirements.txt")
  parser.add_argument(
    "--payload-sha256", help="Expected SHA-256 hash of payload"
  )
  parser.add_argument(
    "--context-sha256", help="Expected SHA-256 hash of context"
  )

  args_parsed, _ = parser.parse_known_args()

  # Fallback to positional arguments for backward compatibility
  if args_parsed.positional and len(args_parsed.positional) >= 3:
    context_gcs = args_parsed.positional[0]
    payload_gcs = args_parsed.positional[1]
    result_gcs = args_parsed.positional[2]
    requirements_gcs = (
      args_parsed.positional[3] if len(args_parsed.positional) > 3 else None
    )
  else:
    context_gcs = args_parsed.context_gcs
    payload_gcs = args_parsed.payload_gcs
    result_gcs = args_parsed.result_gcs
    requirements_gcs = args_parsed.requirements_gcs

  if not (context_gcs and payload_gcs and result_gcs):
    logging.error("Missing required arguments for artifacts.")
    sys.exit(1)

  host_index = _host_index()
  is_leader = host_index == 0
  # Only the leader writes the result.pkl the client reads back, so the
  # returned value is always process 0's — not whichever host happened
  # to upload last. Every other host reports a failure to its own blob,
  # which the client aggregates to surface the real exception.
  host_result_gcs = (
    result_gcs if is_leader else _worker_result_uri(result_gcs, host_index)
  )

  logging.info("Starting remote execution (host index %d)", host_index)

  # Create secure temp directory and register cleanup
  temp_dir = tempfile.mkdtemp(prefix="kinetic-run-")
  atexit.register(shutil.rmtree, temp_dir, ignore_errors=True)

  # Define local paths
  context_path = os.path.join(temp_dir, "context.zip")
  payload_path = os.path.join(temp_dir, "payload.pkl")
  result_path = os.path.join(temp_dir, "result.pkl")
  workspace_dir = os.path.join(temp_dir, "workspace")
  data_dir = os.path.join(temp_dir, "data")

  storage_client = None
  debugger_attached = False
  phase = "startup"

  try:
    phase = "artifact download"
    storage_client = storage.Client()

    # Download artifacts from Cloud Storage
    logging.info("Downloading artifacts...")
    _download_from_gcs(storage_client, context_gcs, context_path)
    _download_from_gcs(storage_client, payload_gcs, payload_path)

    phase = "artifact verification"
    if args_parsed.payload_sha256:
      logging.info("Verifying payload SHA-256...")
      _verify_sha256(payload_path, args_parsed.payload_sha256, "payload.pkl")

    if args_parsed.context_sha256:
      logging.info("Verifying context SHA-256...")
      _verify_sha256(context_path, args_parsed.context_sha256, "context.zip")

    # Install user requirements at startup (prebuilt image mode)
    if requirements_gcs:
      phase = "requirements install"
      _install_requirements(storage_client, requirements_gcs, temp_dir)

    phase = "context extract"
    if os.path.exists(workspace_dir):
      shutil.rmtree(workspace_dir)
    os.makedirs(workspace_dir)

    _extract_context(context_path, workspace_dir)
    workspace_paths = _apply_workspace_plan(workspace_dir)

    phase = "payload unpickle"
    logging.info("Loading function payload")
    with open(payload_path, "rb") as f:
      payload = cloudpickle.load(f)

    phase = "environment setup"
    _warn_on_fingerprint_skew(payload.get("client_fingerprint"))

    # Reconstruct client path parity for debugpy exact file mappings
    working_dir_client = payload.get("working_dir")
    if working_dir_client and not os.path.exists(working_dir_client):
      try:
        os.makedirs(os.path.dirname(working_dir_client), exist_ok=True)
        os.symlink(workspace_dir, working_dir_client)
      except Exception as e:
        logging.warning("Failed to symlink client working dir: %s", e)

    func = payload["func"]
    args = payload["args"]
    kwargs = payload["kwargs"]
    env_vars = payload.get("env_vars", {})
    if env_vars:
      logging.info("Setting %d environment variables", len(env_vars))
      os.environ.update(env_vars)

    # Resolve Data references
    phase = "data resolve"
    volumes = payload.get("volumes", [])
    _preimport_hf_dependencies(args, kwargs, volumes, workspace_paths)
    if volumes:
      resolve_volumes(volumes, storage_client)
    if _payload_has_data_refs(payload, args, kwargs):
      args, kwargs = resolve_data_refs(args, kwargs, storage_client, data_dir)
    else:
      logging.info("No data references in payload; skipping resolution")

    phase = "debugger setup"
    # Start debugpy server if debug mode is enabled
    is_debug = os.environ.get("KINETIC_DEBUG") == "1"
    is_debug_worker = os.environ.get("KINETIC_DEBUG_WAIT_LEADER") == "1"
    if is_debug:
      _install_debugger()
      # Port is propagated from kinetic.debug.DEBUGPY_PORT via the pod spec
      # so there's a single source of truth. Fall back to 5678 (debugpy's
      # default and VS Code's auto-fill) if the env var is missing.
      debug_port = int(os.environ.get("KINETIC_DEBUG_PORT", 5678))
      debugger_attached = _start_debug_server(debug_port)
      # Signal workers (if any) that the leader is about to call the
      # user function, so they can proceed without racing ahead and
      # hanging on the distributed runtime.
      _upload_leader_ready_sentinel()
    elif is_debug_worker:
      # Pathways worker pod — wait for leader's sentinel before running.
      _wait_for_leader_ready_sentinel()
  except BaseException as setup_err:  # noqa: BLE001 - always report a result
    setup_traceback = traceback.format_exc()
    logging.error("kinetic %s failed: %s", phase, setup_err)
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    hint = None
    if phase == "payload unpickle":
      hint = _unpickle_failure_hint(setup_err, payload_path)
    _write_failure_result(
      storage_client,
      result_path,
      host_result_gcs,
      phase,
      setup_err,
      setup_traceback,
      hint,
      host_index=host_index,
    )
    _exit_process(1)
    return

  # Execute function and capture result
  logging.info("Executing %s()", getattr(func, "__name__", repr(func)))
  result = None
  exception = None
  remote_traceback = None

  try:
    if debugger_attached:
      import debugpy

      # === KINETIC DEBUG ===
      # The debugger will pause on the next line.
      # Press Step Into (F11) to enter your function, or
      # Step Over (F10) to run it without stepping.
      debugpy.breakpoint()
    result = func(*args, **kwargs)
    logging.info("Function completed successfully")
  except SystemExit as e:
    # sys.exit(0)/sys.exit() is an ordinary successful early return.
    if e.code is None or e.code == 0:
      logging.info(
        "Function called sys.exit(%r); treating it as success.", e.code
      )
    else:
      remote_traceback = traceback.format_exc()
      logging.error("Function called sys.exit(%r)", e.code)
      traceback.print_exc()
      sys.stdout.flush()
      sys.stderr.flush()
      exception = RuntimeError(f"function called sys.exit({e.code!r})")
  except BaseException as e:
    remote_traceback = traceback.format_exc()
    logging.error("%s: %s", type(e).__name__, e)
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    if isinstance(e, Exception):
      exception = e
    else:
      exception = RuntimeError(f"{type(e).__name__}: {e}")

  if exception is None and not is_leader:
    # The leader's payload is the job's result, so a non-leader's return
    # value is discarded — deliberately without serializing it, which
    # keeps a large per-host return value from being pickled and
    # uploaded once per host for nothing.
    logging.info(
      "Host %d completed successfully; the leader owns the result payload.",
      host_index,
    )
    _exit_process(0)
    return

  # Serialize result or exception
  result_payload = {
    "success": exception is None,
    "result": result if exception is None else None,
    "exception": exception,
    "traceback": remote_traceback,
    "phase": "execute",
    "host_index": host_index,
  }
  serialization_failed = _dump_result_payload(
    result_path, result_payload, result, remote_traceback
  )

  try:
    logging.info("Uploading result to %s", host_result_gcs)
    _upload_to_gcs(storage_client, result_path, host_result_gcs)
  except BaseException as upload_err:  # noqa: BLE001 - report, then exit
    logging.error("Failed to upload result: %s", upload_err)
    traceback.print_exc()
    _exit_process(1)
    return

  logging.info("Execution complete")
  _exit_process(0 if exception is None and not serialization_failed else 1)


def _extract_context(zip_path, dest_dir):
  """Extract context.zip, restoring POSIX permission bits.

  ``ZipFile.extractall`` drops the stored mode, so executable helper
  scripts shipped in the context arrive non-executable.

  Args:
      zip_path: Path to the downloaded context archive.
      dest_dir: Directory to extract into.
  """
  with zipfile.ZipFile(zip_path, "r") as zip_ref:
    for info in zip_ref.infolist():
      extracted = zip_ref.extract(info, dest_dir)
      # Only Unix-created archives store a POSIX mode in the high bits of
      # external_attr; every other creator puts unrelated data there, so
      # reading it as a mode would apply garbage permissions.
      if info.create_system != _ZIP_CREATE_SYSTEM_UNIX:
        continue
      # Masked to the rwx bits: never honor setuid/setgid/sticky from an
      # archive, whoever built it.
      mode = (info.external_attr >> 16) & 0o777
      if not mode:
        continue
      try:
        os.chmod(extracted, mode)
      except OSError as e:
        logging.warning("Could not restore permissions on %s: %s", extracted, e)


def _apply_workspace_plan(workspace_dir):
  """Reconstruct the client's import environment inside the workspace.

  Reads the reserved ``.kinetic/plan.json`` entry written by the client
  packager. When it is absent (older client) the legacy behavior is kept
  exactly: the workspace root is prepended to ``sys.path`` and the
  working directory is left alone.

  Args:
      workspace_dir: Directory the context archive was extracted into.

  Returns:
      The list of absolute paths inserted into ``sys.path``.
  """
  plan_path = os.path.join(workspace_dir, _PLAN_ARCHIVE_PATH)
  plan = None
  if os.path.exists(plan_path):
    try:
      with open(plan_path, encoding="utf-8") as f:
        plan = json.load(f)
    except Exception as e:
      logging.warning("Could not read %s: %s", plan_path, e)
      plan = None

  if not isinstance(plan, dict):
    sys.path.insert(0, workspace_dir)
    return [workspace_dir]

  rel_entries = _plan_sys_path_entries(plan.get("sys_path_rel"))
  if "" not in rel_entries:
    rel_entries = [""] + rel_entries

  inserted = []
  for rel in rel_entries:
    entry = _workspace_subpath(workspace_dir, rel)
    if entry is None or not os.path.isdir(entry):
      logging.warning("Skipping sys.path entry %r: not present in context", rel)
      continue
    if entry not in inserted:
      inserted.append(entry)
  # Insert last-to-first so the plan's order survives at the front.
  for entry in reversed(inserted):
    sys.path.insert(0, entry)
  logging.info("Reconstructed sys.path entries: %s", inserted)

  target = workspace_dir
  cwd_rel = plan.get("client_cwd_rel")
  # "" means the client ran from the package root itself: the workspace
  # root is already the right target. Only non-empty paths need resolving.
  if isinstance(cwd_rel, str):
    if cwd_rel:
      candidate = _workspace_subpath(workspace_dir, cwd_rel)
      if candidate is not None and os.path.isdir(candidate):
        target = candidate
      else:
        logging.warning(
          "Client working directory %r is not present in the context; "
          "using the package root instead.",
          cwd_rel,
        )
  elif cwd_rel is not None:
    logging.warning(
      "Ignoring plan client_cwd_rel %r: expected a string; "
      "using the package root instead.",
      cwd_rel,
    )
  os.chdir(target)
  logging.info("Working directory: %s", target)
  return inserted or [workspace_dir]


def _plan_sys_path_entries(raw):
  """Sanitize the plan's ``sys_path_rel`` into a list of relative strings.

  The plan is optional metadata: malformed content degrades toward the
  legacy behavior (workspace root only) with a warning rather than
  failing a job that would otherwise run.

  Args:
      raw: The ``sys_path_rel`` value as read from ``plan.json``.

  Returns:
      The usable relative entries, in the order the plan listed them.
  """
  if raw is None:
    return []
  if not isinstance(raw, list):
    logging.warning(
      "Ignoring plan sys_path_rel: expected a list of strings, got %s (%r).",
      type(raw).__name__,
      raw,
    )
    return []
  entries = [rel for rel in raw if isinstance(rel, str)]
  if len(entries) != len(raw):
    logging.warning(
      "Dropping non-string plan sys_path_rel entries: %s.",
      [rel for rel in raw if not isinstance(rel, str)],
    )
  return entries


def _workspace_subpath(workspace_dir, rel):
  """Resolve *rel* under *workspace_dir*, rejecting escapes."""
  if rel in ("", ".", None):
    return workspace_dir
  candidate = os.path.normpath(os.path.join(workspace_dir, rel))
  if candidate != workspace_dir and not candidate.startswith(
    workspace_dir + os.sep
  ):
    logging.warning("Ignoring plan path %r: escapes the workspace", rel)
    return None
  return candidate


def _exit_process(exit_code):
  """Exit the process, bypassing a shutdown that would hang the pod.

  CPython joins non-daemon threads and runs atexit hooks at interpreter
  shutdown; a stray worker thread left behind by the user function keeps
  the pod alive forever even though the result is already in GCS.
  """
  sys.stdout.flush()
  sys.stderr.flush()
  lingering = [
    t
    for t in threading.enumerate()
    if t is not threading.current_thread() and t.is_alive() and not t.daemon
  ]
  if lingering:
    logging.warning(
      "Non-daemon threads still alive after the result was uploaded: %s. "
      "Exiting immediately so the pod does not hang; start background "
      "threads with daemon=True to shut down cleanly.",
      ", ".join(t.name for t in lingering),
    )
    os._exit(exit_code)
  sys.exit(exit_code)


def _dump_result_payload(result_path, result_payload, result, remote_traceback):
  """Write the result payload, degrading gracefully when it cannot pickle.

  Args:
      result_path: Local destination for the pickled payload.
      result_payload: The payload to serialize.
      result: The user function's return value (for the repr fallback).
      remote_traceback: Traceback of a user-function exception, if any.

  Returns:
      True when the real payload could not be serialized.
  """
  try:
    with open(result_path, "wb") as f:
      cloudpickle.dump(result_payload, f)
    return False
  except BaseException as serialize_err:  # noqa: BLE001 - user __reduce__
    logging.error("Failed to serialize result: %s", serialize_err)
    fallback_payload = {
      "success": False,
      "result": None,
      "exception": RuntimeError(
        f"Result serialization failed: {serialize_err}"
      ),
      "traceback": remote_traceback,
      "phase": "result serialization",
      "serialization_failed": True,
      "result_repr": _safe_repr(result),
      "host_index": result_payload.get("host_index", 0),
    }
    try:
      with open(result_path, "wb") as f:
        cloudpickle.dump(fallback_payload, f)
    except BaseException as fallback_err:  # noqa: BLE001 - last resort
      logging.error("Fallback result serialization failed: %s", fallback_err)
      with open(result_path, "wb") as f:
        pickle.dump(fallback_payload, f)
    return True


def _safe_repr(obj):
  """repr() that never raises, truncated to a loggable size."""
  try:
    text = repr(obj)
  except BaseException as e:  # noqa: BLE001 - user __repr__
    return f"<unreprable {type(obj).__name__}: {e}>"
  return text[:_RESULT_REPR_LIMIT]


def _write_failure_result(
  storage_client,
  result_path,
  result_gcs,
  phase,
  exc,
  tb,
  hint=None,
  host_index=0,
):
  """Write and upload a result payload for a pre-execution failure.

  Without this the pod exits with no artifact at all and the client can
  only report "no result payload was found".

  Args:
      storage_client: Cloud Storage client, or None if it never came up.
      result_path: Local path to write the payload to.
      result_gcs: GCS URI to upload the payload to — this host's blob,
          which for a non-leader is its own ``result-worker-N.pkl``.
      phase: Runner phase that failed, e.g. ``"payload unpickle"``.
      exc: The exception that caused the failure.
      tb: Formatted traceback for the failure.
      hint: Optional extra diagnostics appended to the message.
      host_index: This pod's process index, recorded so the client can
          name the host that failed.
  """
  message = f"kinetic {phase} failed: {type(exc).__name__}: {exc}"
  if hint:
    message = f"{message}\n{hint}"
  failure_payload = {
    "success": False,
    "result": None,
    "exception": RuntimeError(message),
    "traceback": tb,
    "phase": phase,
    "host_index": host_index,
  }
  try:
    with open(result_path, "wb") as f:
      cloudpickle.dump(failure_payload, f)
    client = storage_client if storage_client is not None else storage.Client()
    _upload_to_gcs(client, result_path, result_gcs)
    logging.info("Uploaded failure result payload to %s", result_gcs)
  except BaseException as e:  # noqa: BLE001 - never mask the original error
    logging.error("Could not upload failure result payload: %s", e)


def _python_version_string():
  """The running interpreter's version as ``"X.Y.Z"``."""
  return ".".join(str(part) for part in sys.version_info[:3])


def _normalize_version(value):
  """Coerce a fingerprint version entry to a dotted string."""
  if value is None:
    return None
  if isinstance(value, (list, tuple)):
    return ".".join(str(part) for part in value)
  return str(value)


def _warn_on_fingerprint_skew(fingerprint):
  """Log a warning when the pod's runtime differs from the client's.

  A Python minor-version mismatch loads fine and then segfaults inside
  the user's function, so the log line is the only early signal.
  """
  if not isinstance(fingerprint, dict):
    return
  client_python = _normalize_version(fingerprint.get("python"))
  pod_python = _python_version_string()
  if client_python and _minor_version(client_python) != _minor_version(
    pod_python
  ):
    logging.warning(
      "Python version skew: client %s, pod %s. Pickled code objects are not "
      "portable across minor versions and may crash the interpreter.",
      client_python,
      pod_python,
    )
  client_cloudpickle = _normalize_version(fingerprint.get("cloudpickle"))
  pod_cloudpickle = getattr(cloudpickle, "__version__", "unknown")
  if client_cloudpickle and client_cloudpickle != pod_cloudpickle:
    logging.warning(
      "cloudpickle version skew: client %s, pod %s.",
      client_cloudpickle,
      pod_cloudpickle,
    )


def _minor_version(version_string):
  """The ``X.Y`` prefix of a dotted version string."""
  return ".".join(version_string.split(".")[:2])


def _read_client_fingerprint(payload_path):
  """Best-effort read of the payload fingerprint after a load failure.

  The payload could not be unpickled normally, so unknown globals are
  replaced with a stub: enough of the outer dict survives to recover the
  small ``client_fingerprint`` entry.
  """

  def _stub(*args, **kwargs):
    return None

  class _TolerantUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
      try:
        return super().find_class(module, name)
      except BaseException:  # noqa: BLE001 - diagnostics only
        return _stub

  try:
    with open(payload_path, "rb") as f:
      payload = _TolerantUnpickler(f).load()
  except BaseException:  # noqa: BLE001 - diagnostics only
    return None
  if isinstance(payload, dict):
    fingerprint = payload.get("client_fingerprint")
    if isinstance(fingerprint, dict):
      return fingerprint
  return None


def _unpickle_failure_hint(exc, payload_path):
  """Explain a payload load failure in terms of client/pod differences."""
  lines = []
  pod_python = _python_version_string()
  pod_cloudpickle = getattr(cloudpickle, "__version__", "unknown")
  fingerprint = _read_client_fingerprint(payload_path)

  if fingerprint:
    client_python = _normalize_version(fingerprint.get("python"))
    if client_python and _minor_version(client_python) != _minor_version(
      pod_python
    ):
      lines.append(
        f"client Python {client_python} / pod Python {pod_python} — "
        "mismatched minor versions break pickled code objects; use "
        "container_image=None (bundled mode) or a prebuilt image built for "
        "your interpreter."
      )
    client_cloudpickle = _normalize_version(fingerprint.get("cloudpickle"))
    if client_cloudpickle and client_cloudpickle != pod_cloudpickle:
      lines.append(
        f"client cloudpickle {client_cloudpickle} / pod cloudpickle "
        f"{pod_cloudpickle} — a pod cloudpickle older than the client's "
        "cannot read the payload; pin cloudpickle in your requirements to "
        "match the client."
      )
  else:
    lines.append(
      f"pod Python {pod_python}, pod cloudpickle {pod_cloudpickle} "
      "(the payload carries no client fingerprint)."
    )

  missing_module = getattr(exc, "name", None)
  if isinstance(exc, ModuleNotFoundError) and missing_module:
    lines.append(
      f"module {missing_module!r} is imported by your pickled function but is "
      "not installed in the image and was not shipped in the context "
      "directory."
    )
  return "\n".join(lines) if lines else None


def _install_requirements(storage_client, requirements_gcs, temp_dir):
  """Download and install user requirements via `uv pip install`.

  Used in prebuilt image mode where user dependencies are not baked
  into the container image.

  Args:
      storage_client: Cloud Storage client.
      requirements_gcs: GCS URI to the requirements.txt file.
      temp_dir: Secure temporary directory path.
  """
  requirements_path = os.path.join(temp_dir, "user_requirements.txt")
  _download_from_gcs(storage_client, requirements_gcs, requirements_path)

  if os.path.getsize(requirements_path) == 0:
    logging.info("No user requirements to install")
    return

  logging.info("Installing user requirements...")
  result = subprocess.run(
    ["uv", "pip", "install", "--system", "-r", requirements_path],
    capture_output=True,
    text=True,
  )
  if result.returncode != 0:
    raise RuntimeError(
      f"Failed to install requirements (exit {result.returncode}).\n"
      f"stderr:\n{result.stderr}"
    )
  logging.info("User requirements installed successfully")


def _upload_leader_ready_sentinel():
  """Write a GCS sentinel telling Pathways workers the leader is ready."""
  bucket_name = os.environ.get("GCS_BUCKET")
  job_id = os.environ.get("JOB_ID")
  if not bucket_name or not job_id:
    logging.warning(
      "GCS_BUCKET or JOB_ID not set; skipping leader-ready sentinel."
    )
    return
  try:
    blob = (
      storage.Client()
      .bucket(bucket_name)
      .blob(f"{job_id}/{_LEADER_READY_SENTINEL}")
    )
    blob.upload_from_string("")
    logging.info(
      "Published leader-ready sentinel to gs://%s/%s/%s",
      bucket_name,
      job_id,
      _LEADER_READY_SENTINEL,
    )
  except cloud_exceptions.GoogleCloudError as e:
    logging.warning("Failed to publish leader-ready sentinel: %s", e)


def _wait_for_leader_ready_sentinel():
  """Poll GCS until the leader signals readiness, or time out.

  Pathways worker pods call this before executing the user function.
  Without it, workers race ahead of a paused leader and hang trying
  to initialize JAX's distributed runtime.
  """
  bucket_name = os.environ.get("GCS_BUCKET")
  job_id = os.environ.get("JOB_ID")
  if not bucket_name or not job_id:
    logging.warning("GCS_BUCKET or JOB_ID not set; skipping leader-ready wait.")
    return

  # Wait slightly longer than the leader's attach timeout so we don't
  # fail the job due to normal GCS write latency at the deadline.
  leader_timeout = int(
    os.environ.get("KINETIC_DEBUG_WAIT_TIMEOUT", _DEBUG_WAIT_TIMEOUT_DEFAULT)
  )
  timeout = leader_timeout + _WORKER_WAIT_BUFFER_SECONDS
  poll_interval = 5

  logging.info(
    "[DEBUG-WORKER] Waiting up to %ds for leader-ready sentinel at "
    "gs://%s/%s/%s",
    timeout,
    bucket_name,
    job_id,
    _LEADER_READY_SENTINEL,
  )

  client = storage.Client()
  bucket = client.bucket(bucket_name)
  blob_name = f"{job_id}/{_LEADER_READY_SENTINEL}"

  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    try:
      if bucket.blob(blob_name).exists(client=client):
        logging.info("[DEBUG-WORKER] Leader is ready, proceeding.")
        return
    except cloud_exceptions.GoogleCloudError as e:
      logging.warning("Error polling leader-ready sentinel: %s", e)
    time.sleep(poll_interval)

  raise RuntimeError(
    f"Leader did not signal readiness within {timeout}s. The leader "
    "pod may have crashed before starting debugpy, or GCS may be "
    f"unreachable. Expected sentinel at "
    f"gs://{bucket_name}/{blob_name}."
  )


def _install_debugger():
  """Install debugpy via uv pip at pod startup."""
  logging.info("Installing debugpy...")
  result = subprocess.run(
    ["uv", "pip", "install", "--system", "debugpy"],
    capture_output=True,
    text=True,
  )
  if result.returncode != 0:
    raise RuntimeError(
      f"Failed to install debugpy (exit {result.returncode}).\n"
      f"stderr:\n{result.stderr}"
    )
  logging.info("debugpy installed successfully")


# Fallback if KINETIC_DEBUG_WAIT_TIMEOUT env var is not set.
# The pod spec normally sets the env var from kinetic.debug's
# resolve_debug_wait_timeout(), so the client and the pod wait out
# the same window. Keep this in sync with DEFAULT_DEBUG_WAIT_TIMEOUT
# there; the runner cannot import kinetic on the pod.
_DEBUG_WAIT_TIMEOUT_DEFAULT = 600


def _start_debug_server(port):
  """Start debugpy server and wait for client attachment.

  Waits up to ``KINETIC_DEBUG_WAIT_TIMEOUT`` seconds (default 600) for
  a debugger to attach. If no client connects in time, execution
  proceeds without the debugger so the pod doesn't hang indefinitely.

  Args:
      port: TCP port for debugpy to listen on.

  Returns:
      True if a debugger client attached, False if timed out.
  """
  import debugpy

  debugpy.listen(("0.0.0.0", port))

  try:
    # Signal readiness via a GCS sentinel so the local client can detect it.
    # Use env vars set by the pod spec rather than parsing sys.argv.
    bucket_name = os.environ.get("GCS_BUCKET")
    job_id = os.environ.get("JOB_ID")
    if not bucket_name or not job_id:
      logging.warning("GCS_BUCKET or JOB_ID not set; skipping debug sentinel.")
    else:
      blob = storage.Client().bucket(bucket_name).blob(f"{job_id}/.debug_ready")
      blob.upload_from_string("")
      logging.info(
        "Published debugpy GCS sentinel to gs://%s/%s/.debug_ready",
        bucket_name,
        job_id,
      )
  except cloud_exceptions.GoogleCloudError as e:
    logging.warning("Failed to publish debug readiness sentinel to GCS: %s", e)

  logging.info("[DEBUGPY] Ready \u2014 listening on 0.0.0.0:%d", port)

  timeout = int(
    os.environ.get("KINETIC_DEBUG_WAIT_TIMEOUT", _DEBUG_WAIT_TIMEOUT_DEFAULT)
  )
  logging.info("[DEBUGPY] Waiting up to %ds for debugger to attach...", timeout)

  # debugpy.wait_for_client() has no timeout parameter, so we use a
  # background thread + Event to implement one.
  attached = threading.Event()

  def _wait():
    debugpy.wait_for_client()
    attached.set()

  waiter = threading.Thread(target=_wait, daemon=True)
  waiter.start()

  if attached.wait(timeout=timeout):
    logging.info("[DEBUGPY] Debugger attached!")
    return True

  logging.warning(
    "[DEBUGPY] No debugger attached after %ds \u2014 proceeding without debugger.",
    timeout,
  )
  return False


def resolve_volumes(
  volume_refs: list[dict], storage_client: storage.Client
) -> None:
  """Download volume data to their specified mount paths.

  Volumes with `fuse=True` are already mounted via the GCS FUSE CSI
  driver and are skipped.
  """
  for ref in volume_refs:
    mount_path = ref["mount_path"]
    if ref.get("fuse"):
      logging.info(
        "Skipping download for FUSE-mounted volume: %s -> %s",
        ref["uri"],
        mount_path,
      )
      continue
    logging.info("Resolving volume: %s -> %s", ref["uri"], mount_path)
    _download_data(ref, mount_path, storage_client)


def _resolve_fuse_single_file(mount_path: str) -> str | None:
  """Find the single data file inside a FUSE mount directory.

  GCS FUSE mounts directories, not individual files.  For single-file
  data refs the mount is scoped to the hash directory containing the
  file, so a flat listing is sufficient.

  Returns the file path, or `None` if no data file is found.
  """
  try:
    entries = os.listdir(mount_path)
  except OSError:
    return None
  if entries:
    return os.path.join(mount_path, entries[0])
  return None


def _is_data_ref(obj) -> bool:
  """True when *obj* is a data-ref dict written by the client packager."""
  return isinstance(obj, dict) and bool(obj.get("__data_ref__"))


def _iter_data_refs(obj):
  """Yield data-ref dicts reachable through plain containers.

  Iterative so that deeply nested arguments cannot exhaust the stack, and
  identity-guarded so that self-referential arguments terminate.
  """
  stack = [obj]
  seen = set()
  while stack:
    current = stack.pop()
    if isinstance(current, dict):
      if _is_data_ref(current):
        yield current
        continue
      if id(current) in seen:
        continue
      seen.add(id(current))
      stack.extend(current.keys())
      stack.extend(current.values())
    elif isinstance(current, (list, tuple, set, frozenset)):
      if id(current) in seen:
        continue
      seen.add(id(current))
      stack.extend(current)


def _contains_data_ref(obj) -> bool:
  """Cheap scan used when the payload predates ``has_data_refs``."""
  # Ref dicts are never empty, so any() short-circuits on the first one.
  return any(_iter_data_refs(obj))


def _payload_has_data_refs(payload: dict, args: tuple, kwargs: dict) -> bool:
  """Decide whether the argument walk is needed at all.

  Newer clients declare ``has_data_refs``; older payloads fall back to a
  scan. Skipping the walk keeps argument types and object identity fully
  intact for the overwhelming majority of jobs.
  """
  declared = payload.get("has_data_refs")
  if declared is not None:
    return bool(declared)
  return _contains_data_ref(args) or _contains_data_ref(kwargs)


def _preimport_hf_dependencies(args, kwargs, volumes, workspace_paths) -> None:
  """Import ``datasets`` before the workspace can shadow it.

  The extracted project sits at the front of ``sys.path``, so a project
  file named ``datasets.py`` hijacks the runner's own import. The refs
  are only known after the payload is unpickled (which needs the
  workspace on the path), so the workspace entries are lifted off
  ``sys.path`` just for this import.
  """
  if "datasets" in sys.modules:
    return
  refs = list(volumes or [])
  refs.extend(_iter_data_refs(args))
  refs.extend(_iter_data_refs(kwargs))
  if not any(str(ref.get("uri", "")).startswith("hf://") for ref in refs):
    return

  # Absolutized on both sides: sys.path can hold relative entries ("" or
  # ".") which, after the plan's chdir, resolve inside the workspace and
  # would otherwise survive the filter and shadow the import anyway.
  roots = {os.path.abspath(p) for p in (workspace_paths or ())}
  saved_path = sys.path[:]
  sys.path[:] = [p for p in sys.path if not _is_under_any(p, roots)]
  try:
    import datasets  # noqa: F401
  except Exception as e:
    logging.warning("Could not pre-import 'datasets': %s", e)
  finally:
    sys.path[:] = saved_path


def _is_under_any(entry, roots):
  """True when a ``sys.path`` entry resolves to or into one of *roots*.

  Args:
      entry: A ``sys.path`` entry; non-string entries can never shadow a
          filesystem import and are reported as outside.
      roots: Absolute directory paths.

  Returns:
      True when *entry* is one of *roots* or lies beneath one of them.
  """
  if not isinstance(entry, str):
    return False
  try:
    resolved = os.path.abspath(entry)
  except (OSError, ValueError):
    return False
  return any(
    resolved == root or resolved.startswith(root + os.sep) for root in roots
  )


def _sequence_matches(rebuilt, items):
  """True when *rebuilt* holds exactly the objects in *items*.

  Compared by identity: a subclass constructor with a non-iterable
  signature accepts the list and returns an EMPTY container, which would
  otherwise drop the user's arguments silently.
  """
  try:
    if len(rebuilt) != len(items):
      return False
  except TypeError:
    return False
  return all(a is b for a, b in zip(rebuilt, items, strict=False))


def _mapping_matches(rebuilt, items):
  """True when *rebuilt* holds exactly the pairs in *items*."""
  try:
    if len(rebuilt) != len(items):
      return False
  except TypeError:
    return False
  for key, value in items.items():
    # `in` rather than .get so a defaultdict is not populated here.
    if key not in rebuilt or rebuilt[key] is not value:
      return False
  return True


def _rebuild_tuple(original, items):
  """Rebuild a tuple, preserving NamedTuple and tuple-subclass types."""
  if hasattr(original, "_fields"):
    try:
      return type(original)(*items)
    except Exception as e:
      logging.warning(
        "Could not rebuild %s (%s); passing a plain tuple instead.",
        type(original).__name__,
        e,
      )
      return tuple(items)
  if type(original) is tuple:
    return tuple(items)
  try:
    rebuilt = type(original)(items)
  except Exception as e:
    logging.warning(
      "Could not rebuild %s (%s); passing a plain tuple instead.",
      type(original).__name__,
      e,
    )
    return tuple(items)
  if not _sequence_matches(rebuilt, items):
    logging.warning(
      "Rebuilding %s dropped its contents; passing a plain tuple instead.",
      type(original).__name__,
    )
    return tuple(items)
  return rebuilt


def _rebuild_sequence(original, items):
  """Rebuild a list subclass, falling back to a plain list."""
  try:
    rebuilt = type(original)(items)
  except Exception as e:
    logging.warning(
      "Could not rebuild %s (%s); passing a plain list instead.",
      type(original).__name__,
      e,
    )
    return items
  if not _sequence_matches(rebuilt, items):
    logging.warning(
      "Rebuilding %s dropped its contents; passing a plain list instead.",
      type(original).__name__,
    )
    return items
  return rebuilt


def _rebuild_mapping(original, items):
  """Rebuild a dict subclass, preserving its type and factory state."""
  factory = None
  if isinstance(original, collections.defaultdict):
    factory = original.default_factory
  try:
    rebuilt = (
      type(original)(factory) if factory is not None else type(original)()
    )
    rebuilt.update(items)
    if _mapping_matches(rebuilt, items):
      return rebuilt
  except Exception:
    pass
  try:
    rebuilt = type(original)(items)
    if _mapping_matches(rebuilt, items):
      return rebuilt
  except Exception as e:
    logging.warning(
      "Could not rebuild %s (%s); passing a plain dict instead.",
      type(original).__name__,
      e,
    )
    return items
  logging.warning(
    "Rebuilding %s dropped its contents; passing a plain dict instead.",
    type(original).__name__,
  )
  return items


def resolve_data_refs(
  args: tuple,
  kwargs: dict,
  storage_client: storage.Client,
  data_dir: str,
) -> tuple[tuple, dict]:
  """Recursively resolve data ref dicts in args/kwargs to local paths.

  Containers are rebuilt only when something inside them actually
  changed, so argument types, subclass state and aliasing between
  arguments survive the walk. An ``id()``-keyed memo makes repeated and
  self-referential structures resolve once.
  """
  counter = 0
  resolved_uris: dict[str, str] = {}
  memo: dict[int, object] = {}
  # Keeps every walked container alive so ids cannot be recycled.
  keepalive = [args, kwargs]

  def _resolve_ref(obj):
    nonlocal counter
    if obj.get("mount_path") is not None:
      # For FUSE-mounted single files, resolve to the actual file path
      # rather than returning the mount directory.
      if obj.get("fuse") and not obj.get("is_dir"):
        resolved = _resolve_fuse_single_file(obj["mount_path"])
        if resolved:
          return resolved
      return obj["mount_path"]
    uri = obj["uri"]
    if uri in resolved_uris:
      return resolved_uris[uri]
    local_dir = os.path.join(data_dir, str(counter))
    counter += 1
    _download_data(obj, local_dir, storage_client)
    # Return file path for single files, directory path otherwise
    if not obj["is_dir"]:
      files = os.listdir(local_dir)
      if len(files) == 1:
        path = os.path.join(local_dir, files[0])
        resolved_uris[uri] = path
        return path
    resolved_uris[uri] = local_dir
    return local_dir

  def _resolve(obj):
    obj_id = id(obj)
    if obj_id in memo:
      return memo[obj_id]

    if _is_data_ref(obj):
      keepalive.append(obj)
      resolved = _resolve_ref(obj)
      memo[obj_id] = resolved
      return resolved

    if isinstance(obj, dict):
      keepalive.append(obj)
      rebuilt: dict = {}
      # Seeded before recursing so cycles terminate.
      memo[obj_id] = rebuilt
      changed = False
      for key, value in obj.items():
        if _is_data_ref(key):
          raise ValueError(
            "Data objects are not supported as dict keys "
            f"(found {key.get('uri')!r} used as a key)."
          )
        new_value = _resolve(value)
        changed = changed or new_value is not value
        rebuilt[key] = new_value
      if not changed:
        memo[obj_id] = obj
        return obj
      if type(obj) is dict:
        return rebuilt
      converted = _rebuild_mapping(obj, rebuilt)
      memo[obj_id] = converted
      return converted

    if isinstance(obj, list):
      keepalive.append(obj)
      rebuilt_list: list = []
      memo[obj_id] = rebuilt_list
      changed = False
      for item in obj:
        new_item = _resolve(item)
        changed = changed or new_item is not item
        rebuilt_list.append(new_item)
      if not changed:
        memo[obj_id] = obj
        return obj
      if type(obj) is list:
        return rebuilt_list
      converted = _rebuild_sequence(obj, rebuilt_list)
      memo[obj_id] = converted
      return converted

    if isinstance(obj, tuple):
      # A tuple can only take part in a cycle through a mutable
      # container, which is already memo-seeded above.
      keepalive.append(obj)
      items = []
      changed = False
      for item in obj:
        new_item = _resolve(item)
        changed = changed or new_item is not item
        items.append(new_item)
      if not changed:
        memo[obj_id] = obj
        return obj
      rebuilt_tuple = _rebuild_tuple(obj, items)
      memo[obj_id] = rebuilt_tuple
      return rebuilt_tuple

    # Sets can never hold a data ref: ref dicts are unhashable, and so is
    # anything nesting one. Returning as-is preserves type and identity.
    return obj

  resolved_args = tuple(_resolve(a) for a in args)
  resolved_kwargs = {k: _resolve(v) for k, v in kwargs.items()}
  return resolved_args, resolved_kwargs


def _download_hf_data(
  uri: str, target_dir: str, trust_remote_code: bool = False
) -> None:
  """Download data from Hugging Face Datasets to a local directory."""
  try:
    import datasets
  except ImportError as e:
    raise RuntimeError(
      "The 'datasets' package is required to load 'hf://' URIs. "
      "Please install it in your Kinetic base image or requirements."
    ) from e

  logging.info("Downloading Hugging Face dataset from %s", uri)

  parsed = urllib.parse.urlparse(uri)
  repo_id = (parsed.netloc + parsed.path).strip("/")
  query = urllib.parse.parse_qs(parsed.query)
  split = query.get("split", [None])[0]
  config_name = query.get("config_name", [None])[0]
  revision = query.get("revision", [None])[0]

  if trust_remote_code:
    logging.warning(
      "================================================================================\n"
      "WARNING: trust_remote_code=True is enabled for Hugging Face dataset loading.\n"
      "This allows execution of arbitrary code from the dataset repository.\n"
      "Ensure you trust the repository %r before proceeding.\n"
      "================================================================================",
      repo_id,
    )

  # Use a temporary directory for the cache to ensure it is cleaned up after saving,
  # preventing persistent disk accumulation (though it causes a transient 2x peak).
  # This also avoids save_to_disk failing on a non-empty directory.
  with tempfile.TemporaryDirectory(prefix="hf-cache-") as cache_dir:
    ds = datasets.load_dataset(
      repo_id,
      name=config_name,
      split=split,
      revision=revision,
      trust_remote_code=trust_remote_code,
      cache_dir=cache_dir,
    )
    ds.save_to_disk(target_dir)

    # Calculate total size of the saved dataset
    total_size = 0
    for dirpath, _, filenames in os.walk(target_dir):
      for f in filenames:
        total_size += os.path.getsize(os.path.join(dirpath, f))

    logging.info("Saved HF dataset to %s (%d bytes)", target_dir, total_size)


def _download_data(
  ref: dict, target_dir: str, storage_client: storage.Client
) -> None:
  """Download data from a GCS URI (or HF URI) to a local directory."""
  os.makedirs(target_dir, exist_ok=True)
  uri = ref["uri"]

  if uri.startswith("hf://"):
    _download_hf_data(
      uri,
      target_dir,
      trust_remote_code=ref.get("hf_trust_remote_code", False),
    )
    return

  parts = uri.replace("gs://", "").split("/", 1)
  bucket_name = parts[0]
  prefix = parts[1].rstrip("/") if len(parts) > 1 else ""
  bucket = storage_client.bucket(bucket_name)

  blobs = bucket.list_blobs(prefix=prefix + "/")
  total_downloaded = 0
  batch = []
  for blob in blobs:
    if blob.name.endswith("/"):
      continue
    batch.append(blob.name[len(prefix) + 1 :])
    if len(batch) >= _DOWNLOAD_BATCH_SIZE:
      transfer_manager.download_many_to_path(
        bucket,
        batch,
        destination_directory=target_dir,
        blob_name_prefix=prefix + "/",
        worker_type=transfer_manager.THREAD,
        raise_exception=True,
      )
      total_downloaded += len(batch)
      batch = []

  if batch:
    transfer_manager.download_many_to_path(
      bucket,
      batch,
      destination_directory=target_dir,
      blob_name_prefix=prefix + "/",
      worker_type=transfer_manager.THREAD,
      raise_exception=True,
    )
    total_downloaded += len(batch)

  if total_downloaded:
    logging.info(
      "Downloaded %d files from %s to %s", total_downloaded, uri, target_dir
    )


def _download_from_gcs(client, gcs_path, local_path):
  """Download file from GCS.

  Args:
      client: Cloud Storage client
      gcs_path: GCS URI (gs://bucket/path)
      local_path: Local file path
  """
  # Parse gs://bucket/path format
  parts = gcs_path.replace("gs://", "").split("/", 1)
  bucket_name = parts[0]
  blob_path = parts[1]

  bucket = client.bucket(bucket_name)
  blob = bucket.blob(blob_path)
  blob.download_to_filename(local_path)


def _upload_to_gcs(client, local_path, gcs_path):
  """Upload file to GCS.

  Args:
      client: Cloud Storage client
      local_path: Local file path
      gcs_path: GCS URI (gs://bucket/path)
  """
  parts = gcs_path.replace("gs://", "").split("/", 1)
  bucket_name = parts[0]
  blob_path = parts[1]

  bucket = client.bucket(bucket_name)
  blob = bucket.blob(blob_path)
  blob.upload_from_filename(local_path)


if __name__ == "__main__":
  main()
