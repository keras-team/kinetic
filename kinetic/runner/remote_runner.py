#!/usr/bin/env python3
"""Remote execution entrypoint for kinetic.

This script runs on the remote TPU/GPU and executes the user's function.
Artifacts are downloaded from and uploaded to Cloud Storage (GCS).
"""

import atexit
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

import cloudpickle
from absl import logging
from google.cloud import exceptions as cloud_exceptions
from google.cloud import storage
from google.cloud.storage import transfer_manager

_DOWNLOAD_BATCH_SIZE = 10000

# Sentinel blob name written by the leader once it has finished
# waiting for a debugger client and is about to call the user
# function. Workers poll for this to stay in sync with the leader.
_LEADER_READY_SENTINEL = ".leader_ready"

# Extra seconds workers wait beyond the leader's attach timeout,
# to cover GCS write latency and any processing after the leader's
# wait_for_client() returns.
_WORKER_WAIT_BUFFER_SECONDS = 60


def main():
  """Main entry point for remote execution.

  Usage: python remote_runner.py <context_gcs> <payload_gcs> <result_gcs> [requirements_gcs]
  """
  if len(sys.argv) < 4:
    logging.error(
      "Usage: remote_runner.py <context_gcs> <payload_gcs> <result_gcs>"
      " [requirements_gcs]"
    )
    sys.exit(1)

  context_gcs = sys.argv[1]
  payload_gcs = sys.argv[2]
  result_gcs = sys.argv[3]
  requirements_gcs = sys.argv[4] if len(sys.argv) > 4 else None

  logging.info("Starting remote execution")

  # Create secure temp directory and register cleanup
  temp_dir = tempfile.mkdtemp(prefix="kinetic-run-")
  atexit.register(shutil.rmtree, temp_dir, ignore_errors=True)

  # Define local paths
  context_path = os.path.join(temp_dir, "context.zip")
  payload_path = os.path.join(temp_dir, "payload.pkl")
  result_path = os.path.join(temp_dir, "result.pkl")
  workspace_dir = os.path.join(temp_dir, "workspace")
  data_dir = os.path.join(temp_dir, "data")

  try:
    storage_client = storage.Client()

    # Download artifacts from Cloud Storage
    logging.info("Downloading artifacts...")
    _download_from_gcs(storage_client, context_gcs, context_path)
    _download_from_gcs(storage_client, payload_gcs, payload_path)

    # Install user requirements at startup (prebuilt image mode)
    if requirements_gcs:
      _install_requirements(storage_client, requirements_gcs, temp_dir)

    # Extract context
    if os.path.exists(workspace_dir):
      shutil.rmtree(workspace_dir)
    os.makedirs(workspace_dir)

    with zipfile.ZipFile(context_path, "r") as zip_ref:
      zip_ref.extractall(workspace_dir)

    # Add workspace to Python path
    sys.path.insert(0, workspace_dir)

    # Load and deserialize payload
    logging.info("Loading function payload")
    with open(payload_path, "rb") as f:
      payload = cloudpickle.load(f)

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
    volumes = payload.get("volumes", [])
    if volumes:
      resolve_volumes(volumes, storage_client)
    args, kwargs = resolve_data_refs(args, kwargs, storage_client, data_dir)

    # Start debugpy server if debug mode is enabled
    is_debug = os.environ.get("KINETIC_DEBUG") == "1"
    is_debug_worker = os.environ.get("KINETIC_DEBUG_WAIT_LEADER") == "1"
    debugger_attached = False
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

    # Execute function and capture result
    logging.info("Executing %s()", func.__name__)
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

    # Serialize result or exception
    result_payload = {
      "success": exception is None,
      "result": result if exception is None else None,
      "exception": exception,
      "traceback": remote_traceback,
    }

    try:
      with open(result_path, "wb") as f:
        cloudpickle.dump(result_payload, f)
    except (pickle.PicklingError, TypeError) as serialize_err:
      logging.error("Failed to serialize result: %s", serialize_err)
      fallback_payload = {
        "success": False,
        "result": None,
        "exception": RuntimeError(
          f"Result serialization failed: {serialize_err}"
        ),
        "traceback": remote_traceback,
      }
      with open(result_path, "wb") as f:
        cloudpickle.dump(fallback_payload, f)

    # Upload result to Cloud Storage
    logging.info("Uploading result...")
    _upload_to_gcs(storage_client, result_path, result_gcs)

    logging.info("Execution complete")
    sys.exit(0 if exception is None else 1)

  except Exception as e:
    logging.fatal("%s", e)
    traceback.print_exc()
    sys.exit(1)


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
# The pod spec normally propagates DEBUG_WAIT_TIMEOUT from
# kinetic.debug as the env var, keeping both sides in sync.
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
  import threading

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
        ref["gcs_uri"],
        mount_path,
      )
      continue
    logging.info("Resolving volume: %s -> %s", ref["gcs_uri"], mount_path)
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


def resolve_data_refs(
  args: tuple,
  kwargs: dict,
  storage_client: storage.Client,
  data_dir: str,
) -> tuple[tuple, dict]:
  """Recursively resolve data ref dicts in args/kwargs to local paths."""
  counter = 0
  resolved_uris: dict[str, str] = {}

  def _resolve(obj):
    nonlocal counter
    # Data ref that needs downloading (no mount_path means not volume-mounted)
    if isinstance(obj, dict) and obj.get("__data_ref__"):
      if obj.get("mount_path") is not None:
        # For FUSE-mounted single files, resolve to the actual file path
        # rather than returning the mount directory.
        if obj.get("fuse") and not obj.get("is_dir"):
          resolved = _resolve_fuse_single_file(obj["mount_path"])
          if resolved:
            return resolved
        return obj["mount_path"]
      gcs_uri = obj["gcs_uri"]
      if gcs_uri in resolved_uris:
        return resolved_uris[gcs_uri]
      local_dir = os.path.join(data_dir, str(counter))
      counter += 1
      _download_data(obj, local_dir, storage_client)
      # Return file path for single files, directory path otherwise
      if not obj["is_dir"]:
        files = os.listdir(local_dir)
        if len(files) == 1:
          path = os.path.join(local_dir, files[0])
          resolved_uris[gcs_uri] = path
          return path
      resolved_uris[gcs_uri] = local_dir
      return local_dir
    # Recurse into containers to find nested data refs
    if isinstance(obj, dict):
      return {k: _resolve(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
      return type(obj)(_resolve(item) for item in obj)
    return obj

  resolved_args = tuple(_resolve(a) for a in args)
  resolved_kwargs = {k: _resolve(v) for k, v in kwargs.items()}
  return resolved_args, resolved_kwargs


def _download_hf_data(uri: str, target_dir: str) -> None:
  """Download data from Hugging Face Datasets to a local directory."""
  try:
    import datasets
  except ImportError as e:
    raise RuntimeError(
      "The 'datasets' package is required to load 'hf://' URIs. "
      "Please install it in your Kinetic base image or requirements."
    ) from e

  import urllib.parse

  logging.info("Downloading Hugging Face dataset from %s", uri)

  parsed = urllib.parse.urlparse(uri)
  repo_id = (parsed.netloc + parsed.path).strip("/")
  query = urllib.parse.parse_qs(parsed.query)
  split = query.get("split", [None])[0]
  config_name = query.get("config_name", [None])[0]
  revision = query.get("revision", [None])[0]
  trust_remote_code = (
    query.get("trust_remote_code", ["False"])[0].lower() == "true"
  )

  # Use a temporary directory for the cache to avoid save_to_disk failing
  # on a non-empty directory and to prevent doubling disk usage.
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


def _download_data(
  ref: dict, target_dir: str, storage_client: storage.Client
) -> None:
  """Download data from a GCS URI (or HF URI) to a local directory."""
  os.makedirs(target_dir, exist_ok=True)
  gcs_uri = ref["gcs_uri"]

  if gcs_uri.startswith("hf://"):
    _download_hf_data(gcs_uri, target_dir)
    return

  parts = gcs_uri.replace("gs://", "").split("/", 1)
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
      "Downloaded %d files from %s to %s", total_downloaded, gcs_uri, target_dir
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
