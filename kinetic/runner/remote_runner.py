#!/usr/bin/env python3
"""Remote execution entrypoint for kinetic.

This script runs on the remote TPU/GPU and executes the user's function.
Artifacts are downloaded from and uploaded to Cloud Storage (GCS).
"""

import os
import shutil
import sys
import tempfile
import traceback
import zipfile

import cloudpickle
from absl import logging
from google.cloud import storage
from google.cloud.storage import transfer_manager

_DOWNLOAD_BATCH_SIZE = 10000

# Base temp directory for remote execution artifacts
TEMP_DIR = tempfile.gettempdir()
DATA_DIR = os.path.join(TEMP_DIR, "data")


def main():
  """Main entry point for remote execution.

  Usage: python remote_runner.py gs://bucket/context.zip gs://bucket/payload.pkl gs://bucket/result.pkl
  """

  if len(sys.argv) < 4:
    logging.error(
      "Usage: remote_runner.py <context_gcs> <payload_gcs> <result_gcs>"
    )
    sys.exit(1)

  run_gcs_mode()


def run_gcs_mode():
  """Execute with Cloud Storage artifacts.

  Args from sys.argv:
      sys.argv[1]: GCS path to context.zip
      sys.argv[2]: GCS path to payload.pkl
      sys.argv[3]: GCS path to result.pkl (output)
  """
  context_gcs = sys.argv[1]
  payload_gcs = sys.argv[2]
  result_gcs = sys.argv[3]

  logging.info("Starting GCS execution mode")

  # Define local paths using tempfile
  context_path = os.path.join(TEMP_DIR, "context.zip")
  payload_path = os.path.join(TEMP_DIR, "payload.pkl")
  result_path = os.path.join(TEMP_DIR, "result.pkl")
  workspace_dir = os.path.join(TEMP_DIR, "workspace")

  try:
    storage_client = storage.Client()

    # Download artifacts from Cloud Storage
    logging.info("Downloading artifacts...")
    _download_from_gcs(storage_client, context_gcs, context_path)
    _download_from_gcs(storage_client, payload_gcs, payload_path)

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
    args, kwargs = resolve_data_refs(args, kwargs, storage_client)

    # Execute function and capture result
    logging.info("Executing %s()", func.__name__)
    result = None
    exception = None

    try:
      result = func(*args, **kwargs)
      logging.info("Function completed successfully")
    except BaseException as e:
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
      "traceback": traceback.format_exc() if exception else None,
    }

    with open(result_path, "wb") as f:
      cloudpickle.dump(result_payload, f)

    # Upload result to Cloud Storage
    logging.info("Uploading result...")
    _upload_to_gcs(storage_client, result_path, result_gcs)

    logging.info("Execution complete")
    sys.exit(0 if exception is None else 1)

  except Exception as e:
    logging.fatal("%s", e)
    traceback.print_exc()
    sys.exit(1)


def resolve_volumes(volume_refs, storage_client):
  """Download volume data to their specified mount paths."""
  for ref in volume_refs:
    mount_path = ref["mount_path"]
    logging.info("Resolving volume: %s -> %s", ref["gcs_uri"], mount_path)
    _download_data(ref, mount_path, storage_client)


def resolve_data_refs(args, kwargs, storage_client):
  """Recursively resolve data ref dicts in args/kwargs to local paths."""
  counter = 0
  resolved_uris = {}

  def _resolve(obj):
    nonlocal counter
    # Data ref that needs downloading (no mount_path means not volume-mounted)
    if isinstance(obj, dict) and obj.get("__data_ref__"):
      # Volume-mounted data refs are handled by Kubernetes, skip download
      if obj.get("mount_path") is not None:
        return obj["mount_path"]
      gcs_uri = obj["gcs_uri"]
      if gcs_uri in resolved_uris:
        return resolved_uris[gcs_uri]
      local_dir = os.path.join(DATA_DIR, str(counter))
      counter += 1
      _download_data(obj, local_dir, storage_client)
      # Return file path for single files, directory path otherwise
      if not obj["is_dir"]:
        files = [f for f in os.listdir(local_dir) if f != ".cache_marker"]
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


def _download_data(ref, target_dir, storage_client):
  """Download data from a GCS URI to a local directory."""
  os.makedirs(target_dir, exist_ok=True)
  gcs_uri = ref["gcs_uri"]

  parts = gcs_uri.replace("gs://", "").split("/", 1)
  bucket_name = parts[0]
  prefix = parts[1].rstrip("/") if len(parts) > 1 else ""
  bucket = storage_client.bucket(bucket_name)

  blobs = bucket.list_blobs(prefix=prefix + "/")
  total_downloaded = 0
  batch = []
  for blob in blobs:
    if blob.name.endswith("/") or blob.name.endswith(".cache_marker"):
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
