"""Cloud Storage operations for kinetic."""

from __future__ import annotations

import json
import os
import tempfile
import threading

from absl import logging
from google.cloud import exceptions as cloud_exceptions
from google.cloud import storage
from google.cloud.storage import transfer_manager
from google.cloud.storage.retry import DEFAULT_RETRY

from kinetic.constants import get_default_project
from kinetic.data import Data

_cached_clients: dict[str | None, storage.Client] = {}
_client_lock = threading.Lock()


def _get_client(project: str | None) -> storage.Client:
  """Return a cached storage client for the given project."""
  with _client_lock:
    if project not in _cached_clients:
      _cached_clients[project] = storage.Client(project=project)
    return _cached_clients[project]


def _get_bucket(
  bucket_name: str, project: str | None = None
) -> tuple[storage.Client, storage.Bucket]:
  """Return a (client, bucket) pair, resolving the project default."""
  project = project or get_default_project()
  client = _get_client(project)
  return client, client.bucket(bucket_name)


def _upload_json(
  bucket_name: str,
  blob_path: str,
  payload: dict,
  project: str | None = None,
) -> None:
  """Upload a dict as JSON to *blob_path* inside *bucket_name*."""
  _, bucket = _get_bucket(bucket_name, project)
  blob = bucket.blob(blob_path)
  blob.upload_from_string(
    json.dumps(payload, sort_keys=True),
    content_type="application/json",
    retry=DEFAULT_RETRY,
  )
  logging.info("Uploaded JSON to gs://%s/%s", bucket_name, blob_path)


def _download_json(
  bucket_name: str,
  blob_path: str,
  project: str | None = None,
) -> dict:
  """Download and deserialize JSON from *blob_path*."""
  _, bucket = _get_bucket(bucket_name, project)
  blob = bucket.blob(blob_path)
  text = blob.download_as_text()
  logging.info("Downloaded JSON from gs://%s/%s", bucket_name, blob_path)
  return json.loads(text)


def _delete_prefix(
  bucket_name: str,
  prefix: str,
  project: str | None = None,
) -> int:
  """Delete all blobs under *prefix*. Returns the count deleted."""
  _, bucket = _get_bucket(bucket_name, project)
  blobs = list(bucket.list_blobs(prefix=prefix))
  if not blobs:
    return 0
  try:
    bucket.delete_blobs(blobs, retry=DEFAULT_RETRY)
  except cloud_exceptions.NotFound:
    logging.warning(
      "Some blobs missing during cleanup of gs://%s/%s",
      bucket_name,
      prefix,
      exc_info=True,
    )
  return len(blobs)


def blob_exists(
  bucket_name: str,
  blob_path: str,
  project: str | None = None,
) -> bool:
  """Check if a specific blob exists in Cloud Storage."""
  _, bucket = _get_bucket(bucket_name, project)
  blob = bucket.blob(blob_path)
  return blob.exists()


def upload_empty_blob(
  bucket_name: str,
  blob_path: str,
  project: str | None = None,
) -> None:
  """Upload an empty string to a specific blob."""
  _, bucket = _get_bucket(bucket_name, project)
  blob = bucket.blob(blob_path)
  blob.upload_from_string("", retry=DEFAULT_RETRY)


def upload_artifacts(
  bucket_name: str,
  job_id: str,
  payload_path: str,
  context_path: str,
  project: str | None = None,
  requirements_content: str | None = None,
) -> None:
  """Upload execution artifacts to Cloud Storage.

  Args:
      bucket_name: Name of the GCS bucket
      job_id: Unique job identifier
      payload_path: Local path to payload.pkl
      context_path: Local path to context.zip
      project: GCP project ID (optional, uses env vars if not provided)
      requirements_content: Filtered requirements text for runtime install
          (prebuilt image mode only). Uploaded as `requirements.txt`.
  """
  client, bucket = _get_bucket(bucket_name, project)

  # Upload payload
  blob = bucket.blob(f"{job_id}/payload.pkl")
  blob.upload_from_filename(payload_path, retry=DEFAULT_RETRY)
  logging.info(
    "Uploaded payload to gs://%s/%s/payload.pkl", bucket_name, job_id
  )

  # Upload context
  blob = bucket.blob(f"{job_id}/context.zip")
  blob.upload_from_filename(context_path, retry=DEFAULT_RETRY)
  logging.info(
    "Uploaded context to gs://%s/%s/context.zip", bucket_name, job_id
  )

  # Upload requirements (prebuilt mode only)
  if requirements_content is not None:
    blob = bucket.blob(f"{job_id}/requirements.txt")
    blob.upload_from_string(requirements_content, retry=DEFAULT_RETRY)
    logging.info(
      "Uploaded requirements to gs://%s/%s/requirements.txt",
      bucket_name,
      job_id,
    )

  # Get project ID for console link
  logging.info(
    "View artifacts: https://console.cloud.google.com/storage/browser/%s/%s?project=%s",
    bucket_name,
    job_id,
    client.project,
  )


def download_result(
  bucket_name: str, job_id: str, project: str | None = None
) -> str:
  """Download result from Cloud Storage.

  Args:
      bucket_name: Name of the GCS bucket
      job_id: Unique job identifier
      project: GCP project ID (optional, uses env vars if not provided)

  Returns:
      Local path to downloaded result file
  """
  _, bucket = _get_bucket(bucket_name, project)

  blob = bucket.blob(f"{job_id}/result.pkl")
  local_path = os.path.join(tempfile.gettempdir(), f"result-{job_id}.pkl")
  blob.download_to_filename(local_path)
  logging.info(
    "Downloaded result from gs://%s/%s/result.pkl", bucket_name, job_id
  )

  return local_path


def upload_handle(
  bucket_name: str,
  job_id: str,
  handle_payload: dict[str, str],
  project: str | None = None,
) -> None:
  """Upload a job handle to Cloud Storage as JSON."""
  _upload_json(bucket_name, f"{job_id}/handle.json", handle_payload, project)


def download_handle(
  bucket_name: str, job_id: str, project: str | None = None
) -> dict[str, str]:
  """Download and deserialize a job handle from Cloud Storage."""
  return _download_json(bucket_name, f"{job_id}/handle.json", project)


def cleanup_artifacts(
  bucket_name: str, job_id: str, project: str | None = None
) -> None:
  """Clean up job artifacts from Cloud Storage.

  Args:
      bucket_name: Name of the GCS bucket
      job_id: Unique job identifier
      project: GCP project ID (optional, uses env vars if not provided)
  """
  count = _delete_prefix(bucket_name, f"{job_id}/", project)
  if count:
    logging.info(
      "Cleaned up %d artifacts from gs://%s/%s/", count, bucket_name, job_id
    )


def _manifest_prefix(group_id: str) -> str:
  return f"_groups/{group_id}"


def upload_manifest(
  bucket_name: str,
  group_id: str,
  manifest: dict,
  project: str | None = None,
) -> None:
  """Upload a group manifest to Cloud Storage."""
  _upload_json(
    bucket_name,
    f"{_manifest_prefix(group_id)}/manifest.json",
    manifest,
    project,
  )


def download_manifest(
  bucket_name: str,
  group_id: str,
  project: str | None = None,
) -> dict:
  """Download and deserialize a group manifest from Cloud Storage."""
  blob_path = f"{_manifest_prefix(group_id)}/manifest.json"
  try:
    return _download_json(bucket_name, blob_path, project)
  except cloud_exceptions.NotFound as e:
    raise FileNotFoundError(
      f"Manifest not found for group '{group_id}' at "
      f"gs://{bucket_name}/{blob_path}"
    ) from e


def cleanup_manifest(
  bucket_name: str,
  group_id: str,
  project: str | None = None,
) -> None:
  """Delete a group manifest and its GCS prefix."""
  count = _delete_prefix(bucket_name, f"{_manifest_prefix(group_id)}/", project)
  if count:
    logging.info(
      "Cleaned up manifest for group %s from gs://%s/_groups/%s/",
      group_id,
      bucket_name,
      group_id,
    )


def upload_data(
  bucket_name: str,
  data: Data,
  project: str | None = None,
  namespace_prefix: str = "default",
) -> str:
  """Upload a Data object to GCS with content-based caching.

  For GCS Data: returns the original URI (no upload).
  For local Data: computes content hash, uploads on cache miss.

  Args:
      bucket_name: GCS bucket name.
      data: Data object to upload.
      project: GCP project ID (auto-detected if None).
      namespace_prefix: Namespace GCS prefix. Defaults to "default".

  Returns:
      GCS URI where the data is available.
  """
  if data.is_gcs:
    logging.info("Data already on GCS: %s", data.path)
    return data.path

  content_hash = data.content_hash()
  namespace_prefix = namespace_prefix.strip("/")
  cache_prefix = f"{namespace_prefix}/data-cache/{content_hash}"

  _, bucket = _get_bucket(bucket_name, project)

  # O(1) cache hit check via sentinel blob.  Markers live under a separate
  # top-level prefix ("data-markers/") so they never appear inside
  # FUSE-mounted directories (which scope to "data-cache/{hash}/").
  marker_prefix = f"{namespace_prefix}/data-markers/{content_hash}"
  marker_blob = bucket.blob(marker_prefix)
  if marker_blob.exists():
    gcs_uri = f"gs://{bucket_name}/{cache_prefix}"
    logging.info(
      "Data cache hit (hash=%s...): %s",
      content_hash[:12],
      gcs_uri,
    )
    return gcs_uri

  # Size warning for large local data
  total_size = _compute_total_size(data.path)
  if total_size > 10 * 1024**3:  # 10 GB
    size_gb = total_size / (1024**3)
    logging.warning(
      "Data at '%s' is %.1f GB. For large datasets, consider using "
      'a direct GCS URI (Data("gs://...")) with framework-native '
      "I/O (tf.data, grain) for better performance.",
      data._raw_path,
      size_gb,
    )

  # Cache miss — upload
  logging.info(
    "Uploading data (hash=%s...) to gs://%s/%s/",
    content_hash[:12],
    bucket_name,
    cache_prefix,
  )

  if data.is_dir:
    _upload_directory(bucket, data.path, cache_prefix)
  else:
    filename = os.path.basename(data.path)
    blob = bucket.blob(f"{cache_prefix}/{filename}")
    blob.upload_from_filename(data.path, retry=DEFAULT_RETRY)

  # Write sentinel last — signals upload-complete
  marker_blob.upload_from_string("", retry=DEFAULT_RETRY)
  logging.info("Data uploaded to gs://%s/%s/", bucket_name, cache_prefix)
  return f"gs://{bucket_name}/{cache_prefix}"


def _compute_total_size(path: str) -> int:
  """Compute total size in bytes of a file or directory."""
  if os.path.isfile(path):
    return os.path.getsize(path)
  total = 0
  for root, _dirs, files in os.walk(path):
    for fname in files:
      total += os.path.getsize(os.path.join(root, fname))
  return total


def _upload_directory(
  bucket: storage.Bucket, local_dir: str, gcs_prefix: str
) -> None:
  """Upload a local directory to GCS preserving structure."""
  filenames = []
  for root, _dirs, files in os.walk(local_dir):
    for fname in files:
      local_path = os.path.join(root, fname)
      rel_path = os.path.relpath(local_path, local_dir).replace(os.sep, "/")
      filenames.append(rel_path)

  if not filenames:
    return

  logging.info("Uploading %d files to GCS...", len(filenames))

  transfer_manager.upload_many_from_filenames(
    bucket,
    filenames,
    source_directory=local_dir,
    blob_name_prefix=f"{gcs_prefix}/",
    worker_type=transfer_manager.THREAD,
    raise_exception=True,
  )
