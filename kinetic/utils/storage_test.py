"""Tests for kinetic.utils.storage — Cloud Storage operations."""

import json
import os
import pathlib
import tempfile
from unittest import mock
from unittest.mock import MagicMock

from absl.testing import absltest, parameterized

from kinetic.constants import get_default_project
from kinetic.data import Data
from kinetic.utils import storage as storage_module
from kinetic.utils.storage import (
  DEFAULT_RETRY,
  _compute_total_size,
  _upload_directory,
  cleanup_artifacts,
  download_handle,
  download_result,
  upload_artifacts,
  upload_data,
  upload_handle,
)


def _make_temp_path(test_case):
  """Create a temp directory that is cleaned up after the test."""
  td = tempfile.TemporaryDirectory()
  test_case.addCleanup(td.cleanup)
  return pathlib.Path(td.name)


class _GcsTestBase(absltest.TestCase):
  """Base class that provides a mocked GCS client."""

  def setUp(self):
    super().setUp()
    storage_module._cached_clients.clear()
    self.mock_gcs = self.enterContext(
      mock.patch(
        "kinetic.utils.storage.storage.Client",
      )
    ).return_value


class TestUploadArtifacts(_GcsTestBase):
  def test_uploads_payload_and_context(self):
    mock_bucket = self.mock_gcs.bucket.return_value
    mock_blob = mock_bucket.blob.return_value

    upload_artifacts(
      bucket_name="my-bucket",
      job_id="job-abc123",
      payload_path="/tmp/payload.pkl",
      context_path="/tmp/context.zip",
      project="test-project",
    )

    mock_bucket.blob.assert_any_call("job-abc123/payload.pkl")
    mock_bucket.blob.assert_any_call("job-abc123/context.zip")
    self.assertEqual(mock_blob.upload_from_filename.call_count, 2)

  def test_uses_correct_bucket(self):
    upload_artifacts(
      bucket_name="my-custom-bucket",
      job_id="job-123",
      payload_path="/tmp/p.pkl",
      context_path="/tmp/c.zip",
      project="proj",
    )
    self.mock_gcs.bucket.assert_called_with("my-custom-bucket")


class TestDownloadResult(_GcsTestBase):
  def test_downloads_result_blob(self):
    mock_bucket = self.mock_gcs.bucket.return_value
    mock_blob = mock_bucket.blob.return_value

    download_result("my-bucket", "job-abc", project="proj")

    mock_bucket.blob.assert_called_once_with("job-abc/result.pkl")
    mock_blob.download_to_filename.assert_called_once()

  def test_returns_path_with_job_id(self):
    result = download_result("my-bucket", "job-xyz", project="proj")
    self.assertIn("result-job-xyz.pkl", result)


class TestHandleStorage(_GcsTestBase):
  def test_upload_handle_writes_json_blob(self):
    mock_bucket = self.mock_gcs.bucket.return_value
    mock_blob = mock_bucket.blob.return_value
    handle = {"job_id": "job-abc", "backend": "gke"}

    upload_handle("my-bucket", "job-abc", handle, project="proj")

    mock_bucket.blob.assert_called_once_with("job-abc/handle.json")
    payload = mock_blob.upload_from_string.call_args[0][0]
    self.assertEqual(json.loads(payload), handle)
    self.assertEqual(
      mock_blob.upload_from_string.call_args.kwargs["content_type"],
      "application/json",
    )

  def test_download_handle_reads_json_blob(self):
    mock_bucket = self.mock_gcs.bucket.return_value
    mock_blob = mock_bucket.blob.return_value
    mock_blob.download_as_text.return_value = json.dumps(
      {"job_id": "job-xyz", "backend": "pathways"}
    )

    handle = download_handle("my-bucket", "job-xyz", project="proj")

    mock_bucket.blob.assert_called_once_with("job-xyz/handle.json")
    self.assertEqual(handle["job_id"], "job-xyz")
    self.assertEqual(handle["backend"], "pathways")


class TestCleanupArtifacts(_GcsTestBase):
  def test_deletes_all_blobs(self):
    mock_bucket = self.mock_gcs.bucket.return_value
    blob1 = MagicMock()
    blob2 = MagicMock()
    blob3 = MagicMock()
    mock_bucket.list_blobs.return_value = [blob1, blob2, blob3]

    cleanup_artifacts("my-bucket", "job-abc", project="proj")

    mock_bucket.list_blobs.assert_called_once_with(prefix="job-abc/")
    mock_bucket.delete_blobs.assert_called_once_with(
      [blob1, blob2, blob3], retry=DEFAULT_RETRY
    )

  def test_no_blobs_no_error(self):
    mock_bucket = self.mock_gcs.bucket.return_value
    mock_bucket.list_blobs.return_value = []

    cleanup_artifacts("my-bucket", "job-abc", project="proj")

    mock_bucket.list_blobs.assert_called_once_with(prefix="job-abc/")


class TestGetProject(parameterized.TestCase):
  @parameterized.named_parameters(
    dict(
      testcase_name="kinetic_project_only",
      kn_project="kn-proj",
      gc_project=None,
      expected="kn-proj",
    ),
    dict(
      testcase_name="google_cloud_project_fallback",
      kn_project=None,
      gc_project="gc-proj",
      expected="gc-proj",
    ),
    dict(
      testcase_name="neither_set",
      kn_project=None,
      gc_project=None,
      expected=None,
    ),
    dict(
      testcase_name="kinetic_takes_precedence",
      kn_project="kn-proj",
      gc_project="gc-proj",
      expected="kn-proj",
    ),
  )
  def test_resolves_project(self, kn_project, gc_project, expected):
    env = {}
    if kn_project:
      env["KINETIC_PROJECT"] = kn_project
    if gc_project:
      env["GOOGLE_CLOUD_PROJECT"] = gc_project
    with mock.patch.dict(os.environ, env, clear=True):
      self.assertEqual(get_default_project(), expected)


class TestUploadData(_GcsTestBase):
  def test_gcs_data_returns_uri_no_upload(self):
    d = Data("gs://my-bucket/datasets/cifar10/")

    result = upload_data("jobs-bucket", d, project="proj")

    self.assertEqual(result, "gs://my-bucket/datasets/cifar10/")
    # No blob operations should occur
    self.mock_gcs.bucket.assert_not_called()

  def test_cache_hit_skips_upload(self):
    tmp = _make_temp_path(self)
    f = tmp / "data.csv"
    f.write_text("training data")
    d = Data(str(f))

    mock_bucket = self.mock_gcs.bucket.return_value
    marker_blob = MagicMock()
    marker_blob.exists.return_value = True
    mock_bucket.blob.return_value = marker_blob

    result = upload_data("jobs-bucket", d, project="proj")

    self.assertIn("gs://jobs-bucket/default/data-cache/", result)
    # Only the marker check should happen, no upload
    marker_blob.upload_from_filename.assert_not_called()

  def test_cache_miss_uploads_file_and_marker(self):
    tmp = _make_temp_path(self)
    f = tmp / "data.csv"
    f.write_text("training data")
    d = Data(str(f))
    content_hash = d.content_hash()

    mock_bucket = self.mock_gcs.bucket.return_value
    blobs = {}

    def track_blob(name):
      b = MagicMock()
      blobs[name] = b
      if "data-markers/" in name:
        b.exists.return_value = False
      return b

    mock_bucket.blob.side_effect = track_blob

    result = upload_data("jobs-bucket", d, project="proj")

    expected_prefix = f"default/data-cache/{content_hash}"
    self.assertEqual(result, f"gs://jobs-bucket/{expected_prefix}")
    # File blob uploaded
    file_blob_name = f"{expected_prefix}/data.csv"
    self.assertIn(file_blob_name, blobs)
    blobs[file_blob_name].upload_from_filename.assert_called_once()
    # Marker written last under separate prefix
    marker_name = f"default/data-markers/{content_hash}"
    self.assertIn(marker_name, blobs)
    blobs[marker_name].upload_from_string.assert_called_once_with(
      "", retry=mock.ANY
    )

  @mock.patch(
    "kinetic.utils.storage.transfer_manager.upload_many_from_filenames",
  )
  def test_cache_miss_uploads_directory(self, mock_upload):
    tmp = _make_temp_path(self)
    d_dir = tmp / "dataset"
    d_dir.mkdir()
    (d_dir / "train.csv").write_text("train")
    (d_dir / "val.csv").write_text("val")
    d = Data(str(d_dir))
    content_hash = d.content_hash()

    mock_bucket = self.mock_gcs.bucket.return_value
    marker_blob = MagicMock()
    marker_blob.exists.return_value = False
    mock_bucket.blob.return_value = marker_blob

    result = upload_data("jobs-bucket", d, project="proj")

    expected_prefix = f"default/data-cache/{content_hash}"
    self.assertEqual(result, f"gs://jobs-bucket/{expected_prefix}")
    # Directory upload via transfer_manager
    mock_upload.assert_called_once()
    filenames = sorted(mock_upload.call_args[0][1])
    self.assertEqual(filenames, ["train.csv", "val.csv"])
    # Marker written after upload
    marker_blob.upload_from_string.assert_called_once_with("", retry=mock.ANY)

  def test_custom_namespace(self):
    tmp = _make_temp_path(self)
    f = tmp / "data.csv"
    f.write_text("data")
    d = Data(str(f))

    mock_bucket = self.mock_gcs.bucket.return_value
    marker_blob = MagicMock()
    marker_blob.exists.return_value = True
    mock_bucket.blob.return_value = marker_blob

    result = upload_data(
      "jobs-bucket",
      d,
      project="proj",
      namespace_prefix="team-nlp",
    )

    self.assertIn("team-nlp/data-cache/", result)


class TestComputeTotalSize(absltest.TestCase):
  def test_single_file(self):
    tmp = _make_temp_path(self)
    f = tmp / "data.bin"
    f.write_bytes(b"x" * 100)
    self.assertEqual(_compute_total_size(str(f)), 100)

  def test_directory(self):
    tmp = _make_temp_path(self)
    d = tmp / "dir"
    d.mkdir()
    (d / "a.txt").write_bytes(b"x" * 50)
    (d / "b.txt").write_bytes(b"y" * 30)
    self.assertEqual(_compute_total_size(str(d)), 80)

  def test_empty_directory(self):
    tmp = _make_temp_path(self)
    d = tmp / "empty"
    d.mkdir()
    self.assertEqual(_compute_total_size(str(d)), 0)


class TestUploadDirectory(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.mock_upload = self.enterContext(
      mock.patch(
        "kinetic.utils.storage.transfer_manager.upload_many_from_filenames",
      )
    )

  def test_preserves_structure(self):
    tmp = _make_temp_path(self)
    d = tmp / "dataset"
    sub = d / "sub"
    sub.mkdir(parents=True)
    (d / "a.csv").write_text("a")
    (sub / "b.csv").write_text("b")

    mock_bucket = MagicMock()

    _upload_directory(mock_bucket, str(d), "prefix/hash")

    self.mock_upload.assert_called_once()
    call_kwargs = self.mock_upload.call_args
    filenames = sorted(call_kwargs[0][1])  # second positional arg
    self.assertEqual(filenames, ["a.csv", "sub/b.csv"])
    self.assertEqual(call_kwargs.kwargs["source_directory"], str(d))
    self.assertEqual(call_kwargs.kwargs["blob_name_prefix"], "prefix/hash/")
    self.assertTrue(call_kwargs.kwargs["raise_exception"])

  def test_empty_directory_is_noop(self):
    tmp = _make_temp_path(self)
    d = tmp / "empty_dataset"
    d.mkdir()

    mock_bucket = MagicMock()

    _upload_directory(mock_bucket, str(d), "prefix/hash")

    self.mock_upload.assert_not_called()


if __name__ == "__main__":
  absltest.main()
