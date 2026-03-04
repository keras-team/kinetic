"""Tests for keras_remote.utils.storage — Cloud Storage operations."""

import os
import pathlib
import tempfile
from unittest import mock
from unittest.mock import MagicMock

from absl.testing import absltest, parameterized

from keras_remote.data import Data
from keras_remote.infra.infra import get_default_project
from keras_remote.utils.storage import (
  _compute_total_size,
  _upload_directory,
  cleanup_artifacts,
  download_result,
  upload_artifacts,
  upload_data,
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
    self.mock_gcs = self.enterContext(
      mock.patch(
        "keras_remote.utils.storage.storage.Client",
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


class TestCleanupArtifacts(_GcsTestBase):
  def test_deletes_all_blobs(self):
    mock_bucket = self.mock_gcs.bucket.return_value
    blob1 = MagicMock()
    blob2 = MagicMock()
    blob3 = MagicMock()
    mock_bucket.list_blobs.return_value = [blob1, blob2, blob3]

    cleanup_artifacts("my-bucket", "job-abc", project="proj")

    mock_bucket.list_blobs.assert_called_once_with(prefix="job-abc/")
    blob1.delete.assert_called_once()
    blob2.delete.assert_called_once()
    blob3.delete.assert_called_once()

  def test_no_blobs_no_error(self):
    mock_bucket = self.mock_gcs.bucket.return_value
    mock_bucket.list_blobs.return_value = []

    cleanup_artifacts("my-bucket", "job-abc", project="proj")

    mock_bucket.list_blobs.assert_called_once_with(prefix="job-abc/")


class TestGetProject(parameterized.TestCase):
  @parameterized.named_parameters(
    dict(
      testcase_name="keras_remote_project_only",
      kr_project="kr-proj",
      gc_project=None,
      expected="kr-proj",
    ),
    dict(
      testcase_name="google_cloud_project_fallback",
      kr_project=None,
      gc_project="gc-proj",
      expected="gc-proj",
    ),
    dict(
      testcase_name="neither_set",
      kr_project=None,
      gc_project=None,
      expected=None,
    ),
    dict(
      testcase_name="keras_remote_takes_precedence",
      kr_project="kr-proj",
      gc_project="gc-proj",
      expected="kr-proj",
    ),
  )
  def test_resolves_project(self, kr_project, gc_project, expected):
    env = {}
    if kr_project:
      env["KERAS_REMOTE_PROJECT"] = kr_project
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
      if name.endswith(".cache_marker"):
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
    # Marker written last
    marker_name = f"{expected_prefix}/.cache_marker"
    self.assertIn(marker_name, blobs)
    blobs[marker_name].upload_from_string.assert_called_once_with("")

  def test_cache_miss_uploads_directory(self):
    tmp = _make_temp_path(self)
    d_dir = tmp / "dataset"
    d_dir.mkdir()
    (d_dir / "train.csv").write_text("train")
    (d_dir / "val.csv").write_text("val")
    d = Data(str(d_dir))

    mock_bucket = self.mock_gcs.bucket.return_value
    blobs = {}

    def track_blob(name):
      b = MagicMock()
      blobs[name] = b
      if name.endswith(".cache_marker"):
        b.exists.return_value = False
      return b

    mock_bucket.blob.side_effect = track_blob

    result = upload_data("jobs-bucket", d, project="proj")

    self.assertIn("gs://jobs-bucket/default/data-cache/", result)
    # Both files + marker should have blobs
    blob_names = list(blobs.keys())
    csv_blobs = [n for n in blob_names if n.endswith(".csv")]
    self.assertEqual(len(csv_blobs), 2)

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


class TestUploadDirectory(_GcsTestBase):
  def test_preserves_structure(self):
    tmp = _make_temp_path(self)
    d = tmp / "dataset"
    sub = d / "sub"
    sub.mkdir(parents=True)
    (d / "a.csv").write_text("a")
    (sub / "b.csv").write_text("b")

    mock_bucket = MagicMock()
    uploaded = {}

    def track_blob(name):
      b = MagicMock()
      uploaded[name] = b
      return b

    mock_bucket.blob.side_effect = track_blob

    _upload_directory(mock_bucket, str(d), "prefix/hash")

    self.assertIn("prefix/hash/a.csv", uploaded)
    self.assertIn("prefix/hash/sub/b.csv", uploaded)
    for blob in uploaded.values():
      blob.upload_from_filename.assert_called_once()


if __name__ == "__main__":
  absltest.main()
