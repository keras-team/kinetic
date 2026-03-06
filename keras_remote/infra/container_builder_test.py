"""Tests for keras_remote.infra.container_builder — hashing, Dockerfile gen, caching."""

import pathlib
import tempfile
from unittest import mock
from unittest.mock import MagicMock

from absl.testing import absltest, parameterized
from google.api_core import exceptions as google_exceptions

from keras_remote.infra.container_builder import (
  _generate_dockerfile,
  _hash_requirements,
  _image_exists,
  get_or_build_container,
)


def _make_temp_path(test_case):
  """Create a temp directory that is cleaned up after the test."""
  td = tempfile.TemporaryDirectory()
  test_case.addCleanup(td.cleanup)
  return pathlib.Path(td.name)


class TestHashRequirements(parameterized.TestCase):
  def test_deterministic(self):
    tmp_path = _make_temp_path(self)
    req = tmp_path / "requirements.txt"
    req.write_text("numpy==1.26\n")

    h1 = _hash_requirements(str(req), "gpu", "python:3.12-slim")
    h2 = _hash_requirements(str(req), "gpu", "python:3.12-slim")
    self.assertEqual(h1, h2)

  def test_different_requirements_different_hash(self):
    tmp_path = _make_temp_path(self)
    req1 = tmp_path / "r1.txt"
    req1.write_text("numpy==1.26\n")
    req2 = tmp_path / "r2.txt"
    req2.write_text("scipy==1.12\n")

    h1 = _hash_requirements(str(req1), "gpu", "python:3.12-slim")
    h2 = _hash_requirements(str(req2), "gpu", "python:3.12-slim")
    self.assertNotEqual(h1, h2)

  def test_different_category_different_hash(self):
    tmp_path = _make_temp_path(self)
    req = tmp_path / "requirements.txt"
    req.write_text("numpy\n")

    h1 = _hash_requirements(str(req), "gpu", "python:3.12-slim")
    h2 = _hash_requirements(str(req), "tpu", "python:3.12-slim")
    self.assertNotEqual(h1, h2)

  def test_different_base_image_different_hash(self):
    tmp_path = _make_temp_path(self)
    req = tmp_path / "requirements.txt"
    req.write_text("numpy\n")

    h1 = _hash_requirements(str(req), "gpu", "python:3.12-slim")
    h2 = _hash_requirements(str(req), "gpu", "python:3.11-slim")
    self.assertNotEqual(h1, h2)

  @parameterized.named_parameters(
    dict(testcase_name="none", requirements_path=None),
    dict(
      testcase_name="nonexistent",
      requirements_path="/nonexistent/path.txt",
    ),
  )
  def test_missing_requirements_valid(self, requirements_path):
    h = _hash_requirements(requirements_path, "cpu", "python:3.12-slim")
    self.assertIsInstance(h, str)
    self.assertLen(h, 64)

  def test_returns_hex_string(self):
    tmp_path = _make_temp_path(self)
    req = tmp_path / "r.txt"
    req.write_text("keras\n")
    h = _hash_requirements(str(req), "gpu", "python:3.12-slim")
    self.assertRegex(h, r"^[0-9a-f]{64}$")


class TestGenerateDockerfile(parameterized.TestCase):
  @parameterized.named_parameters(
    dict(
      testcase_name="cpu",
      category="cpu",
      expected=["pip install jax"],
      not_expected=["cuda", "tpu"],
    ),
    dict(
      testcase_name="gpu",
      category="gpu",
      expected=["jax[cuda12]"],
      not_expected=[],
    ),
    dict(
      testcase_name="tpu",
      category="tpu",
      expected=["jax[tpu]", "libtpu_releases"],
      not_expected=[],
    ),
  )
  def test_jax_install(self, category, expected, not_expected):
    content = _generate_dockerfile(
      base_image="python:3.12-slim",
      requirements_path=None,
      category=category,
    )
    for s in expected:
      self.assertIn(s, content)
    for s in not_expected:
      self.assertNotIn(s, content)

  def test_with_requirements(self):
    tmp_path = _make_temp_path(self)
    req = tmp_path / "requirements.txt"
    req.write_text("numpy\n")

    content = _generate_dockerfile(
      base_image="python:3.12-slim",
      requirements_path=str(req),
      category="cpu",
    )
    self.assertIn("COPY requirements.txt", content)
    self.assertIn("pip install -r", content)

  def test_without_requirements(self):
    content = _generate_dockerfile(
      base_image="python:3.12-slim",
      requirements_path=None,
      category="cpu",
    )
    self.assertNotIn("COPY requirements.txt", content)

  @parameterized.named_parameters(
    dict(
      testcase_name="remote_runner_copy",
      expected_substring="COPY remote_runner.py /app/remote_runner.py",
    ),
    dict(
      testcase_name="keras_backend_env",
      expected_substring="ENV KERAS_BACKEND=jax",
    ),
  )
  def test_contains_expected_content(self, expected_substring):
    content = _generate_dockerfile(
      base_image="python:3.12-slim",
      requirements_path=None,
      category="cpu",
    )
    self.assertIn(expected_substring, content)

  def test_uses_base_image(self):
    content = _generate_dockerfile(
      base_image="python:3.11-bullseye",
      requirements_path=None,
      category="cpu",
    )
    self.assertIn("FROM python:3.11-bullseye", content)


class TestImageExists(parameterized.TestCase):
  def test_returns_true_when_tag_found(self):
    mock_client = MagicMock()
    with mock.patch(
      "keras_remote.infra.container_builder.artifactregistry_v1.ArtifactRegistryClient",
      return_value=mock_client,
    ):
      result = _image_exists(
        "us-docker.pkg.dev/my-proj/keras-remote/base:gpu-abc123",
        "my-proj",
      )
    self.assertTrue(result)
    mock_client.get_tag.assert_called_once()

  @parameterized.named_parameters(
    dict(
      testcase_name="not_found",
      side_effect=google_exceptions.NotFound("nope"),
    ),
    dict(
      testcase_name="other_error",
      side_effect=RuntimeError("unexpected"),
    ),
  )
  def test_returns_false_on_error(self, side_effect):
    mock_client = MagicMock()
    mock_client.get_tag.side_effect = side_effect
    with mock.patch(
      "keras_remote.infra.container_builder.artifactregistry_v1.ArtifactRegistryClient",
      return_value=mock_client,
    ):
      result = _image_exists(
        "us-docker.pkg.dev/my-proj/keras-remote/base:gpu-abc123",
        "my-proj",
      )
    self.assertFalse(result)

  def test_correct_resource_name(self):
    mock_client = MagicMock()
    with mock.patch(
      "keras_remote.infra.container_builder.artifactregistry_v1.ArtifactRegistryClient",
      return_value=mock_client,
    ):
      _image_exists(
        "us-docker.pkg.dev/my-proj/keras-remote/base:tpu-abc123def456",
        "my-proj",
      )
    call_args = mock_client.get_tag.call_args
    request = call_args.kwargs["request"]
    self.assertEqual(
      request.name,
      "projects/my-proj/locations/us"
      "/repositories/keras-remote"
      "/packages/base/tags/tpu-abc123def456",
    )


class TestGetOrBuildContainer(absltest.TestCase):
  def test_returns_cached_when_image_exists(self):
    with (
      mock.patch(
        "keras_remote.infra.container_builder._image_exists",
        return_value=True,
      ),
      mock.patch(
        "keras_remote.infra.container_builder._build_and_push",
      ) as mock_build,
    ):
      result = get_or_build_container(
        base_image="python:3.12-slim",
        requirements_path=None,
        accelerator_type="l4",
        project="test-proj",
        zone="us-central1-a",
        cluster_name="my-cluster",
      )

    mock_build.assert_not_called()
    self.assertIn("us-docker.pkg.dev/test-proj/kr-my-cluster/base:", result)

  def test_builds_when_image_missing(self):
    with (
      mock.patch(
        "keras_remote.infra.container_builder._image_exists",
        return_value=False,
      ),
      mock.patch(
        "keras_remote.infra.container_builder._build_and_push",
        return_value="us-docker.pkg.dev/proj/kr-my-cluster/base:gpu-bbbbbbbbbbbb",
      ) as mock_build,
    ):
      result = get_or_build_container(
        base_image="python:3.12-slim",
        requirements_path=None,
        accelerator_type="l4",
        project="proj",
        zone="us-central1-a",
        cluster_name="my-cluster",
      )

    mock_build.assert_called_once()
    self.assertEqual(
      result,
      "us-docker.pkg.dev/proj/kr-my-cluster/base:gpu-bbbbbbbbbbbb",
    )

  def _get_image_uri(self, accelerator_type, project, zone):
    with mock.patch(
      "keras_remote.infra.container_builder._image_exists",
      return_value=True,
    ):
      return get_or_build_container(
        base_image="python:3.12-slim",
        requirements_path=None,
        accelerator_type=accelerator_type,
        project=project,
        zone=zone,
        cluster_name="my-cluster",
      )

  def test_image_uri_format_tpu_europe(self):
    result = self._get_image_uri("v3-4", "my-proj", "europe-west4-b")

    self.assertTrue(
      result.startswith("europe-docker.pkg.dev/my-proj/kr-my-cluster/base:")
    )
    tag = result.split(":")[-1]
    self.assertRegex(tag, r"^tpu-[0-9a-f]{12}$")

  def test_image_uri_format_gpu_us(self):
    result = self._get_image_uri("a100-80gb", "proj", "us-central1-a")

    self.assertTrue(
      result.startswith("us-docker.pkg.dev/proj/kr-my-cluster/base:")
    )
    tag = result.split(":")[-1]
    self.assertRegex(tag, r"^gpu-[0-9a-f]{12}$")


if __name__ == "__main__":
  absltest.main()
