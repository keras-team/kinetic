"""Tests for kinetic.infra.container_builder — hashing, Dockerfile gen, caching."""

import os
import tempfile
from unittest import mock
from unittest.mock import MagicMock

from absl.testing import absltest, parameterized
from google.api_core import exceptions as google_exceptions

from kinetic.infra.container_builder import (
  _filter_jax_requirements,
  _generate_dockerfile,
  _hash_requirements,
  _image_exists,
  _parse_pyproject_dependencies,
  get_or_build_container,
)


class TestFilterJaxRequirements(parameterized.TestCase):
  @parameterized.named_parameters(
    dict(testcase_name="bare_jax", line="jax\n"),
    dict(testcase_name="jax_with_tpu_extras", line="jax[tpu]>=0.4.6\n"),
    dict(testcase_name="jax_cuda", line="jax[cuda12]==0.4.30\n"),
    dict(testcase_name="jax_cpu", line="jax[cpu]\n"),
    dict(testcase_name="jaxlib", line="jaxlib>=0.4.6\n"),
    dict(testcase_name="libtpu", line="libtpu\n"),
    dict(testcase_name="libtpu_nightly_hyphen", line="libtpu-nightly\n"),
    dict(testcase_name="libtpu_nightly_underscore", line="libtpu_nightly\n"),
    dict(testcase_name="jax_uppercase", line="JAX\n"),
    dict(testcase_name="jax_mixed_case", line="Jax[tpu]\n"),
  )
  def test_filters_jax_packages(self, line):
    self.assertEqual(_filter_jax_requirements(line), "")

  @parameterized.named_parameters(
    dict(testcase_name="numpy", line="numpy==1.26\n"),
    dict(testcase_name="keras", line="keras\n"),
    dict(testcase_name="scipy", line="scipy>=1.12\n"),
    dict(testcase_name="comment", line="# jax should be here\n"),
    dict(testcase_name="blank", line="\n"),
    dict(testcase_name="pip_flag", line="-e git+https://foo\n"),
    dict(testcase_name="index_url", line="--index-url https://pypi.org\n"),
  )
  def test_preserves_non_jax_packages(self, line):
    self.assertEqual(_filter_jax_requirements(line), line)

  @parameterized.named_parameters(
    dict(testcase_name="jax_keep", line="jax==0.4.30  # kr:keep\n"),
    dict(testcase_name="jaxlib_keep", line="jaxlib  # kr:keep\n"),
    dict(testcase_name="libtpu_keep", line="libtpu-nightly  # kr:keep\n"),
  )
  def test_kr_keep_overrides_filter(self, line):
    self.assertEqual(_filter_jax_requirements(line), line)

  def test_mixed_requirements(self):
    content = (
      "numpy==1.26\njax[tpu]>=0.4.6\nscipy\n"
      "jaxlib\nkeras\njax==0.4.30  # kr:keep\n"
    )
    result = _filter_jax_requirements(content)
    self.assertEqual(
      result, "numpy==1.26\nscipy\nkeras\njax==0.4.30  # kr:keep\n"
    )

  def test_empty_string(self):
    self.assertEqual(_filter_jax_requirements(""), "")

  def test_only_jax_packages(self):
    self.assertEqual(_filter_jax_requirements("jax\njaxlib\nlibtpu\n"), "")

  def test_preserves_comments_and_blanks(self):
    content = "# ML deps\nnumpy\n\njax\n# end\n"
    result = _filter_jax_requirements(content)
    self.assertEqual(result, "# ML deps\nnumpy\n\n# end\n")


class TestParsePyprojectDependencies(absltest.TestCase):
  def _write_toml(self, content):
    """Write content to a temp pyproject.toml and return its path."""
    td = tempfile.TemporaryDirectory()
    self.addCleanup(td.cleanup)
    path = os.path.join(td.name, "pyproject.toml")
    with open(path, "w") as f:
      f.write(content)
    return path

  def test_extracts_dependencies(self):
    path = self._write_toml(
      '[project]\ndependencies = ["numpy>=1.20", "pandas"]\n'
    )
    result = _parse_pyproject_dependencies(path)
    self.assertEqual(result, "numpy>=1.20\npandas\n")

  def test_returns_empty_when_no_dependencies(self):
    path = self._write_toml("[project]\nname = 'foo'\n")
    self.assertEqual(_parse_pyproject_dependencies(path), "")

  def test_returns_empty_when_no_project_table(self):
    path = self._write_toml("[tool.ruff]\nline-length = 88\n")
    self.assertEqual(_parse_pyproject_dependencies(path), "")

  def test_returns_empty_for_empty_dependencies(self):
    path = self._write_toml("[project]\ndependencies = []\n")
    self.assertEqual(_parse_pyproject_dependencies(path), "")

  def test_ignores_optional_dependencies(self):
    path = self._write_toml(
      '[project]\ndependencies = ["numpy"]\n\n'
      '[project.optional-dependencies]\ndev = ["pytest"]\n'
    )
    result = _parse_pyproject_dependencies(path)
    self.assertEqual(result, "numpy\n")


class TestHashRequirements(parameterized.TestCase):
  def test_deterministic(self):
    h1 = _hash_requirements("numpy==1.26\n", "gpu", "python:3.12-slim")
    h2 = _hash_requirements("numpy==1.26\n", "gpu", "python:3.12-slim")
    self.assertEqual(h1, h2)

  def test_different_requirements_different_hash(self):
    h1 = _hash_requirements("numpy==1.26\n", "gpu", "python:3.12-slim")
    h2 = _hash_requirements("scipy==1.12\n", "gpu", "python:3.12-slim")
    self.assertNotEqual(h1, h2)

  def test_different_category_different_hash(self):
    h1 = _hash_requirements("numpy\n", "gpu", "python:3.12-slim")
    h2 = _hash_requirements("numpy\n", "tpu", "python:3.12-slim")
    self.assertNotEqual(h1, h2)

  def test_different_base_image_different_hash(self):
    h1 = _hash_requirements("numpy\n", "gpu", "python:3.12-slim")
    h2 = _hash_requirements("numpy\n", "gpu", "python:3.11-slim")
    self.assertNotEqual(h1, h2)

  def test_missing_requirements_valid(self):
    h = _hash_requirements(None, "cpu", "python:3.12-slim")
    self.assertIsInstance(h, str)
    self.assertLen(h, 64)

  def test_returns_hex_string(self):
    h = _hash_requirements("keras\n", "gpu", "python:3.12-slim")
    self.assertRegex(h, r"^[0-9a-f]{64}$")

  def test_jax_in_requirements_does_not_affect_hash(self):
    filtered_without_jax = _filter_jax_requirements("numpy==1.26\n")
    filtered_with_jax = _filter_jax_requirements(
      "numpy==1.26\njax[tpu]>=0.4.6\n"
    )

    h1 = _hash_requirements(filtered_without_jax, "tpu", "python:3.12-slim")
    h2 = _hash_requirements(filtered_with_jax, "tpu", "python:3.12-slim")
    self.assertEqual(h1, h2)


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
      has_requirements=False,
      category=category,
    )
    for s in expected:
      self.assertIn(s, content)
    for s in not_expected:
      self.assertNotIn(s, content)

  def test_with_requirements(self):
    content = _generate_dockerfile(
      base_image="python:3.12-slim",
      has_requirements=True,
      category="cpu",
    )
    self.assertIn("COPY requirements.txt", content)
    self.assertIn("pip install -r", content)

  def test_without_requirements(self):
    content = _generate_dockerfile(
      base_image="python:3.12-slim",
      has_requirements=False,
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
      has_requirements=False,
      category="cpu",
    )
    self.assertIn(expected_substring, content)

  def test_uses_base_image(self):
    content = _generate_dockerfile(
      base_image="python:3.11-bullseye",
      has_requirements=False,
      category="cpu",
    )
    self.assertIn("FROM python:3.11-bullseye", content)


class TestImageExists(parameterized.TestCase):
  def test_returns_true_when_tag_found(self):
    mock_client = MagicMock()
    with mock.patch(
      "kinetic.infra.container_builder.artifactregistry_v1.ArtifactRegistryClient",
      return_value=mock_client,
    ):
      result = _image_exists(
        "us-docker.pkg.dev/my-proj/kinetic/base:gpu-abc123",
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
      "kinetic.infra.container_builder.artifactregistry_v1.ArtifactRegistryClient",
      return_value=mock_client,
    ):
      result = _image_exists(
        "us-docker.pkg.dev/my-proj/kinetic/base:gpu-abc123",
        "my-proj",
      )
    self.assertFalse(result)

  def test_correct_resource_name(self):
    mock_client = MagicMock()
    with mock.patch(
      "kinetic.infra.container_builder.artifactregistry_v1.ArtifactRegistryClient",
      return_value=mock_client,
    ):
      _image_exists(
        "us-docker.pkg.dev/my-proj/kinetic/base:tpu-abc123def456",
        "my-proj",
      )
    call_args = mock_client.get_tag.call_args
    request = call_args.kwargs["request"]
    self.assertEqual(
      request.name,
      "projects/my-proj/locations/us"
      "/repositories/kinetic"
      "/packages/base/tags/tpu-abc123def456",
    )


class TestGetOrBuildContainer(absltest.TestCase):
  def test_returns_cached_when_image_exists(self):
    with (
      mock.patch(
        "kinetic.infra.container_builder._image_exists",
        return_value=True,
      ),
      mock.patch(
        "kinetic.infra.container_builder._build_and_push",
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
        "kinetic.infra.container_builder._image_exists",
        return_value=False,
      ),
      mock.patch(
        "kinetic.infra.container_builder._build_and_push",
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
      "kinetic.infra.container_builder._image_exists",
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
