"""Container image building for kinetic."""

import hashlib
import os
import re
import shutil
import string
import tarfile
import tempfile
import time
import tomllib
import uuid

from absl import logging
from google.api_core import exceptions as google_exceptions
from google.cloud import artifactregistry_v1, storage
from google.cloud.devtools import cloudbuild_v1

from kinetic.constants import (
  get_default_cluster_name,
  get_default_zone,
  zone_to_ar_location,
)
from kinetic.core import accelerators

REMOTE_RUNNER_FILE_NAME = "remote_runner.py"
# Paths relative to this file's location (kinetic/infra/)
_PACKAGE_ROOT = os.path.normpath(
  os.path.join(os.path.dirname(__file__), os.pardir)
)
_RUNNER_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "runner")

# JAX-related packages managed by the Dockerfile template.
# User requirements containing these are filtered out to prevent overriding
# the accelerator-specific JAX installation (e.g., jax[tpu], jax[cuda12]).
_JAX_PACKAGE_NAMES = frozenset({"jax", "jaxlib", "libtpu", "libtpu-nightly"})
_PACKAGE_NAME_RE = re.compile(r"^([a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?)")
_KEEP_MARKER = "# kr:keep"


def _filter_jax_requirements(requirements_content: str) -> str:
  """Remove JAX-related packages from requirements content.

  Strips lines that would override the accelerator-specific JAX installation
  managed by the Dockerfile template. Logs a warning for each filtered line.

  To preserve a JAX line, append ``# kr:keep`` to it in requirements.txt.

  Args:
      requirements_content: Raw text of a requirements.txt file.

  Returns:
      Filtered requirements text with JAX-related lines removed.
  """
  filtered_lines = []
  for line in requirements_content.splitlines(keepends=True):
    stripped = line.strip()
    # Preserve blanks, comments, and pip flags (-e, --index-url, etc.)
    if not stripped or stripped.startswith("#") or stripped.startswith("-"):
      filtered_lines.append(line)
      continue

    # Allow users to bypass the filter with an inline marker.
    if _KEEP_MARKER in line:
      filtered_lines.append(line)
      continue

    m = _PACKAGE_NAME_RE.match(stripped)
    if m:
      # PEP 503 normalization: lowercase, collapse [-_.] to '-'
      normalized = re.sub(r"[-_.]+", "-", m.group(1)).lower()
      if normalized in _JAX_PACKAGE_NAMES:
        logging.warning(
          "Filtered '%s' from requirements — JAX is installed "
          "automatically with the correct accelerator backend. "
          "To override, add '# kr:keep' to the line.",
          m.group(1),
        )
        continue

    filtered_lines.append(line)

  return "".join(filtered_lines)


def _parse_pyproject_dependencies(pyproject_path: str) -> str:
  """Extract ``[project.dependencies]`` from a pyproject.toml file.

  Reads only the core dependency list defined under the ``[project]`` table.
  Optional dependency groups (``[project.optional-dependencies]``) are ignored;
  users who need those should use a ``requirements.txt`` instead.

  Args:
      pyproject_path: Absolute path to a ``pyproject.toml`` file.

  Returns:
      Newline-separated dependency strings in PEP 508 format suitable for
      ``pip install``, or an empty string if the file declares no
      dependencies.
  """
  with open(pyproject_path, "rb") as f:
    data = tomllib.load(f)

  deps = data.get("project", {}).get("dependencies", [])
  if not deps:
    return ""
  return "\n".join(deps) + "\n"


def get_or_build_container(
  base_image: str,
  requirements_path: str | None,
  accelerator_type: str,
  project: str,
  zone: str | None = None,
  cluster_name: str | None = None,
) -> str:
  """Get existing container or build if requirements changed.

  Uses content-based hashing to detect requirement changes. Dependencies can
  be supplied via a ``requirements.txt`` or a ``pyproject.toml`` (from which
  ``[project.dependencies]`` are extracted).

  Args:
      base_image: Base Docker image (e.g., 'python:3.12-slim')
      requirements_path: Path to requirements.txt or pyproject.toml (or
          None).  When a pyproject.toml is provided,
          ``[project.dependencies]`` are extracted and used as the
          install list.
      accelerator_type: TPU/GPU type (e.g., 'v3-8')
      project: GCP project ID
      zone: GCP zone for region derivation (defaults to KERAS_REMOTE_ZONE)
      cluster_name: GKE cluster name (defaults to KERAS_REMOTE_CLUSTER)

  Returns:
      Container image URI in Artifact Registry
  """
  ar_location = zone_to_ar_location(zone or get_default_zone())
  cluster_name = cluster_name or get_default_cluster_name()
  category = accelerators.get_category(accelerator_type)

  # Read and filter requirements once, reuse for hashing and building.
  filtered_requirements = None
  if requirements_path and os.path.exists(requirements_path):
    if requirements_path.endswith(".toml"):
      raw_requirements = _parse_pyproject_dependencies(requirements_path)
    else:
      with open(requirements_path, "r") as f:
        raw_requirements = f.read()
    if raw_requirements:
      filtered_requirements = _filter_jax_requirements(raw_requirements)

  # Generate deterministic hash from requirements + base image + category
  requirements_hash = _hash_requirements(
    filtered_requirements, category, base_image
  )

  # Use category for image name (e.g., 'tpu-hash', 'gpu-hash')
  image_tag = f"{category}-{requirements_hash[:12]}"

  # Use Artifact Registry (cluster-scoped repo)
  repo_id = f"kr-{cluster_name}"
  registry = f"{ar_location}-docker.pkg.dev/{project}/{repo_id}"
  image_uri = f"{registry}/base:{image_tag}"

  # Check if image exists
  if _image_exists(image_uri, project):
    logging.info("Using cached container: %s", image_uri)
    ar_url = (
      "https://console.cloud.google.com/artifacts"
      f"/docker/{project}/{ar_location}"
      f"/{repo_id}/base?project={project}"
    )
    logging.info("View image: %s", ar_url)
    return image_uri

  # Build new image
  logging.info("Building new container (requirements changed): %s", image_uri)
  return _build_and_push(
    base_image,
    filtered_requirements,
    category,
    project,
    image_uri,
    ar_location,
    cluster_name,
  )


def _hash_requirements(
  filtered_requirements: str | None, category: str, base_image: str
) -> str:
  """Create deterministic hash from requirements + category + remote_runner + base image.

  Args:
      filtered_requirements: Pre-filtered requirements content (or None)
      category: Accelerator category ('cpu', 'gpu', 'tpu')
      base_image: Base Docker image (e.g., 'python:3.12-slim')

  Returns:
      SHA256 hex digest
  """
  content = f"base_image={base_image}\ncategory={category}\n"

  if filtered_requirements:
    content += filtered_requirements

  # Include remote_runner.py in the hash so container rebuilds when it changes
  remote_runner_path = os.path.join(_RUNNER_DIR, REMOTE_RUNNER_FILE_NAME)
  if os.path.exists(remote_runner_path):
    with open(remote_runner_path, "r") as f:
      content += f"\n---{REMOTE_RUNNER_FILE_NAME}---\n{f.read()}"

  # Include Dockerfile template in the hash so container rebuilds when it changes
  template_path = os.path.join(_PACKAGE_ROOT, "Dockerfile.template")
  if os.path.exists(template_path):
    with open(template_path, "r") as f:
      content += f"\n---Dockerfile.template---\n{f.read()}"

  return hashlib.sha256(content.encode()).hexdigest()


def _image_exists(image_uri: str, project: str) -> bool:
  """Check if image exists in Artifact Registry.

  Args:
      image_uri: Full image URI
          (e.g., 'us-docker.pkg.dev/my-project/kinetic/base:tag')
      project: GCP project ID

  Returns:
      True if image exists, False otherwise
  """
  try:
    # Parse: {location}-docker.pkg.dev/{project}/{repo}/{image}:{tag}
    host, _, repo, image_and_tag = image_uri.split("/", 3)
    location = host.split("-docker.pkg.dev")[0]
    image, tag = image_and_tag.split(":", 1)

    # Look up the tag directly — dockerImages resources use digests,
    # not tags, so get_docker_image cannot resolve image:tag URIs.
    name = (
      f"projects/{project}/locations/{location}"
      f"/repositories/{repo}/packages/{image}/tags/{tag}"
    )
    client = artifactregistry_v1.ArtifactRegistryClient()
    client.get_tag(
      request=artifactregistry_v1.GetTagRequest(name=name),
    )
    return True

  except google_exceptions.NotFound:
    return False
  except Exception:
    logging.warning("Unexpected error checking image existence", exc_info=True)
    return False


def _build_and_push(
  base_image: str,
  filtered_requirements: str | None,
  category: str,
  project: str,
  image_uri: str,
  ar_location: str = "us",
  cluster_name: str | None = None,
) -> str:
  """Build and push Docker image using Cloud Build.

  Args:
      base_image: Base Docker image
      filtered_requirements: Pre-filtered requirements content (or None)
      category: Accelerator category ('cpu', 'gpu', 'tpu')
      project: GCP project ID
      image_uri: Target image URI
      ar_location: Artifact Registry multi-region (e.g., 'us')

  Returns:
      Image URI
  """
  with tempfile.TemporaryDirectory() as tmpdir:
    # Generate Dockerfile
    dockerfile_content = _generate_dockerfile(
      base_image=base_image,
      has_requirements=filtered_requirements is not None,
      category=category,
    )

    dockerfile_path = os.path.join(tmpdir, "Dockerfile")
    with open(dockerfile_path, "w") as f:
      f.write(dockerfile_content)

    # Write pre-filtered requirements.txt
    if filtered_requirements is not None:
      with open(os.path.join(tmpdir, "requirements.txt"), "w") as f:
        f.write(filtered_requirements)

    # Copy remote_runner.py
    remote_runner_src = os.path.join(_RUNNER_DIR, REMOTE_RUNNER_FILE_NAME)
    remote_runner_dst = os.path.join(tmpdir, REMOTE_RUNNER_FILE_NAME)
    shutil.copy(remote_runner_src, remote_runner_dst)

    # Create tarball for Cloud Build
    tarball_path = os.path.join(tmpdir, "source.tar.gz")
    with tarfile.open(tarball_path, "w:gz") as tar:
      tar.add(dockerfile_path, arcname="Dockerfile")
      tar.add(remote_runner_dst, arcname=REMOTE_RUNNER_FILE_NAME)
      if filtered_requirements is not None:
        tar.add(
          os.path.join(tmpdir, "requirements.txt"), arcname="requirements.txt"
        )

    # Upload source to GCS (cluster-scoped bucket)
    cluster_name = cluster_name or get_default_cluster_name()
    bucket_name = f"{project}-kr-{cluster_name}-builds"
    source_gcs = _upload_build_source(tarball_path, bucket_name, project)

    # Submit build to Cloud Build
    build_client = cloudbuild_v1.CloudBuildClient()

    build_config = cloudbuild_v1.Build(
      steps=[
        cloudbuild_v1.BuildStep(
          name="gcr.io/cloud-builders/docker",
          args=["build", "-t", image_uri, "."],
        ),
      ],
      images=[image_uri],
      source=cloudbuild_v1.Source(
        storage_source=cloudbuild_v1.StorageSource(
          bucket=bucket_name,
          object_=source_gcs.split(f"gs://{bucket_name}/")[1],
        )
      ),
    )

    logging.info("Submitting build to Cloud Build...")
    operation = build_client.create_build(
      project_id=project, build=build_config
    )

    # Get build ID from the operation metadata
    build_id = (
      operation.metadata.build.id if hasattr(operation, "metadata") else None
    )
    if build_id:
      logging.info("Build ID: %s", build_id)
      logging.info(
        "View build: https://console.cloud.google.com/cloud-build/builds/%s?project=%s",
        build_id,
        project,
      )

    logging.info("Building container image (this may take 5-10 minutes)...")
    result = operation.result(timeout=1200)  # 20 minute timeout

    if result.status == cloudbuild_v1.Build.Status.SUCCESS:
      logging.info("Container built successfully: %s", image_uri)
      ar_url = (
        "https://console.cloud.google.com/artifacts"
        f"/docker/{project}/{ar_location}"
        f"/kinetic/base?project={project}"
      )
      logging.info("View image: %s", ar_url)
      return image_uri
    else:
      raise RuntimeError(f"Build failed with status: {result.status}")


def _generate_dockerfile(
  base_image: str, has_requirements: bool, category: str
) -> str:
  """Generate Dockerfile content based on configuration.

  Args:
      base_image: Base Docker image
      has_requirements: Whether filtered requirements content is available
      category: Accelerator category ('cpu', 'gpu', 'tpu')

  Returns:
      Dockerfile content as string
  """
  # Determine JAX installation command based on accelerator category
  if category == "cpu":
    jax_install = "RUN python3 -m pip install jax"
  elif category == "tpu":
    jax_install = (
      "RUN python3 -m pip install 'jax[tpu]>=0.4.6' "
      "-f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
    )
  else:
    jax_install = "RUN python3 -m pip install 'jax[cuda12]'"

  requirements_section = ""
  if has_requirements:
    requirements_section = (
      "COPY requirements.txt /tmp/requirements.txt\n"
      "RUN python3 -m pip install -r /tmp/requirements.txt\n"
    )

  template_path = os.path.join(_PACKAGE_ROOT, "Dockerfile.template")
  with open(template_path, "r") as f:
    template = string.Template(f.read())

  return template.substitute(
    base_image=base_image,
    jax_install=jax_install,
    requirements_section=requirements_section,
  )


def _upload_build_source(
  tarball_path: str, bucket_name: str, project: str
) -> str:
  """Upload build source tarball to Cloud Storage.

  Args:
      tarball_path: Local path to tarball
      bucket_name: GCS bucket name
      project: GCP project ID

  Returns:
      GCS URI of uploaded tarball
  """
  client = storage.Client(project=project)
  bucket = client.bucket(bucket_name)

  # Upload tarball
  blob_name = f"source-{int(time.time())}-{uuid.uuid4().hex[:8]}.tar.gz"
  blob = bucket.blob(blob_name)
  blob.upload_from_filename(tarball_path)

  gcs_uri = f"gs://{bucket_name}/{blob_name}"
  logging.info("Uploaded build source to %s", gcs_uri)
  logging.info(
    "View source: https://console.cloud.google.com/storage/browser/%s?project=%s",
    bucket_name,
    project,
  )

  return gcs_uri
