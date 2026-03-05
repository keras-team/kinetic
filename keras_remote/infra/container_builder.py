"""Container image building for keras_remote."""

import hashlib
import os
import shutil
import string
import tarfile
import tempfile
import time
import uuid

from absl import logging
from google.api_core import exceptions as google_exceptions
from google.cloud import artifactregistry_v1, storage
from google.cloud.devtools import cloudbuild_v1

from keras_remote.constants import get_default_zone, zone_to_ar_location
from keras_remote.core import accelerators

REMOTE_RUNNER_FILE_NAME = "remote_runner.py"
# Paths relative to this file's location (keras_remote/infra/)
_PACKAGE_ROOT = os.path.normpath(
  os.path.join(os.path.dirname(__file__), os.pardir)
)
_RUNNER_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "runner")


def get_or_build_container(
  base_image, requirements_path, accelerator_type, project, zone=None
):
  """Get existing container or build if requirements changed.

  Uses content-based hashing to detect requirement changes.

  Args:
      base_image: Base Docker image (e.g., 'python:3.12-slim')
      requirements_path: Path to requirements.txt (or None)
      accelerator_type: TPU/GPU type (e.g., 'v3-8')
      project: GCP project ID
      zone: GCP zone for region derivation (defaults to KERAS_REMOTE_ZONE)

  Returns:
      Container image URI in Artifact Registry
  """
  ar_location = zone_to_ar_location(zone or get_default_zone())
  category = accelerators.get_category(accelerator_type)

  # Generate deterministic hash from requirements + base image + category
  requirements_hash = _hash_requirements(
    requirements_path, category, base_image
  )

  # Use category for image name (e.g., 'tpu-hash', 'gpu-hash')
  image_tag = f"{category}-{requirements_hash[:12]}"

  # Use Artifact Registry
  registry = f"{ar_location}-docker.pkg.dev/{project}/keras-remote"
  image_uri = f"{registry}/base:{image_tag}"

  # Check if image exists
  if _image_exists(image_uri, project):
    logging.info("Using cached container: %s", image_uri)
    ar_url = (
      "https://console.cloud.google.com/artifacts"
      f"/docker/{project}/{ar_location}"
      f"/keras-remote/base?project={project}"
    )
    logging.info("View image: %s", ar_url)
    return image_uri

  # Build new image
  logging.info("Building new container (requirements changed): %s", image_uri)
  return _build_and_push(
    base_image,
    requirements_path,
    category,
    project,
    image_uri,
    ar_location,
  )


def _hash_requirements(requirements_path, category, base_image):
  """Create deterministic hash from requirements + category + remote_runner + base image.

  Args:
      requirements_path: Path to requirements.txt (or None)
      category: Accelerator category ('cpu', 'gpu', 'tpu')
      base_image: Base Docker image (e.g., 'python:3.12-slim')

  Returns:
      SHA256 hex digest
  """
  content = f"base_image={base_image}\ncategory={category}\n"

  if requirements_path and os.path.exists(requirements_path):
    with open(requirements_path, "r") as f:
      content += f.read()

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


def _image_exists(image_uri, project):
  """Check if image exists in Artifact Registry.

  Args:
      image_uri: Full image URI
          (e.g., 'us-docker.pkg.dev/my-project/keras-remote/base:tag')
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
  base_image,
  requirements_path,
  category,
  project,
  image_uri,
  ar_location="us",
):
  """Build and push Docker image using Cloud Build.

  Args:
      base_image: Base Docker image
      requirements_path: Path to requirements.txt (or None)
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
      requirements_path=requirements_path,
      category=category,
    )

    dockerfile_path = os.path.join(tmpdir, "Dockerfile")
    with open(dockerfile_path, "w") as f:
      f.write(dockerfile_content)

    # Copy requirements.txt if it exists
    if requirements_path and os.path.exists(requirements_path):
      shutil.copy(requirements_path, os.path.join(tmpdir, "requirements.txt"))

    # Copy remote_runner.py
    remote_runner_src = os.path.join(_RUNNER_DIR, REMOTE_RUNNER_FILE_NAME)
    remote_runner_dst = os.path.join(tmpdir, REMOTE_RUNNER_FILE_NAME)
    shutil.copy(remote_runner_src, remote_runner_dst)

    # Create tarball for Cloud Build
    tarball_path = os.path.join(tmpdir, "source.tar.gz")
    with tarfile.open(tarball_path, "w:gz") as tar:
      tar.add(dockerfile_path, arcname="Dockerfile")
      tar.add(remote_runner_dst, arcname=REMOTE_RUNNER_FILE_NAME)
      if requirements_path and os.path.exists(requirements_path):
        tar.add(
          os.path.join(tmpdir, "requirements.txt"), arcname="requirements.txt"
        )

    # Upload source to GCS
    bucket_name = f"{project}-keras-remote-builds"
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
        f"/keras-remote/base?project={project}"
      )
      logging.info("View image: %s", ar_url)
      return image_uri
    else:
      raise RuntimeError(f"Build failed with status: {result.status}")


def _generate_dockerfile(base_image, requirements_path, category):
  """Generate Dockerfile content based on configuration.

  Args:
      base_image: Base Docker image
      requirements_path: Path to requirements.txt (or None)
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
  if requirements_path and os.path.exists(requirements_path):
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


def _upload_build_source(tarball_path, bucket_name, project):
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
