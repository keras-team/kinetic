"""Container image building for kinetic."""

import hashlib
import os
import re
import shutil
import string
import sys
import tarfile
import tempfile
import time
import tomllib
import uuid

from absl import logging
from google.api_core import exceptions as google_exceptions
from google.cloud import artifactregistry_v1, storage
from google.cloud.devtools import cloudbuild_v1

from kinetic import version
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
_KEEP_MARKER = "# kn:keep"

_AR_CONSOLE_URL = "https://console.cloud.google.com/artifacts"
_CLOUD_BUILD_CONSOLE_URL = "https://console.cloud.google.com/cloud-build"
_CLOUD_BUILDER_DOCKER = "gcr.io/cloud-builders/docker"
_LIBTPU_FIND_LINKS = (
  "https://storage.googleapis.com/jax-releases/libtpu_releases.html"
)

# JAX install specifiers per accelerator category.
_JAX_INSTALL = {
  "cpu": "jax",
  "tpu": "'jax[tpu]>=0.4.6'",
  "gpu": "'jax[cuda12]'",
}

# Core Python packages installed in every kinetic container image.
_CORE_DEPS = ["keras", "cloudpickle", "google-cloud-storage"]

_DEFAULT_BASE_IMAGE_REPO = "kinetic"


def _filter_jax_requirements(requirements_content: str) -> str:
  """Remove JAX-related packages from requirements content.

  Strips lines that would override the accelerator-specific JAX installation
  managed by the Dockerfile template. Logs a warning for each filtered line.

  To preserve a JAX line, append `# kn:keep` to it in requirements.txt.

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
          "To override, add '# kn:keep' to the line.",
          m.group(1),
        )
        continue

    filtered_lines.append(line)

  return "".join(filtered_lines)


def _parse_pyproject_dependencies(pyproject_path: str) -> str:
  """Extract `[project.dependencies]` from a pyproject.toml file.

  Reads only the core dependency list defined under the `[project]` table.
  Optional dependency groups (`[project.optional-dependencies]`) are ignored;
  users who need those should use a `requirements.txt` instead.

  Args:
      pyproject_path: Absolute path to a `pyproject.toml` file.

  Returns:
      Newline-separated dependency strings in PEP 508 format suitable for
      `uv pip install`, or an empty string if the file declares no
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
  be supplied via a `requirements.txt` or a `pyproject.toml` (from which
  `[project.dependencies]` are extracted).

  Args:
      base_image: Base Docker image (e.g., 'python:3.12-slim')
      requirements_path: Path to requirements.txt or pyproject.toml (or
          None).  When a pyproject.toml is provided,
          `[project.dependencies]` are extracted and used as the
          install list.
      accelerator_type: TPU/GPU type (e.g., 'v3-8')
      project: GCP project ID
      zone: GCP zone for region derivation (defaults to KINETIC_ZONE)
      cluster_name: GKE cluster name (defaults to KINETIC_CLUSTER)

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
  repo_id = f"kn-{cluster_name}"
  registry = f"{ar_location}-docker.pkg.dev/{project}/{repo_id}"
  image_uri = f"{registry}/base:{image_tag}"

  # Check if image exists
  if _image_exists(image_uri, project):
    logging.info("Using cached container: %s", image_uri)
    ar_url = (
      f"{_AR_CONSOLE_URL}/docker/{project}/{ar_location}"
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


def _build_service_account(project: str, cluster_name: str) -> str:
  """Return the full Cloud Build service account resource name."""
  return (
    f"projects/{project}/serviceAccounts/"
    f"kn-{cluster_name}-builds@{project}.iam.gserviceaccount.com"
  )


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

    # Optional requirements file for the build context
    extra_files = {}
    if filtered_requirements is not None:
      req_path = os.path.join(tmpdir, "requirements.txt")
      with open(req_path, "w") as f:
        f.write(filtered_requirements)
      extra_files["requirements.txt"] = req_path

    # Package, upload, and build
    cluster_name = cluster_name or get_default_cluster_name()
    bucket_name = f"{project}-kn-{cluster_name}-builds"
    tarball_path = _pack_build_context(tmpdir, dockerfile_path, extra_files)
    source_gcs = _upload_build_source(tarball_path, bucket_name, project)

    source_object = source_gcs.removeprefix(f"gs://{bucket_name}/")
    build_sa = _build_service_account(project, cluster_name)
    build_config = _ar_build_config(
      image_uri, bucket_name, source_object, build_sa
    )

    _submit_and_wait_build(build_config, project, image_uri)
    ar_url = (
      f"{_AR_CONSOLE_URL}/docker/{project}/{ar_location}"
      f"/kinetic/base?project={project}"
    )
    logging.info("View image: %s", ar_url)
    return image_uri


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
  # Build a single uv pip install command for all dependencies so that
  # uv resolves everything in one pass (faster, more consistent).
  parts = ["RUN uv pip install --system"]

  # JAX with accelerator-specific extras.
  parts.append(_JAX_INSTALL[category])

  # Core dependencies.
  parts.extend(_CORE_DEPS)

  # User requirements.
  if has_requirements:
    parts.append("-r /tmp/requirements.txt")

  # TPU needs an extra find-links index.
  if category == "tpu":
    parts.append(f"-f {_LIBTPU_FIND_LINKS}")

  install_command = " ".join(parts)

  requirements_copy = ""
  if has_requirements:
    requirements_copy = "COPY requirements.txt /tmp/requirements.txt"

  template_path = os.path.join(_PACKAGE_ROOT, "Dockerfile.template")
  with open(template_path, "r") as f:
    template = string.Template(f.read())

  return template.substitute(
    base_image=base_image,
    requirements_copy=requirements_copy,
    install_command=install_command,
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


def get_prebuilt_image(
  accelerator_type: str,
  base_image_repo: str | None = None,
) -> str:
  """Return the Docker Hub URI for a prebuilt base image.

  The image repository is resolved in order:
  1. `base_image_repo` parameter (if not None)
  2. `KINETIC_BASE_IMAGE_REPO` environment variable
  3. Default: `"kinetic"`

  Args:
      accelerator_type: TPU/GPU type (e.g., 'v3-8', 'l4', 'a100').
      base_image_repo: Docker Hub repository override.

  Returns:
      Docker Hub image URI, e.g. `kinetic/base-gpu:0.0.1`.
  """
  repo = (
    base_image_repo
    or os.environ.get("KINETIC_BASE_IMAGE_REPO")
    or _DEFAULT_BASE_IMAGE_REPO
  )
  category = accelerators.get_category(accelerator_type)
  return f"{repo}/base-{category}:{version.__version__}"


def prepare_requirements_content(
  requirements_path: str | None,
) -> str | None:
  """Read, parse, and filter user requirements for runtime installation.

  Handles both `requirements.txt` and `pyproject.toml` (extracting
  `[project.dependencies]`).  JAX-related packages are filtered out
  to prevent conflicts with accelerator-specific installations in the
  prebuilt base image.

  Args:
      requirements_path: Path to `requirements.txt` or `pyproject.toml`,
          or None.

  Returns:
      Filtered requirements content suitable for `uv pip install -r`,
      or None if no dependencies were found.
  """
  if not requirements_path or not os.path.exists(requirements_path):
    return None

  if requirements_path.endswith(".toml"):
    raw = _parse_pyproject_dependencies(requirements_path)
  else:
    with open(requirements_path, "r") as f:
      raw = f.read()

  if not raw:
    return None

  filtered = _filter_jax_requirements(raw)
  return filtered if filtered.strip() else None


def _prepare_dockerfile(
  tmpdir: str, category: str, dockerfile: str | None
) -> str:
  """Write the Dockerfile into *tmpdir* and return its path.

  Uses a caller-supplied file when *dockerfile* is set, otherwise
  auto-generates one with core deps for the given accelerator
  *category*.
  """
  dst = os.path.join(tmpdir, "Dockerfile")
  if dockerfile:
    shutil.copy(dockerfile, dst)
  else:
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    content = _generate_dockerfile(
      base_image=f"python:{py_version}-slim",
      has_requirements=False,
      category=category,
    )
    with open(dst, "w") as f:
      f.write(content)
  return dst


def _pack_build_context(
  tmpdir: str,
  dockerfile_path: str,
  extra_files: dict[str, str] | None = None,
) -> str:
  """Create a Cloud Build source tarball and return its path.

  Always bundles the Dockerfile and `remote_runner.py`.  Additional
  files can be included via *extra_files* mapping `{arcname: local_path}`.
  """
  remote_runner_src = os.path.join(_RUNNER_DIR, REMOTE_RUNNER_FILE_NAME)
  remote_runner_dst = os.path.join(tmpdir, REMOTE_RUNNER_FILE_NAME)
  shutil.copy(remote_runner_src, remote_runner_dst)

  tarball_path = os.path.join(tmpdir, "source.tar.gz")
  with tarfile.open(tarball_path, "w:gz") as tar:
    tar.add(dockerfile_path, arcname="Dockerfile")
    tar.add(remote_runner_dst, arcname=REMOTE_RUNNER_FILE_NAME)
    for arcname, local_path in (extra_files or {}).items():
      tar.add(local_path, arcname=arcname)
  return tarball_path


def _dockerhub_build_config(
  image_tag: str,
  project: str,
  bucket_name: str,
  source_object: str,
  service_account: str,
) -> cloudbuild_v1.Build:
  """Return a Cloud Build config that builds, authenticates, and pushes."""
  return cloudbuild_v1.Build(
    service_account=service_account,
    options=cloudbuild_v1.BuildOptions(
      logging=cloudbuild_v1.BuildOptions.LoggingMode.CLOUD_LOGGING_ONLY,
    ),
    steps=[
      cloudbuild_v1.BuildStep(
        name=_CLOUD_BUILDER_DOCKER,
        args=["build", "-t", image_tag, "."],
      ),
      cloudbuild_v1.BuildStep(
        name=_CLOUD_BUILDER_DOCKER,
        entrypoint="bash",
        args=[
          "-c",
          "echo $$DOCKERHUB_TOKEN | docker login"
          " -u $$DOCKERHUB_USERNAME --password-stdin",
        ],
        secret_env=["DOCKERHUB_USERNAME", "DOCKERHUB_TOKEN"],
      ),
      cloudbuild_v1.BuildStep(
        name=_CLOUD_BUILDER_DOCKER,
        args=["push", image_tag],
      ),
    ],
    available_secrets=cloudbuild_v1.Secrets(
      secret_manager=[
        cloudbuild_v1.SecretManagerSecret(
          version_name=(
            f"projects/{project}/secrets/dockerhub-username/versions/latest"
          ),
          env="DOCKERHUB_USERNAME",
        ),
        cloudbuild_v1.SecretManagerSecret(
          version_name=(
            f"projects/{project}/secrets/dockerhub-token/versions/latest"
          ),
          env="DOCKERHUB_TOKEN",
        ),
      ]
    ),
    source=cloudbuild_v1.Source(
      storage_source=cloudbuild_v1.StorageSource(
        bucket=bucket_name,
        object_=source_object,
      )
    ),
  )


def _submit_and_wait_build(
  build_config: cloudbuild_v1.Build,
  project: str,
  image_tag: str,
  timeout: int = 1200,
) -> None:
  """Submit a Cloud Build, log progress, and wait for completion.

  Raises:
      RuntimeError: If the build does not succeed.
  """
  build_client = cloudbuild_v1.CloudBuildClient()

  logging.info("Submitting Cloud Build for %s...", image_tag)
  operation = build_client.create_build(project_id=project, build=build_config)

  metadata = getattr(operation, "metadata", None)
  build_id = metadata.build.id if metadata is not None else None
  if build_id:
    logging.info("Build ID: %s", build_id)
    logging.info(
      "View build: %s/builds/%s?project=%s",
      _CLOUD_BUILD_CONSOLE_URL,
      build_id,
      project,
    )

  logging.info(
    "Building and pushing %s (this may take 5-10 minutes)...", image_tag
  )
  result = operation.result(timeout=timeout)

  if result is None:
    raise RuntimeError("Cloud Build returned no result")
  if result.status != cloudbuild_v1.Build.Status.SUCCESS:
    raise RuntimeError(f"Build failed with status: {result.status}")

  logging.info("Pushed %s successfully", image_tag)


def _ar_build_config(
  image_tag: str,
  bucket_name: str,
  source_object: str,
  service_account: str,
) -> cloudbuild_v1.Build:
  """Return a Cloud Build config that builds and pushes to Artifact Registry."""
  return cloudbuild_v1.Build(
    service_account=service_account,
    options=cloudbuild_v1.BuildOptions(
      logging=cloudbuild_v1.BuildOptions.LoggingMode.CLOUD_LOGGING_ONLY,
    ),
    steps=[
      cloudbuild_v1.BuildStep(
        name=_CLOUD_BUILDER_DOCKER,
        args=["build", "-t", image_tag, "."],
      ),
    ],
    images=[image_tag],
    source=cloudbuild_v1.Source(
      storage_source=cloudbuild_v1.StorageSource(
        bucket=bucket_name,
        object_=source_object,
      )
    ),
  )


def build_and_push_prebuilt_image(
  category: str,
  repo: str,
  tag: str,
  project: str,
  cluster_name: str | None = None,
  dockerfile: str | None = None,
) -> str:
  """Build a prebuilt base image via Cloud Build and push to a registry.

  Supports both Docker Hub and Artifact Registry targets.  The registry
  type is detected from *repo*: values containing `".pkg.dev"` are
  treated as Artifact Registry URIs; everything else as Docker Hub.

  When no custom *dockerfile* is provided, a Dockerfile with core
  dependencies (JAX + keras + cloudpickle + google-cloud-storage) is
  auto-generated.  A custom *dockerfile* is used verbatim — the caller
  must ensure it installs `uv`, `cloudpickle`,
  `google-cloud-storage` and copies `remote_runner.py` to
  `/app/remote_runner.py`.

  For Docker Hub, credentials are read from GCP Secret Manager
  (`dockerhub-username` and `dockerhub-token`).  For Artifact
  Registry, the build service account pushes directly (no extra
  credentials needed).

  Args:
      category: Accelerator category (`"cpu"`, `"gpu"`, or `"tpu"`).
      repo: Image repository — Docker Hub (e.g. `"kinetic"`) or
          Artifact Registry (e.g.
          `"us-docker.pkg.dev/my-project/kinetic-base"`).
      tag: Image version tag (e.g., `"0.0.1"`).
      project: GCP project ID for Cloud Build.
      cluster_name: GKE cluster name (for builds bucket).
      dockerfile: Optional path to a custom Dockerfile.

  Returns:
      The full image tag that was pushed.
  """
  cluster_name = cluster_name or get_default_cluster_name()
  bucket_name = f"{project}-kn-{cluster_name}-builds"
  image_tag = f"{repo}/base-{category}:{tag}"

  with tempfile.TemporaryDirectory() as tmpdir:
    dockerfile_path = _prepare_dockerfile(tmpdir, category, dockerfile)
    tarball_path = _pack_build_context(tmpdir, dockerfile_path)
    source_gcs = _upload_build_source(tarball_path, bucket_name, project)

    source_object = source_gcs.removeprefix(f"gs://{bucket_name}/")
    build_sa = _build_service_account(project, cluster_name)
    if ".pkg.dev" in repo:
      build_config = _ar_build_config(
        image_tag, bucket_name, source_object, build_sa
      )
    else:
      build_config = _dockerhub_build_config(
        image_tag, project, bucket_name, source_object, build_sa
      )
    _submit_and_wait_build(build_config, project, image_tag)

  return image_tag
