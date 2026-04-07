"""kinetic build-base command — build and push prebuilt base images."""

import concurrent.futures

import click
from google.api_core import exceptions as google_exceptions
from google.cloud import secretmanager

from kinetic.cli.constants import DEFAULT_CLUSTER_NAME
from kinetic.cli.options import common_options
from kinetic.cli.output import banner, console, warning
from kinetic.cli.prompts import resolve_project
from kinetic.infra.container_builder import build_and_push_prebuilt_image
from kinetic.version import __version__

ALL_CATEGORIES = ("cpu", "gpu", "tpu")


def _is_ar_repo(repo):
  """Return True if *repo* is an Artifact Registry URI."""
  return ".pkg.dev" in repo


def _parse_ar_repo(repo):
  """Extract (location, project, repo_name) from an AR Docker URI.

  Expected format: `{location}-docker.pkg.dev/{project}/{repo_name}`
  """
  parts = repo.split("/")
  host = parts[0]  # e.g. "us-docker.pkg.dev"
  location = host.split("-docker.pkg.dev")[0]  # e.g. "us"
  ar_project = parts[1] if len(parts) > 1 else "<project>"
  repo_name = parts[2] if len(parts) > 2 else "<repo>"
  return location, ar_project, repo_name


def _print_ar_setup_instructions(repo, project, cluster_name):
  """Print Artifact Registry setup commands before the confirmation prompt."""
  location, ar_project, repo_name = _parse_ar_repo(repo)
  build_sa = f"kn-{cluster_name}-builds@{project}.iam.gserviceaccount.com"

  console.print(
    "  [bold]Note:[/bold] kinetic does not create or manage this Artifact"
    " Registry repository."
  )
  console.print("  You must create it yourself before running this command.\n")
  console.print("  To create a public AR repository:\n")
  console.print(f"    gcloud artifacts repositories create {repo_name} \\")
  console.print("      --repository-format=docker \\")
  console.print(f"      --location={location} \\")
  console.print(f"      --project={ar_project}\n")
  console.print(
    f"    gcloud artifacts repositories add-iam-policy-binding {repo_name} \\"
  )
  console.print(f"      --location={location} \\")
  console.print(f"      --project={ar_project} \\")
  console.print("      --member=allUsers \\")
  console.print("      --role=roles/artifactregistry.reader\n")
  console.print(
    "  [dim]This grants public read access. To restrict access, replace"
    " allUsers with\n  a specific member"
    " (e.g. serviceAccount:… or user:…).[/dim]\n"
  )
  console.print("  The build service account also needs push access:\n")
  console.print(
    f"    gcloud artifacts repositories add-iam-policy-binding {repo_name} \\"
  )
  console.print(f"      --location={location} \\")
  console.print(f"      --project={ar_project} \\")
  console.print(f"      --member=serviceAccount:{build_sa} \\")
  console.print("      --role=roles/artifactregistry.writer\n")


def _secret_exists(sm_client, project, secret_id):
  """Check if a secret exists in Secret Manager."""
  try:
    sm_client.get_secret(
      request={"name": f"projects/{project}/secrets/{secret_id}"}
    )
    return True
  except google_exceptions.NotFound:
    return False


def _ensure_dockerhub_secrets(project, force_update=False):
  """Check for Docker Hub credentials in Secret Manager, prompt if missing.

  Args:
      project: GCP project ID.
      force_update: If True, prompt for new credentials even if they exist.
  """
  sm_client = secretmanager.SecretManagerServiceClient()
  parent = f"projects/{project}"

  for secret_id, label in [
    ("dockerhub-username", "username"),
    ("dockerhub-token", "access token"),
  ]:
    exists = _secret_exists(sm_client, project, secret_id)

    if exists and not force_update:
      console.print(f"  Using existing Docker Hub {label} secret.")
      continue

    value = click.prompt(
      f"  Enter Docker Hub {label}",
      hide_input=(secret_id == "dockerhub-token"),
    )

    if not exists:
      sm_client.create_secret(
        request={
          "parent": parent,
          "secret_id": secret_id,
          "secret": {"replication": {"automatic": {}}},
        }
      )

    sm_client.add_secret_version(
      request={
        "parent": f"{parent}/secrets/{secret_id}",
        "payload": {"data": value.encode("utf-8")},
      }
    )
    console.print(f"  Docker Hub {label} secret saved.")


def _print_dockerhub_setup_instructions(project):
  """Print Docker Hub setup guidance before the confirmation prompt."""
  console.print("\n  [bold]Docker Hub setup:[/bold]\n")
  console.print(
    "  1. Create a repository at https://hub.docker.com/repositories"
  )
  console.print(
    "  2. Generate an access token at"
    " https://app.docker.com/accounts/<username>/settings/personal-access-tokens\n"
  )
  console.print(
    "  kinetic will store your Docker Hub credentials in"
    f" Secret Manager (project: {project})"
  )
  console.print("  and use them during Cloud Build to push images.\n")


def _prompt_registry_type():
  """Prompt user to choose between Docker Hub and Artifact Registry."""
  click.echo("\nWhere would you like to push images?\n")
  click.echo("  1) Docker Hub")
  click.echo("  2) Artifact Registry")
  choice = click.prompt(
    "\nSelect registry",
    type=click.Choice(["1", "2"]),
  )
  return "docker-hub" if choice == "1" else "artifact-registry"


def _prompt_repo(registry_type, project, cluster_name):
  """Show setup instructions, then prompt for the repository URI."""
  if registry_type == "docker-hub":
    _print_dockerhub_setup_instructions(project)
    return click.prompt(
      "  Enter Docker Hub repository (e.g. 'myuser/kinetic')",
      type=str,
    )

  # AR flow: collect the pieces, show instructions, then confirm the URI.
  console.print("\n  [bold]Artifact Registry setup:[/bold]\n")
  console.print(
    "  You'll need an AR Docker repository. Provide the details below"
    " and kinetic\n  will show you the setup commands.\n"
  )
  location = click.prompt("  AR location (e.g. 'us')", type=str)
  repo_name = click.prompt("  AR repository name (e.g. 'kinetic')", type=str)

  repo = f"{location}-docker.pkg.dev/{project}/{repo_name}"
  _print_ar_setup_instructions(repo, project, cluster_name)

  console.print(f"  Repository URI: [bold]{repo}[/bold]\n")
  return repo


def _prompt_categories():
  """Prompt user to select which accelerator categories to build."""
  click.echo("\nAvailable accelerator categories: cpu, gpu, tpu")
  if click.confirm("Build all categories?", default=True):
    return list(ALL_CATEGORIES)

  selected = []
  for cat in ALL_CATEGORIES:
    if click.confirm(f"  Build {cat}?", default=True):
      selected.append(cat)
  return selected


@click.command("build-base")
@common_options
@click.option(
  "--repo",
  default=None,
  help="Image repository — Docker Hub (e.g. 'kinetic') or Artifact Registry"
  " (e.g. 'us-docker.pkg.dev/my-project/kinetic-base')."
  " Omit to select interactively.",
)
@click.option(
  "--category",
  multiple=True,
  default=None,
  type=click.Choice(["cpu", "gpu", "tpu"]),
  help="Accelerator categories to build (default: all)",
)
@click.option(
  "--tag",
  default=None,
  help="Image version tag (default: kinetic package version)",
)
@click.option(
  "--dockerfile",
  type=click.Path(exists=True, dir_okay=False),
  default=None,
  help="Path to a custom Dockerfile. When set, it is used instead of the "
  "auto-generated one. The Dockerfile must install uv, cloudpickle, "
  "google-cloud-storage, and COPY remote_runner.py to /app/.",
)
@click.option(
  "--update-credentials",
  is_flag=True,
  help="Re-enter Docker Hub credentials even if they already exist",
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def build_base(
  project,
  zone,
  cluster_name,
  repo,
  category,
  tag,
  dockerfile,
  update_credentials,
  yes,
):
  """Build and push prebuilt base images.

  Builds one image per accelerator category (cpu, gpu, tpu) using
  Cloud Build and pushes them to the specified repository. Supports
  both Docker Hub and Artifact Registry — the target is detected
  automatically from the --repo value.

  When --repo is omitted the command enters interactive mode and
  guides you through registry selection and configuration.

  This command works for both kinetic maintainers publishing official
  images and users who want their own base images in a private registry.
  """
  banner("Build Base Images")

  # --- Resolve configuration (interactive when --repo is omitted) ----------
  interactive = repo is None

  if interactive:
    project = project or resolve_project(allow_create=False)
    cluster_name = cluster_name or DEFAULT_CLUSTER_NAME
    registry_type = _prompt_registry_type()
    repo = _prompt_repo(registry_type, project, cluster_name)
    categories = list(category) if category else _prompt_categories()
    tag = tag or click.prompt(
      "\nImage version tag", default=__version__, type=str
    )
  else:
    if not project:
      raise click.UsageError(
        "Missing --project / KINETIC_PROJECT in non-interactive mode."
      )
    categories = list(category) if category else list(ALL_CATEGORIES)
    tag = tag or __version__

  cluster_name = cluster_name or DEFAULT_CLUSTER_NAME
  is_ar = _is_ar_repo(repo)

  if not categories:
    console.print("\nNo categories selected — nothing to build.")
    return

  registry_label = "Artifact Registry" if is_ar else "Docker Hub"

  console.print()
  console.print(f"  Project:    {project}")
  console.print(f"  Repository: {repo}  ({registry_label})")
  console.print(f"  Tag:        {tag}")
  console.print(f"  Categories: {', '.join(categories)}")
  if dockerfile:
    console.print(f"  Dockerfile: {dockerfile}")
  console.print()

  images = [f"{repo}/base-{cat}:{tag}" for cat in categories]
  console.print("Images to build:")
  for img in images:
    console.print(f"  - {img}")
  console.print()

  if is_ar and not interactive:
    _print_ar_setup_instructions(repo, project, cluster_name)

  if not yes:
    click.confirm("Proceed?", abort=True)

  # Credentials — only needed for Docker Hub
  if is_ar:
    if update_credentials:
      console.print(
        "\n  [dim]--update-credentials has no effect for Artifact Registry"
        " (the build service account authenticates automatically).[/dim]"
      )
  else:
    console.print("\nChecking Docker Hub credentials...")
    _ensure_dockerhub_secrets(project, force_update=update_credentials)
  console.print()

  # --- Build and push all categories in parallel --------------------------
  console.print(f"Launching {len(categories)} Cloud Build(s) in parallel...\n")

  def _build_one(cat):
    return build_and_push_prebuilt_image(
      category=cat,
      repo=repo,
      tag=tag,
      project=project,
      cluster_name=cluster_name,
      dockerfile=dockerfile,
    )

  results = []
  with concurrent.futures.ThreadPoolExecutor(
    max_workers=len(categories)
  ) as executor:
    future_to_cat = {
      executor.submit(_build_one, cat): cat for cat in categories
    }
    for future in concurrent.futures.as_completed(future_to_cat):
      cat = future_to_cat[future]
      try:
        pushed_tag = future.result()
        results.append((cat, pushed_tag, None))
        console.print(f"  [green]Pushed {pushed_tag}[/green]")
      except Exception as e:
        results.append((cat, None, str(e)))
        warning(f"  Failed to build {cat}: {e}")

  # --- Summary -------------------------------------------------------------
  console.print()
  banner("Build Summary")
  console.print()
  successes = [r for r in results if r[2] is None]
  failures = [r for r in results if r[2] is not None]

  if successes:
    console.print("Pushed:")
    for _, tag_name, _ in successes:
      console.print(f"  - {tag_name}")

  if failures:
    console.print("\nFailed:")
    for cat, _, err in failures:
      console.print(f"  - {cat}: {err}")

  if successes:
    console.print("\nTo use these images, set the environment variable:")
    console.print(f"  export KINETIC_BASE_IMAGE_REPO={repo}")
    console.print("\nOr pass directly to the decorator:")
    console.print(f'  @kinetic.run(accelerator="l4", base_image_repo="{repo}")')
  console.print()
