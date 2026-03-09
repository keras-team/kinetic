"""keras-remote down command — tear down infrastructure."""

import click

from keras_remote.cli.config import InfraConfig
from keras_remote.cli.constants import DEFAULT_CLUSTER_NAME, DEFAULT_ZONE
from keras_remote.cli.infra.state import apply_destroy
from keras_remote.cli.options import common_options
from keras_remote.cli.output import banner, console, warning
from keras_remote.cli.prerequisites_check import check_all
from keras_remote.cli.prompts import resolve_project


@click.command()
@common_options
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def down(project, zone, cluster_name, yes):
  """Tear down keras-remote GCP infrastructure."""
  banner("keras-remote Cleanup")

  check_all()

  project = project or resolve_project(allow_create=False)
  zone = zone or DEFAULT_ZONE
  cluster_name = cluster_name or DEFAULT_CLUSTER_NAME

  # Warning
  console.print()
  warning(f"This will delete ALL keras-remote resources in project: {project}")
  console.print()
  console.print("This includes:")
  console.print("  - GKE cluster and node pools")
  console.print("  - Artifact Registry repository and images")
  console.print("  - Cloud Storage buckets (jobs and builds)")
  console.print("  - Enabled API services (left enabled)")
  console.print()

  if not yes:
    click.confirm("Are you sure you want to continue?", abort=True)

  console.print()

  config = InfraConfig(project=project, zone=zone, cluster_name=cluster_name)
  apply_destroy(config)

  # Summary
  console.print()
  banner("Cleanup Complete")
  console.print()
  console.print("Check manually for remaining resources:")
  console.print(
    f"  GKE: https://console.cloud.google.com/kubernetes/list?project={project}"
  )
  console.print(
    f"  Billing: https://console.cloud.google.com/billing?project={project}"
  )
  console.print()
