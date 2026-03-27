"""kinetic jobs command group — inspect and manage async jobs."""

import click
from rich.table import Table

from kinetic.cli.options import cleanup_options, common_options, jobs_options
from kinetic.cli.output import banner, console, success, warning
from kinetic.jobs import attach, list_jobs


def _ensure_project(project):
  """Raise a clear CLI error if project is not set."""
  if not project:
    raise click.UsageError(
      "Project is required. Set --project or KINETIC_PROJECT."
    )


def _attach(job_id, project, cluster_name):
  """Attach to a job, validating required options."""
  _ensure_project(project)
  return attach(job_id, project=project, cluster=cluster_name)


@click.group()
def jobs():
  """Inspect and manage async remote jobs."""


@jobs.command("list")
@jobs_options
def list_command(project, zone, cluster_name, namespace):
  """List live async jobs."""
  _ensure_project(project)

  banner("kinetic Jobs")

  handles = list_jobs(
    project=project,
    zone=zone,
    cluster=cluster_name,
    namespace=namespace,
  )
  if not handles:
    warning("No live jobs found.")
    return

  table = Table(title="Async Jobs")
  table.add_column("Job ID", style="bold")
  table.add_column("Function", style="green")
  table.add_column("Accelerator")
  table.add_column("Backend")
  table.add_column("Created", style="dim")

  for handle in handles:
    table.add_row(
      handle.job_id,
      handle.func_name,
      handle.accelerator,
      handle.backend,
      handle.created_at,
    )

  console.print()
  console.print(table)


@jobs.command()
@click.argument("job_id")
@common_options
def status(job_id, project, zone, cluster_name):
  """Show the current status for a job."""
  handle = _attach(job_id, project, cluster_name)
  console.print(f"{job_id}: {handle.status().value}")


@jobs.command()
@click.argument("job_id")
@click.option(
  "--follow", "-f", is_flag=True, help="Stream logs until completion."
)
@click.option(
  "--tail",
  "-n",
  type=int,
  default=None,
  help="Show the last N log lines instead of the full log.",
)
@common_options
def logs(job_id, follow, tail, project, zone, cluster_name):
  """Show or stream logs for a job."""
  if follow and tail is not None:
    raise click.ClickException("Use either --follow or --tail, not both.")

  handle = _attach(job_id, project, cluster_name)

  if follow:
    handle.logs(follow=True)
    return

  if tail is not None:
    console.print(handle.tail(n=tail))
    return

  text = handle.logs()
  if text:
    console.print(text)


@jobs.command()
@click.argument("job_id")
@click.option(
  "--timeout",
  type=float,
  default=None,
  help="Maximum seconds to wait for the result.",
)
@click.option(
  "--cleanup/--no-cleanup",
  default=True,
  help="Delete k8s and GCS artifacts after collecting the result.",
)
@cleanup_options
@common_options
def result(
  job_id,
  timeout,
  cleanup,
  cleanup_timeout,
  cleanup_poll_interval,
  project,
  zone,
  cluster_name,
):
  """Wait for and print a job result."""
  handle = _attach(job_id, project, cluster_name)
  console.print(
    handle.result(
      timeout=timeout,
      cleanup=cleanup,
      cleanup_timeout=cleanup_timeout,
      cleanup_poll_interval=cleanup_poll_interval,
    )
  )


@jobs.command()
@click.argument("job_id")
@cleanup_options
@common_options
def cancel(
  job_id,
  cleanup_timeout,
  cleanup_poll_interval,
  project,
  zone,
  cluster_name,
):
  """Cancel a running job by deleting its k8s resource."""
  handle = _attach(job_id, project, cluster_name)
  handle.cancel(
    cleanup_timeout=cleanup_timeout,
    cleanup_poll_interval=cleanup_poll_interval,
  )
  success(f"Cancelled {job_id}")


@jobs.command()
@click.argument("job_id")
@click.option(
  "--k8s/--no-k8s",
  default=True,
  help="Delete Kubernetes resources.",
)
@click.option(
  "--gcs/--no-gcs",
  default=True,
  help="Delete uploaded GCS artifacts.",
)
@cleanup_options
@common_options
def cleanup(
  job_id,
  k8s,
  gcs,
  cleanup_timeout,
  cleanup_poll_interval,
  project,
  zone,
  cluster_name,
):
  """Clean up Kubernetes and/or GCS resources for a job."""
  handle = _attach(job_id, project, cluster_name)
  handle.cleanup(
    k8s=k8s,
    gcs=gcs,
    cleanup_timeout=cleanup_timeout,
    cleanup_poll_interval=cleanup_poll_interval,
  )
  success(f"Cleaned up {job_id}")
