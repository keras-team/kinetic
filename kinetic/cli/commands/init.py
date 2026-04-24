"""kinetic init command — one-shot onboarding.

Detects the local environment, then routes to either the "Join" path
(use an existing Kinetic cluster) or the "Create" path (delegate to
``kinetic up``). Ends with an active Profile so subsequent commands
need no ``KINETIC_*`` env vars.
"""

import subprocess

import click
from rich.table import Table

from kinetic.cli.commands.up import up
from kinetic.cli.constants import DEFAULT_ZONE
from kinetic.cli.infra.post_deploy import configure_kubectl
from kinetic.cli.infra.state import list_clusters, load_state
from kinetic.cli.output import banner, console, success, warning
from kinetic.cli.prerequisites_check import (
  check_gcloud,
  check_gcloud_auth,
  check_gke_auth_plugin,
  check_kubectl,
)
from kinetic.cli.profiles import Profile, upsert_profile
from kinetic.cli.prompts import resolve_project


@click.command()
@click.option(
  "--project",
  envvar="KINETIC_PROJECT",
  default=None,
  help="GCP project ID [env: KINETIC_PROJECT]",
)
@click.option(
  "--name",
  "profile_name",
  default=None,
  help="Profile name to save (default: cluster name).",
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompts.")
@click.pass_context
def init(ctx, project, profile_name, yes):
  """Onboard: detect environment, pick a cluster, save an active profile."""
  banner("kinetic init")

  # Phase 1: Detect (read-only, non-raising).
  report = _detect(project_hint=project)
  _print_detect_report(report)

  # Phase 2: Gate on prereqs.
  if not report["prereqs_ok"]:
    raise click.ClickException(
      "Fix the issues above, then re-run 'kinetic init'. "
      "Run 'kinetic doctor' for detailed remediation."
    )

  project = report["project"]

  # Phase 3: Pick + execute path.
  if _choose_join_path(report):
    cluster_name, zone = _join_flow(project, report["local_clusters"])
    saved = profile_name or cluster_name
    upsert_profile(
      Profile(
        name=saved,
        project=project,
        zone=zone,
        cluster=cluster_name,
        namespace="default",
      )
    )
    _print_join_ready_screen(saved)
  else:
    # `up` handles provisioning AND profile save AND its own ready screen.
    ctx.invoke(up, project=project, profile_name=profile_name, yes=yes)


def _detect(project_hint):
  """Build a read-only detection report. Never raises."""
  report = {
    "gcloud_ok": _safe(check_gcloud),
    "kubectl_ok": _safe(check_kubectl),
    "auth_plugin_ok": _safe(check_gke_auth_plugin),
    "adc_ok": _safe(check_gcloud_auth),
    "project": None,
    "local_clusters": [],
  }
  report["prereqs_ok"] = all(
    report[k] for k in ("gcloud_ok", "kubectl_ok", "auth_plugin_ok", "adc_ok")
  )

  # Project resolution may still prompt — only attempt if auth looks OK,
  # since resolve_project shells out to gcloud.
  if report["prereqs_ok"]:
    try:
      report["project"] = project_hint or resolve_project()
    except click.ClickException as e:
      report["prereqs_ok"] = False
      report["project_error"] = str(e)

  if report["project"]:
    report["local_clusters"] = list_clusters(report["project"])
  return report


def _safe(fn):
  """Run a prereq check, converting ClickException to a boolean."""
  try:
    fn()
  except click.ClickException:
    return False
  return True


def _print_detect_report(report):
  table = Table(title="Environment")
  table.add_column("Check", style="bold")
  table.add_column("Status")
  table.add_row("gcloud installed", _ok(report["gcloud_ok"]))
  table.add_row("kubectl installed", _ok(report["kubectl_ok"]))
  table.add_row("gke-gcloud-auth-plugin", _ok(report["auth_plugin_ok"]))
  table.add_row("ADC configured", _ok(report["adc_ok"]))
  if report.get("project"):
    table.add_row("GCP project", f"[green]{report['project']}[/green]")
  clusters = report.get("local_clusters") or []
  if clusters:
    table.add_row(
      "Local Kinetic clusters",
      f"[green]{len(clusters)} found[/green]: {', '.join(clusters)}",
    )
  else:
    table.add_row("Local Kinetic clusters", "[dim]none[/dim]")
  console.print()
  console.print(table)
  console.print()


def _ok(flag):
  return "[green]OK[/green]" if flag else "[red]missing[/red]"


def _choose_join_path(report):
  """Return True if the user should Join an existing cluster, False to Create."""
  clusters = report["local_clusters"]
  if not clusters:
    return False
  choice = click.prompt(
    "Found existing cluster(s). Join one, or create a new cluster?",
    type=click.Choice(["join", "create"], case_sensitive=False),
    default="join",
  )
  return choice == "join"


def _join_flow(project, clusters):
  """Interactively select a local cluster and configure kubectl for it.

  Returns (cluster_name, zone).
  """
  if len(clusters) == 1:
    cluster_name = clusters[0]
    console.print(f"Joining cluster [bold]{cluster_name}[/bold].")
  else:
    console.print("Available clusters:")
    for i, name in enumerate(clusters, 1):
      console.print(f"  {i}) {name}")
    cluster_name = click.prompt(
      "\nSelect cluster",
      type=click.Choice(clusters, case_sensitive=False),
      default=clusters[0],
    )

  # Infer zone from the matching Pulumi stack outputs; fall back to default.
  zone = _infer_zone(project, cluster_name) or DEFAULT_ZONE

  try:
    configure_kubectl(cluster_name, zone, project)
  except subprocess.CalledProcessError:
    warning(
      "kubectl configuration failed. Run manually:\n"
      f"  gcloud container clusters get-credentials {cluster_name}"
      f" --zone={zone} --project={project}"
    )

  return cluster_name, zone


def _infer_zone(project, cluster_name):
  """Best-effort zone lookup from the cluster's Pulumi stack. None on failure."""
  try:
    state = load_state(
      project,
      None,
      cluster_name,
      allow_missing=True,
      check_prerequisites=False,
    )
  except click.ClickException:
    return None
  return state.zone


def _print_join_ready_screen(profile_name):
  console.print()
  success(f"Profile '{profile_name}' created and active.")
  console.print()
  console.print("Next steps:")
  console.print("  [bold]kinetic status[/bold]       check cluster state")
  console.print("  [bold]kinetic jobs list[/bold]    see running jobs")
  console.print("  [bold]kinetic profile ls[/bold]   list saved profiles")
  console.print()
