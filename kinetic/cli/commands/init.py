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
from kinetic.cli.infra.post_deploy import configure_kubectl
from kinetic.cli.infra.stack_manager import get_current_zone
from kinetic.cli.infra.state import list_clusters, load_state
from kinetic.cli.options import common_options
from kinetic.cli.output import banner, console, success, warning
from kinetic.cli.prerequisites_check import (
  check_gcloud,
  check_gcloud_auth,
  check_gke_auth_plugin,
  check_kubectl,
)
from kinetic.cli.profiles import Profile, set_current, upsert_profile
from kinetic.cli.prompts import resolve_project


@click.command()
@common_options
@click.option(
  "--profile-name",
  "profile_name",
  default=None,
  help="Profile name to save (default: cluster name).",
)
@click.option(
  "--namespace",
  envvar="KINETIC_NAMESPACE",
  default="default",
  show_default=True,
  help="Kubernetes namespace to record in the saved profile "
  "[env: KINETIC_NAMESPACE]",
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompts.")
@click.pass_context
def init(ctx, project, zone, cluster_name, profile_name, namespace, yes):
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
  path = _choose_path(report, yes=yes)
  if path == "join":
    chosen_cluster, chosen_zone = _join_flow(
      project, report["local_clusters"], cluster_name
    )
    saved = profile_name or chosen_cluster
    upsert_profile(
      Profile(
        name=saved,
        project=project,
        zone=chosen_zone,
        cluster=chosen_cluster,
        namespace=namespace,
      )
    )
    # upsert_profile only sets 'current' when the store is empty; force-
    # activate so subsequent commands actually resolve to the new profile.
    set_current(saved)
    _print_join_ready_screen(saved)
  else:
    # `up` handles provisioning AND profile save AND its own ready screen.
    # Forward all the user's resolved env/profile/flag values explicitly —
    # ctx.invoke calls the callback directly and bypasses Click's envvar
    # resolution, so anything not passed here is lost.
    ctx.invoke(
      up,
      project=project,
      zone=zone,
      cluster_name=cluster_name,
      profile_name=profile_name,
      namespace=namespace,
      yes=yes,
    )


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


def _choose_path(report, *, yes):
  """Return 'join' or 'create'. Aborts cleanly if the user declines.

  When no clusters exist, prints an explainer of what the Create path will
  do and confirms (skippable via ``--yes``). When clusters exist, always
  prompts join/create — the choice is too consequential to default through.
  """
  clusters = report["local_clusters"]
  if not clusters:
    _explain_create("No existing Kinetic clusters were found in this project.")
    if not yes:
      click.confirm(
        "\nProceed to create a new cluster?", default=True, abort=True
      )
    return "create"

  _explain_join_or_create(clusters)
  choice = click.prompt(
    "\nWhat would you like to do?",
    type=click.Choice(["join", "create"], case_sensitive=False),
    default="join",
    show_choices=True,
  )
  return choice


def _explain_create(opener):
  """Print a uniform 'here's what creating a cluster does' explainer."""
  console.print()
  console.print(opener)
  console.print()
  console.print("[bold]Creating a new cluster will:[/bold]")
  console.print(
    "  • Provision a GKE cluster, Artifact Registry, and state bucket"
  )
  console.print("  • Save it as a profile and set the profile active")
  console.print("  • Take roughly 5–10 minutes the first time")
  console.print(
    "  • [yellow]Create GCP resources that incur ongoing cost[/yellow]"
  )
  console.print(
    "  • Run [bold]kinetic down[/bold] later to tear everything down"
  )


def _explain_join_or_create(clusters):
  """Print the join/create explainer for the existing-clusters case."""
  cluster_list = ", ".join(f"[bold]{c}[/bold]" for c in clusters)
  console.print()
  console.print(f"Found existing Kinetic cluster(s): {cluster_list}")
  console.print()
  console.print(
    "  [bold green]join[/bold green]    Configure kubectl for an existing "
    "cluster and save a profile."
  )
  console.print(
    "           No GCP resources are created or modified; no added cost."
  )
  console.print()
  console.print(
    "  [bold yellow]create[/bold yellow]  Provision a NEW GKE cluster "
    "alongside the existing one(s)."
  )
  console.print("           Creates additional GCP resources that incur cost.")


def _join_flow(project, clusters, cluster_override):
  """Select a Kinetic cluster and configure kubectl for it.

  When ``cluster_override`` is provided and matches one of ``clusters``,
  the picker is skipped. Returns ``(cluster_name, zone)``.
  """
  if cluster_override and cluster_override in clusters:
    cluster_name = cluster_override
    console.print(
      f"Using cluster [bold]{cluster_name}[/bold] (from --cluster)."
    )
  elif len(clusters) == 1:
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

  zone = _infer_zone(project, cluster_name)
  if zone is None:
    raise click.ClickException(
      f"Could not determine zone for cluster '{cluster_name}'. "
      "The Pulumi stack may be empty or its 'zone' output is missing. "
      "Run 'kinetic profile create' manually with --zone, or re-provision "
      "with 'kinetic up'."
    )

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
  """Read the cluster's zone from its Pulumi stack outputs. None on failure."""
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
  if state.stack is None:
    return None
  return get_current_zone(state.stack)


def _print_join_ready_screen(profile_name):
  console.print()
  success(f"Profile '{profile_name}' created and active.")
  console.print()
  console.print("Next steps:")
  console.print("  [bold]kinetic status[/bold]       check cluster state")
  console.print("  [bold]kinetic jobs list[/bold]    see running jobs")
  console.print("  [bold]kinetic profile ls[/bold]   list saved profiles")
  console.print()
