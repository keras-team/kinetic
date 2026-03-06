"""keras-remote ns commands — create, delete, list, add-member, remove-member."""

import click
import pulumi.automation as auto

from keras_remote.cli.config import InfraConfig, NamespaceConfig
from keras_remote.cli.constants import (
  DEFAULT_CLUSTER_NAME,
  DEFAULT_ZONE,
  NAMESPACE_MAX_LENGTH,
  NAMESPACE_PATTERN,
  RESERVED_NAMESPACES,
)
from keras_remote.cli.infra.program import create_program
from keras_remote.cli.infra.stack_manager import (
  detect_resources_to_import,
  get_current_namespaces,
  get_current_node_pools,
  get_stack,
)
from keras_remote.cli.output import banner, console, success, warning
from keras_remote.cli.prerequisites_check import check_all
from keras_remote.cli.prompts import resolve_project


def validate_namespace_name(name):
  """Validate a namespace name against naming constraints.

  Raises:
      click.BadParameter: If the name is invalid.
  """
  if len(name) > NAMESPACE_MAX_LENGTH:
    raise click.BadParameter(
      f"Namespace name must be at most {NAMESPACE_MAX_LENGTH} characters, "
      f"got {len(name)}."
    )
  if not NAMESPACE_PATTERN.match(name):
    raise click.BadParameter(
      "Namespace name must contain only lowercase letters, digits, "
      "and hyphens, and must start and end with a letter or digit."
    )
  if name in RESERVED_NAMESPACES:
    raise click.BadParameter(
      f"'{name}' is a reserved namespace name. "
      f"Reserved: {', '.join(sorted(RESERVED_NAMESPACES))}"
    )


def _common_options(f):
  """Shared options for ns subcommands."""
  f = click.option(
    "--project",
    envvar="KERAS_REMOTE_PROJECT",
    default=None,
    help="GCP project ID [env: KERAS_REMOTE_PROJECT]",
  )(f)
  f = click.option(
    "--zone",
    envvar="KERAS_REMOTE_ZONE",
    default=None,
    help=f"GCP zone [env: KERAS_REMOTE_ZONE, default: {DEFAULT_ZONE}]",
  )(f)
  f = click.option(
    "--cluster",
    "cluster_name",
    envvar="KERAS_REMOTE_CLUSTER",
    default=None,
    help="GKE cluster name [default: keras-remote-cluster]",
  )(f)
  return f


def _resolve_common(project, zone, cluster_name):
  """Resolve common options to concrete values."""
  return (
    project or resolve_project(),
    zone or DEFAULT_ZONE,
    cluster_name or DEFAULT_CLUSTER_NAME,
  )


def _load_state(project, zone, cluster_name):
  """Check prerequisites, refresh stack, return existing pools and namespaces."""
  check_all()
  project, zone, cluster_name = _resolve_common(project, zone, cluster_name)

  base_config = InfraConfig(
    project=project, zone=zone, cluster_name=cluster_name
  )
  try:
    program = create_program(base_config)
    stack = get_stack(program, base_config)
  except auto.errors.CommandError as e:
    raise click.ClickException(
      f"No Pulumi stack found for project '{project}': {e}\n"
      "Run 'keras-remote up' to provision infrastructure first."
    ) from e

  console.print("\nRefreshing state...\n")
  try:
    stack.refresh(on_output=print)
  except auto.errors.CommandError as e:
    warning(f"Failed to refresh stack state: {e}")

  existing_pools = get_current_node_pools(stack)
  existing_namespaces = get_current_namespaces(stack)
  return project, zone, cluster_name, existing_pools, existing_namespaces


def _apply_update(
  project, zone, cluster_name, node_pools, namespaces, skip_iam=False
):
  """Run a Pulumi update with the given config.

  Returns:
    True if the update succeeded, False otherwise.
  """
  config = InfraConfig(
    project=project,
    zone=zone,
    cluster_name=cluster_name,
    node_pools=node_pools,
    namespaces=namespaces,
  )
  program = create_program(config, skip_iam=skip_iam)
  stack = get_stack(program, config)

  # Auto-import shared resources that exist in GCP but not in state.
  to_import = detect_resources_to_import(stack, project, zone)
  if to_import:
    program = create_program(
      config, resources_to_import=to_import, skip_iam=skip_iam
    )
    stack = get_stack(program, config)

  console.print("\n[bold]Updating infrastructure...[/bold]\n")
  try:
    result = stack.up(on_output=print)
    console.print()
    success(f"Pulumi update complete. {result.summary.resource_changes}")
    return True
  except auto.errors.CommandError as e:
    console.print()
    warning(f"Pulumi update encountered an issue: {e}")
    return False


@click.group()
def ns():
  """Manage namespaces for multi-tenant isolation."""


@ns.command("create")
@_common_options
@click.argument("name")
@click.option(
  "--members",
  default=None,
  help="Comma-separated GCP identities (e.g. alice@co.com,bob@co.com)",
)
@click.option("--gpus", type=int, default=None, help="GPU quota limit")
@click.option("--tpus", type=int, default=None, help="TPU quota limit")
@click.option("--cpu", type=int, default=None, help="CPU quota limit")
@click.option("--memory", default=None, help="Memory quota (e.g. 768Gi)")
@click.option("--max-jobs", type=int, default=None, help="Max concurrent Jobs")
@click.option("--max-lws", type=int, default=None, help="Max concurrent LWS")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option(
  "--ignore-iam-errors",
  is_flag=True,
  help="Continue even if IAM operations fail (e.g. 403 permission errors)",
)
def ns_create(
  project,
  zone,
  cluster_name,
  name,
  members,
  gpus,
  tpus,
  cpu,
  memory,
  max_jobs,
  max_lws,
  yes,
  ignore_iam_errors,
):
  """Create a namespace with isolation resources."""
  banner("keras-remote Namespace Create")

  validate_namespace_name(name)

  member_list = [m.strip() for m in members.split(",")] if members else []

  project, zone, cluster_name, existing_pools, existing_namespaces = (
    _load_state(project, zone, cluster_name)
  )

  # Check for duplicate — allow retry with --ignore-iam-errors so the user
  # can reconcile resources that failed on a previous attempt.
  already_exists = any(ns.name == name for ns in existing_namespaces)
  if already_exists and not ignore_iam_errors:
    raise click.ClickException(
      f"Namespace '{name}' already exists. "
      "Use 'keras-remote ns add-member' to add members, "
      "or re-run with --ignore-iam-errors to retry resource creation."
    )

  if already_exists:
    warning(
      f"Namespace '{name}' already in state — "
      "retrying resource creation for missing resources."
    )
    all_namespaces = existing_namespaces
  else:
    new_ns = NamespaceConfig(
      name=name,
      members=member_list,
      gpus=gpus,
      tpus=tpus,
      cpu=cpu,
      memory=memory,
      max_jobs=max_jobs,
      max_lws=max_lws,
    )
    all_namespaces = existing_namespaces + [new_ns]

  console.print(f"\nCreating namespace [bold]{name}[/bold]")
  if member_list:
    console.print(f"  Members: {', '.join(member_list)}")
  if gpus is not None:
    console.print(f"  GPU quota: {gpus}")
  if tpus is not None:
    console.print(f"  TPU quota: {tpus}")
  console.print()

  if not yes:
    click.confirm("Proceed?", abort=True)

  if ignore_iam_errors:
    # Phase 1: Create K8s resources only (skip IAM to avoid 403 aborting
    # the entire update).
    console.print("[bold]Phase 1:[/bold] Creating K8s resources...\n")
    k8s_ok = _apply_update(
      project,
      zone,
      cluster_name,
      existing_pools,
      all_namespaces,
      skip_iam=True,
    )
    if not k8s_ok:
      console.print()
      banner("Namespace Creation Failed")
      return

    # Phase 2: Attempt IAM resources. Failure here is non-fatal.
    console.print("\n[bold]Phase 2:[/bold] Attempting IAM resources...\n")
    iam_ok = _apply_update(
      project,
      zone,
      cluster_name,
      existing_pools,
      all_namespaces,
    )

    console.print()
    if iam_ok:
      banner("Namespace Created")
    else:
      banner("Namespace Created With Warnings")
      warning(
        "Some IAM resources failed to provision (see errors above).\n"
        "  K8s namespace and isolation resources were created.\n"
        "  Re-run without --ignore-iam-errors once permissions are fixed."
      )
  else:
    ok = _apply_update(
      project,
      zone,
      cluster_name,
      existing_pools,
      all_namespaces,
    )
    console.print()
    if not ok:
      banner("Namespace Creation Failed")
      return

    banner("Namespace Created")

  console.print(
    f'\nMembers can now use: [bold]@keras_remote.run(namespace="{name}")[/bold]'
  )
  console.print(f"Or set: [bold]export KERAS_REMOTE_NAMESPACE={name}[/bold]\n")


@ns.command("delete")
@_common_options
@click.argument("name")
@click.option("--force", is_flag=True, help="Force delete even if jobs exist")
@click.option("--cleanup-storage", is_flag=True, help="Also delete GCS objects")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def ns_delete(project, zone, cluster_name, name, force, cleanup_storage, yes):
  """Delete a namespace and its isolation resources."""
  banner("keras-remote Namespace Delete")

  project, zone, cluster_name, existing_pools, existing_namespaces = (
    _load_state(project, zone, cluster_name)
  )

  remaining = [ns for ns in existing_namespaces if ns.name != name]
  if len(remaining) == len(existing_namespaces):
    existing_names = [ns.name for ns in existing_namespaces]
    raise click.ClickException(
      f"Namespace '{name}' not found. "
      f"Existing: {', '.join(existing_names) or '(none)'}"
    )

  console.print(f"\nDeleting namespace [bold]{name}[/bold]")
  console.print(f"Remaining namespaces: {len(remaining)}\n")

  if not yes:
    click.confirm("Proceed?", abort=True)

  ok = _apply_update(project, zone, cluster_name, existing_pools, remaining)

  if ok and cleanup_storage:
    _cleanup_namespace_storage(project, name)

  console.print()
  if ok:
    banner("Namespace Deleted")
  else:
    banner("Namespace Deletion Failed")


def _cleanup_namespace_storage(project, namespace):
  """Delete all GCS objects under the namespace prefix."""
  from google.cloud import storage

  bucket_name = f"{project}-keras-remote-jobs"
  client = storage.Client(project=project)
  bucket = client.bucket(bucket_name)
  blobs = list(bucket.list_blobs(prefix=f"{namespace}/"))
  if blobs:
    console.print(
      f"Deleting {len(blobs)} objects from gs://{bucket_name}/{namespace}/"
    )
    for blob in blobs:
      blob.delete()
    success(f"Deleted {len(blobs)} objects")
  else:
    console.print(f"No objects found under gs://{bucket_name}/{namespace}/")


@ns.command("list")
@_common_options
def ns_list(project, zone, cluster_name):
  """List all managed namespaces."""
  banner("keras-remote Namespaces")

  check_all()
  project, zone, cluster_name = _resolve_common(project, zone, cluster_name)

  base_config = InfraConfig(
    project=project, zone=zone, cluster_name=cluster_name
  )

  try:
    program = create_program(base_config)
    stack = get_stack(program, base_config)
  except auto.errors.CommandError as e:
    warning(f"No Pulumi stack found for project '{project}': {e}")
    console.print("Run 'keras-remote up' to provision infrastructure.")
    return

  console.print("\nRefreshing state...\n")
  try:
    stack.refresh(on_output=print)
  except auto.errors.CommandError as e:
    warning(f"Failed to refresh stack state: {e}")

  namespaces = get_current_namespaces(stack)
  if not namespaces:
    console.print("\nNo managed namespaces (single-tenant mode).")
    console.print(
      "Create one with: [bold]keras-remote ns create <name>[/bold]\n"
    )
    return

  from rich.table import Table

  table = Table(title="Managed Namespaces")
  table.add_column("Namespace", style="bold")
  table.add_column("Members")
  table.add_column("GPUs")
  table.add_column("TPUs")
  table.add_column("Max Jobs")

  for ns_conf in namespaces:
    table.add_row(
      ns_conf.name,
      ", ".join(ns_conf.members) if ns_conf.members else "-",
      str(ns_conf.gpus) if ns_conf.gpus is not None else "-",
      str(ns_conf.tpus) if ns_conf.tpus is not None else "-",
      str(ns_conf.max_jobs) if ns_conf.max_jobs is not None else "-",
    )

  console.print()
  console.print(table)
  console.print()


@ns.command("add-member")
@_common_options
@click.argument("name")
@click.option(
  "--member",
  required=True,
  help="GCP identity (e.g. user:alice@co.com or group:team@co.com)",
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def ns_add_member(project, zone, cluster_name, name, member, yes):
  """Add a member to a namespace."""
  banner("keras-remote Add Member")

  project, zone, cluster_name, existing_pools, existing_namespaces = (
    _load_state(project, zone, cluster_name)
  )

  target = None
  for ns_conf in existing_namespaces:
    if ns_conf.name == name:
      target = ns_conf
      break

  if target is None:
    raise click.ClickException(f"Namespace '{name}' not found.")

  if member in target.members:
    console.print(f"Member '{member}' is already in namespace '{name}'.")
    return

  target.members.append(member)

  console.print(
    f"\nAdding [bold]{member}[/bold] to namespace [bold]{name}[/bold]\n"
  )

  if not yes:
    click.confirm("Proceed?", abort=True)

  ok = _apply_update(
    project, zone, cluster_name, existing_pools, existing_namespaces
  )

  console.print()
  if ok:
    banner("Member Added")
  else:
    banner("Member Add Failed")


@ns.command("remove-member")
@_common_options
@click.argument("name")
@click.option(
  "--member",
  required=True,
  help="GCP identity to remove",
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def ns_remove_member(project, zone, cluster_name, name, member, yes):
  """Remove a member from a namespace."""
  banner("keras-remote Remove Member")

  project, zone, cluster_name, existing_pools, existing_namespaces = (
    _load_state(project, zone, cluster_name)
  )

  target = None
  for ns_conf in existing_namespaces:
    if ns_conf.name == name:
      target = ns_conf
      break

  if target is None:
    raise click.ClickException(f"Namespace '{name}' not found.")

  if member not in target.members:
    raise click.ClickException(
      f"Member '{member}' not found in namespace '{name}'."
    )

  target.members.remove(member)

  console.print(
    f"\nRemoving [bold]{member}[/bold] from namespace [bold]{name}[/bold]\n"
  )

  if not yes:
    click.confirm("Proceed?", abort=True)

  ok = _apply_update(
    project, zone, cluster_name, existing_pools, existing_namespaces
  )

  console.print()
  if ok:
    banner("Member Removed")
  else:
    banner("Member Remove Failed")
