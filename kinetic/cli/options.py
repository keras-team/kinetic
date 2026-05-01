"""Shared Click options for kinetic CLI commands."""

import click

from kinetic.cli.constants import DEFAULT_CLUSTER_NAME, DEFAULT_ZONE


def common_options(f):
  """Shared --project, --zone, --cluster options for subcommands."""
  f = click.option(
    "--project",
    envvar="KINETIC_PROJECT",
    default=None,
    help="GCP project ID [env: KINETIC_PROJECT]",
  )(f)
  f = click.option(
    "--zone",
    envvar="KINETIC_ZONE",
    default=None,
    help=f"GCP zone [env: KINETIC_ZONE, default: {DEFAULT_ZONE}]",
  )(f)
  f = click.option(
    "--cluster",
    "cluster_name",
    envvar="KINETIC_CLUSTER",
    default=None,
    help=f"GKE cluster name [default: {DEFAULT_CLUSTER_NAME}]",
  )(f)
  return f


def cleanup_options(f):
  """Shared --cleanup-timeout and --cleanup-poll-interval options."""
  f = click.option(
    "--cleanup-timeout",
    type=float,
    default=180,
    show_default=True,
    help="Maximum seconds to wait for k8s resource deletion.",
  )(f)
  f = click.option(
    "--cleanup-poll-interval",
    type=float,
    default=2,
    show_default=True,
    help="Seconds between k8s deletion-confirmation polls.",
  )(f)
  return f


def force_destroy_option(f):
  """--force-destroy/--no-force-destroy flag.

  Default is None so the caller can distinguish "user did not pass a flag"
  (preserve existing state) from an explicit True/False.
  """
  f = click.option(
    "--force-destroy/--no-force-destroy",
    "force_destroy",
    default=None,
    envvar="KINETIC_FORCE_DESTROY",
    help=(
      "Whether kinetic buckets get auto-emptied on teardown. "
      "Use --no-force-destroy to require manually emptying buckets "
      "before 'kinetic down'. Persisted across commands "
      "[env: KINETIC_FORCE_DESTROY, default: force-destroy]"
    ),
  )(f)
  return f


def jobs_options(f):
  """Shared options for ``kinetic jobs`` subcommands.

  Extends ``common_options`` with ``--namespace`` and `--output-dir`.
  """
  f = common_options(f)
  f = click.option(
    "--namespace",
    envvar="KINETIC_NAMESPACE",
    default="default",
    show_default=True,
    help="Kubernetes namespace [env: KINETIC_NAMESPACE]",
  )(f)
  f = click.option(
    "--output-dir",
    envvar="KINETIC_OUTPUT_DIR",
    default=None,
    help="Output directory [env: KINETIC_OUTPUT_DIR]",
  )(f)
  return f
