"""Shared Click options for kinetic CLI commands."""

import click

from kinetic.cli.constants import DEFAULT_CLUSTER_NAME, DEFAULT_ZONE


def common_options(f):
  """Shared --project, --zone, --cluster options for subcommands."""
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
    help=f"GKE cluster name [default: {DEFAULT_CLUSTER_NAME}]",
  )(f)
  return f
