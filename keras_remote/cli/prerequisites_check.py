"""Prerequisite checks for the keras-remote CLI.

Delegates common credential checks (gcloud, auth plugin, ADC) to
:mod:`keras_remote.credentials` and converts ``RuntimeError`` into
``click.ClickException``.  CLI-only tool checks (kubectl) remain here.
"""

import shutil

import click

from keras_remote import credentials


def check_gcloud():
  """Verify gcloud CLI is installed."""
  try:
    credentials.ensure_gcloud()
  except RuntimeError as e:
    raise click.ClickException(str(e))  # noqa: B904


def check_kubectl():
  """Verify kubectl is installed."""
  if not shutil.which("kubectl"):
    raise click.ClickException(
      "kubectl not found. Install from: https://kubernetes.io/docs/tasks/tools/"
    )


def check_gke_auth_plugin():
  """Verify gke-gcloud-auth-plugin is installed; auto-install if missing."""
  try:
    credentials.ensure_gke_auth_plugin()
  except RuntimeError as e:
    raise click.ClickException(str(e))  # noqa: B904


def check_gcloud_auth():
  """Check if gcloud Application Default Credentials are configured."""
  try:
    credentials.ensure_adc()
  except RuntimeError as e:
    raise click.ClickException(str(e))  # noqa: B904


def check_all():
  """Run all prerequisite checks."""
  check_gcloud()
  check_kubectl()
  check_gke_auth_plugin()
  check_gcloud_auth()
