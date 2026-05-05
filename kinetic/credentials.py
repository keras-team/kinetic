"""Credential verification and auto-setup for remote execution.

Ensures all required credentials (kubeconfig, GCP ADC) are available
before submitting jobs. Used by both the programmatic
``kinetic.run()`` API and the CLI.

All functions raise ``RuntimeError`` on unrecoverable failures — callers
in the CLI layer convert these to ``click.ClickException`` as needed.
"""

import os
import shutil
import subprocess
import threading
import time

import google.auth
import google.auth.exceptions
import google.auth.transport.requests
from absl import logging
from kubernetes import config

_credential_cache: dict[tuple[str, str, str], float] = {}
_cache_lock = threading.Lock()
_CREDENTIAL_CACHE_TTL_SECONDS = 300  # 5 minutes


def invalidate_credential_cache(
  project: str | None = None,
  zone: str | None = None,
  cluster: str | None = None,
) -> None:
  """Invalidate cached credential validation results.

  Call this when credentials are known to have changed (e.g. after a
  re-login or kubeconfig update) so that the next
  ``ensure_credentials`` call performs a fresh check.

  If all three arguments are provided, only that specific entry is
  removed.  Otherwise the entire cache is cleared.
  """
  with _cache_lock:
    if project is not None and zone is not None and cluster is not None:
      _credential_cache.pop((project, zone, cluster), None)
    else:
      _credential_cache.clear()


def ensure_credentials(project: str, zone: str, cluster: str) -> None:
  """Ensure all credentials needed for remote execution are available.

  Results are cached per (project, zone, cluster) tuple for 5 minutes
  to avoid repeated subprocess calls and kubeconfig parsing during
  tight polling loops (e.g. ``JobHandle.result()``).  Call
  ``invalidate_credential_cache()`` to force a fresh check before the
  TTL expires (e.g. after a re-login or kubeconfig change).

  Checks and auto-configures credentials in order:
  1. gcloud CLI (must be installed)
  2. gke-gcloud-auth-plugin (auto-install if missing)
  3. GCP Application Default Credentials (auto-login if missing)
  4. Kubeconfig for the target cluster (auto-configure if wrong/missing)

  Args:
      project: GCP project ID.
      zone: GCP zone (e.g. ``us-central1-a``).
      cluster: GKE cluster name.

  Raises:
      RuntimeError: If a required credential cannot be configured.
  """
  cache_key = (project, zone, cluster)
  with _cache_lock:
    last_validated = _credential_cache.get(cache_key)
    if (
      last_validated is not None
      and time.monotonic() - last_validated < _CREDENTIAL_CACHE_TTL_SECONDS
    ):
      return

    ensure_gcloud()
    ensure_gke_auth_plugin()
    ensure_adc()
    ensure_kubeconfig(project, zone, cluster)

    _credential_cache[cache_key] = time.monotonic()


def ensure_gcloud() -> None:
  """Verify gcloud CLI is installed."""
  if not shutil.which("gcloud"):
    raise RuntimeError(
      "gcloud CLI not found. "
      "Install from: https://cloud.google.com/sdk/docs/install"
    )


def ensure_gke_auth_plugin() -> None:
  """Verify gke-gcloud-auth-plugin is installed; auto-install if missing."""
  if shutil.which("gke-gcloud-auth-plugin"):
    return

  logging.info("gke-gcloud-auth-plugin not found. Installing...")
  try:
    subprocess.run(
      [
        "gcloud",
        "components",
        "install",
        "gke-gcloud-auth-plugin",
        "--quiet",
      ],
      check=True,
      capture_output=True,
    )
    logging.info("gke-gcloud-auth-plugin installed successfully.")
  except subprocess.CalledProcessError as e:
    raise RuntimeError(
      "Failed to install gke-gcloud-auth-plugin. "
      "Install manually: gcloud components install gke-gcloud-auth-plugin"
    ) from e


def ensure_adc() -> None:
  """Verify GCP Application Default Credentials are configured.

  Uses ``google-auth`` to resolve ADC in-process (no subprocess).
  Supports all ADC sources: ``GOOGLE_APPLICATION_CREDENTIALS`` env var,
  gcloud ADC file, Compute Engine metadata, Workload Identity Federation.

  Falls back to interactive ``gcloud auth application-default login``
  if no credentials are found and gcloud is available.
  """
  try:
    credentials, _ = google.auth.default()
  except google.auth.exceptions.DefaultCredentialsError:
    _adc_interactive_login()
    return

  try:
    credentials.refresh(google.auth.transport.requests.Request())
  except google.auth.exceptions.RefreshError:
    _adc_interactive_login()


def _adc_interactive_login() -> None:
  """Attempt ``gcloud auth application-default login``; raise if unavailable."""
  if not shutil.which("gcloud"):
    raise RuntimeError(
      "No Application Default Credentials found and gcloud CLI is "
      "not available for interactive login.\n"
      "Set the GOOGLE_APPLICATION_CREDENTIALS environment variable "
      "to a service account key file, or install gcloud and run:\n"
      "  gcloud auth application-default login"
    )
  logging.info("Application Default Credentials not found. Running login...")
  try:
    subprocess.run(
      ["gcloud", "auth", "application-default", "login"],
      check=True,
    )
  except subprocess.CalledProcessError as e:
    raise RuntimeError(
      "Failed to configure Application Default Credentials. "
      "Run manually: gcloud auth application-default login"
    ) from e


def ensure_kubeconfig(project: str, zone: str, cluster: str) -> None:
  """Ensure kubeconfig is configured for the target GKE cluster.

  Loads the existing kubeconfig and verifies the active context points to the
  expected cluster (``gke_{project}_{zone}_{cluster}``).  If the context is
  wrong or kubeconfig is missing, runs ``gcloud container clusters
  get-credentials`` to configure it.
  """
  expected = f"gke_{project}_{zone}_{cluster}"

  # Try loading existing kubeconfig and validate the active context.
  try:
    config.load_kube_config()
    contexts, active_context = config.list_kube_config_contexts()

    if active_context:
      active_cluster = active_context.get("context", {}).get("cluster", "")

      if active_cluster != expected:
        logging.info(
          "Active kubeconfig context '%s' does not match expected "
          "cluster '%s'. Reconfiguring...",
          active_cluster,
          expected,
        )
      else:
        return
    # No active context — fall through to reconfigure.
  except config.ConfigException:
    logging.info(
      "No valid kubeconfig found. Configuring for cluster '%s' "
      "in project '%s', zone '%s'...",
      cluster,
      project,
      zone,
    )

  _configure_kubeconfig(cluster, zone, project)


def _configure_kubeconfig(cluster_name: str, zone: str, project: str) -> None:
  """Run ``gcloud container clusters get-credentials``."""
  env = {**os.environ, "USE_GKE_GCLOUD_AUTH_PLUGIN": "True"}
  try:
    subprocess.run(
      [
        "gcloud",
        "container",
        "clusters",
        "get-credentials",
        "--",
        cluster_name,
        f"--zone={zone}",
        f"--project={project}",
      ],
      check=True,
      env=env,
      capture_output=True,
    )
    logging.info("Kubeconfig configured for cluster '%s'.", cluster_name)
  except subprocess.CalledProcessError as e:
    raise RuntimeError(
      f"Failed to configure kubeconfig for cluster '{cluster_name}' "
      f"in zone '{zone}', project '{project}'. "
      f"Ensure the cluster exists and you have access. "
      f"Run manually: gcloud container clusters get-credentials "
      f"{cluster_name} --zone={zone} --project={project}"
    ) from e
