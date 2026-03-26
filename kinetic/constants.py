"""Zone, region, and location constants for kinetic."""

import os

ZONE_ENV_VAR = "KINETIC_ZONE"
DEFAULT_ZONE = "us-central1-a"
DEFAULT_CLUSTER_NAME = "kinetic-cluster"
DEFAULT_REGION = DEFAULT_ZONE.rsplit("-", 1)[0]  # "us-central1"


def get_default_zone():
  """Return zone from KINETIC_ZONE env var, or DEFAULT_ZONE."""
  return os.environ.get(ZONE_ENV_VAR, DEFAULT_ZONE)


def get_default_cluster_name():
  """Return cluster name from KINETIC_CLUSTER env var, or DEFAULT_CLUSTER_NAME."""
  return os.environ.get("KINETIC_CLUSTER", DEFAULT_CLUSTER_NAME)


def zone_to_region(zone):
  """Convert a GCP zone to its region (e.g. 'us-central1-a' -> 'us-central1')."""
  return zone.rsplit("-", 1)[0] if zone and "-" in zone else DEFAULT_REGION


def zone_to_ar_location(zone):
  """Convert a GCP zone to Artifact Registry multi-region (e.g. 'us-central1-a' -> 'us')."""
  return zone_to_region(zone).split("-")[0]


def get_default_project() -> str | None:
  """Get project ID from KINETIC_PROJECT or GOOGLE_CLOUD_PROJECT."""
  return os.environ.get("KINETIC_PROJECT") or os.environ.get(
    "GOOGLE_CLOUD_PROJECT"
  )


def get_required_project(project: str | None = None) -> str:
  """Resolve the GCP project or raise a clear error."""
  project = project or get_default_project()
  if not project:
    raise ValueError(
      "project must be specified or set KINETIC_PROJECT "
      "(or GOOGLE_CLOUD_PROJECT) environment variable"
    )
  return project


def get_default_namespace(namespace: str | None = None) -> str:
  """Return namespace from arg, KINETIC_NAMESPACE env var, or 'default'."""
  return namespace or os.environ.get("KINETIC_NAMESPACE", "default")


def build_bucket_name(project: str, cluster_name: str) -> str:
  """Return the jobs bucket name for a project and cluster."""
  return f"{project}-kn-{cluster_name}-jobs"
