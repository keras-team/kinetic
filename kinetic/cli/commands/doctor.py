"""kinetic doctor command — diagnose environment and infrastructure health."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum

import google.auth
import google.auth.exceptions
import google.oauth2.credentials
import google.oauth2.service_account

import click
from google.api_core import exceptions as google_exceptions
from google.cloud import (
  artifactregistry_v1,
  billing_v1,
  compute_v1,
  container_v1,
  iam_admin_v1,
  resourcemanager_v3,
  service_usage_v1,
  storage,
)
from rich.table import Table

from kinetic.cli.constants import (
  DEFAULT_CLUSTER_NAME,
  DEFAULT_ZONE,
  KINETIC_KSA_NAME,
  LWS_INSTALL_URL,
  REQUIRED_APIS,
  STATE_DIR,
)
from kinetic.cli.options import common_options
from kinetic.cli.output import LiveOutputPanel, banner, console
from kinetic.constants import (
  get_default_cluster_name,
  get_default_project,
  get_default_zone,
  zone_to_ar_location,
  zone_to_region,
)

_SUBPROCESS_TIMEOUT = 15


class CheckStatus(Enum):
  """Status of a single diagnostic check."""

  PASS = "pass"
  FAIL = "fail"
  WARN = "warn"
  SKIP = "skip"


@dataclass
class CheckResult:
  """Result of a single diagnostic check."""

  name: str
  status: CheckStatus
  message: str
  fix_hint: str = ""


_STATUS_ICON = {
  CheckStatus.PASS: "[green]\u2714 PASS[/green]",
  CheckStatus.FAIL: "[red]\u2718 FAIL[/red]",
  CheckStatus.WARN: "[yellow]\u25b2 WARN[/yellow]",
  CheckStatus.SKIP: "[dim]\u2500 SKIP[/dim]",
}

# Compact status icons for the live progress panel (no Rich markup).
_PROGRESS_ICON = {
  CheckStatus.PASS: "\u2714",  # ✔
  CheckStatus.FAIL: "\u2718",  # ✘
  CheckStatus.WARN: "\u25b2",  # ▲
  CheckStatus.SKIP: "\u2500",  # ─
}

# Section labels inserted before each group in the results table.
_SECTIONS = (
  "Local Tools",
  "Authentication",
  "Configuration",
  "GCP Project",
  "GCP APIs",
  "GCP Resources",
  "Infrastructure",
  "Kubernetes",
)


# ---------------------------------------------------------------------------
# Group 1: Local tools
# ---------------------------------------------------------------------------


def _check_tool(name, binary, fix_hint):
  """Check if a binary is on PATH."""
  path = shutil.which(binary)
  if path:
    return CheckResult(name, CheckStatus.PASS, path)
  return CheckResult(name, CheckStatus.FAIL, "Not found", fix_hint)


def _check_local_tools():
  """Check gcloud, kubectl, and gke-gcloud-auth-plugin."""
  return [
    _check_tool(
      "gcloud CLI",
      "gcloud",
      "Install from: https://cloud.google.com/sdk/docs/install\n"
      "On macOS: brew install --cask google-cloud-sdk",
    ),
    _check_tool(
      "kubectl",
      "kubectl",
      "Install from: https://kubernetes.io/docs/tasks/tools/\n"
      "Or: gcloud components install kubectl",
    ),
    _check_tool(
      "gke-gcloud-auth-plugin",
      "gke-gcloud-auth-plugin",
      "Run: gcloud components install gke-gcloud-auth-plugin\n"
      "Or install via the package manager for your OS",
    ),
  ]


# ---------------------------------------------------------------------------
# Group 2: Authentication
# ---------------------------------------------------------------------------


def _check_adc():
  """Check Application Default Credentials (read-only, no gcloud needed)."""
  try:
    creds, project = google.auth.default()
  except google.auth.exceptions.DefaultCredentialsError:
    return CheckResult(
      "Application Default Credentials",
      CheckStatus.FAIL,
      "Not configured",
      "Run: gcloud auth application-default login\n"
      "If expired: gcloud auth application-default revoke && "
      "gcloud auth application-default login\n"
      "Service account: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json",
    )

  if isinstance(creds, google.oauth2.service_account.Credentials):
    cred_label = "Service account"
  elif isinstance(creds, google.oauth2.credentials.Credentials):
    cred_label = "User credentials (gcloud ADC)"
  else:
    cred_label = type(creds).__name__

  detail = cred_label
  if project:
    detail += f", project: {project}"

  return CheckResult(
    "Application Default Credentials",
    CheckStatus.PASS,
    detail,
  )


def _check_gcloud_account():
  """Check active gcloud account (read-only)."""
  try:
    result = subprocess.run(
      ["gcloud", "config", "get-value", "account"],
      capture_output=True,
      text=True,
      timeout=_SUBPROCESS_TIMEOUT,
    )
    account = result.stdout.strip() if result.returncode == 0 else ""
    if account and account != "(unset)":
      return CheckResult("gcloud account", CheckStatus.PASS, account)
    return CheckResult(
      "gcloud account",
      CheckStatus.FAIL,
      "No active account",
      "Run: gcloud auth login\n"
      "To switch accounts: gcloud config set account USER@EXAMPLE.COM",
    )
  except subprocess.TimeoutExpired:
    return CheckResult(
      "gcloud account", CheckStatus.WARN, "Timed out checking account"
    )


def _check_auth(has_gcloud):
  """Run authentication checks.

  ADC check always runs (uses google-auth, no gcloud needed).
  gcloud account check is skipped if gcloud is missing.
  """
  results = [_check_adc()]

  if not has_gcloud:
    results.append(
      CheckResult(
        "gcloud account",
        CheckStatus.SKIP,
        "Skipped (requires: gcloud CLI)",
      )
    )
  else:
    results.append(_check_gcloud_account())

  return results


# ---------------------------------------------------------------------------
# Group 3: Configuration
# ---------------------------------------------------------------------------


def _check_config(project, zone, cluster_name):
  """Check environment variable configuration."""
  results = []

  if project:
    env_project = os.environ.get("KINETIC_PROJECT")
    gcp_project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if env_project and project == env_project:
      source = "KINETIC_PROJECT"
    elif gcp_project and project == gcp_project:
      source = "GOOGLE_CLOUD_PROJECT"
    else:
      source = "flag"
    results.append(
      CheckResult("Project ID", CheckStatus.PASS, f"{project} ({source})")
    )
  else:
    results.append(
      CheckResult(
        "Project ID",
        CheckStatus.WARN,
        "Not set",
        "Set: export KINETIC_PROJECT=your-project-id\nOr pass --project flag",
      )
    )

  env_zone = os.environ.get("KINETIC_ZONE")
  if env_zone and zone == env_zone:
    zone_source = "KINETIC_ZONE"
  elif zone == DEFAULT_ZONE:
    zone_source = "default"
  else:
    zone_source = "flag"
  results.append(
    CheckResult("Zone", CheckStatus.PASS, f"{zone} ({zone_source})")
  )

  env_cluster = os.environ.get("KINETIC_CLUSTER")
  if env_cluster and cluster_name == env_cluster:
    cluster_source = "KINETIC_CLUSTER"
  elif cluster_name == DEFAULT_CLUSTER_NAME:
    cluster_source = "default"
  else:
    cluster_source = "flag"
  results.append(
    CheckResult(
      "Cluster name",
      CheckStatus.PASS,
      f"{cluster_name} ({cluster_source})",
    )
  )

  return results


# ---------------------------------------------------------------------------
# Group 4: GCP project access
# ---------------------------------------------------------------------------


def _check_project_access(project):
  """Check if the GCP project exists and is accessible."""
  try:
    client = resourcemanager_v3.ProjectsClient()
    client.get_project(name=f"projects/{project}")
    return CheckResult("GCP project access", CheckStatus.PASS, project)
  except google_exceptions.NotFound:
    return CheckResult(
      "GCP project access",
      CheckStatus.FAIL,
      f"Project '{project}' not found",
      f"If project doesn't exist: run 'kinetic up' "
      f"(it can create the project interactively)\n"
      f"Verify at: https://console.cloud.google.com/"
      f"home/dashboard?project={project}",
    )
  except google_exceptions.PermissionDenied as exc:
    return CheckResult(
      "GCP project access",
      CheckStatus.FAIL,
      f"Permission denied for project '{project}'",
      f"Ask a project owner to grant you roles/editor or roles/owner\n"
      f"Detail: {exc}",
    )
  except Exception as exc:
    return CheckResult(
      "GCP project access",
      CheckStatus.WARN,
      f"Could not check: {exc}",
    )


def _check_billing(project):
  """Check if billing is enabled on the project."""
  try:
    client = billing_v1.CloudBillingClient()
    info = client.get_project_billing_info(name=f"projects/{project}")
    if info.billing_enabled:
      return CheckResult("Billing enabled", CheckStatus.PASS, "Enabled")
    return CheckResult(
      "Billing enabled",
      CheckStatus.FAIL,
      "Billing not enabled",
      f"Link a billing account at: https://console.cloud.google.com/"
      f"billing/linkedaccount?project={project}\n"
      f"Or: 'kinetic up' will prompt to link billing during setup",
    )
  except google_exceptions.PermissionDenied:
    return CheckResult(
      "Billing enabled",
      CheckStatus.WARN,
      "Could not check (insufficient permissions or Billing API not enabled)",
      f"Enable the API: gcloud services enable "
      f"cloudbilling.googleapis.com --project {project}\n"
      f"Then link a billing account at: https://console.cloud.google.com/"
      f"billing/linkedaccount?project={project}\n"
      f"Or: 'kinetic up' will prompt to link billing during setup",
    )
  except Exception as exc:
    return CheckResult(
      "Billing enabled",
      CheckStatus.WARN,
      f"Could not check: {exc}",
    )


def _check_gcp_project(has_gcloud, has_adc, project):
  """Run GCP project checks. Skips if prerequisites are missing."""
  skip_reason = None
  if not has_gcloud:
    skip_reason = "Skipped (requires: gcloud CLI)"
  elif not has_adc:
    skip_reason = "Skipped (requires: Application Default Credentials)"
  elif not project:
    skip_reason = "Skipped (requires: Project ID)"

  if skip_reason:
    return [
      CheckResult("GCP project access", CheckStatus.SKIP, skip_reason),
      CheckResult("Billing enabled", CheckStatus.SKIP, skip_reason),
    ]

  project_result = _check_project_access(project)
  if project_result.status != CheckStatus.PASS:
    return [
      project_result,
      CheckResult(
        "Billing enabled",
        CheckStatus.SKIP,
        "Skipped (requires: GCP project access)",
      ),
    ]
  return [project_result, _check_billing(project)]


# ---------------------------------------------------------------------------
# Group 5: GCP APIs
# ---------------------------------------------------------------------------


def _check_apis(has_project_access, project):
  """Check required GCP APIs are enabled."""
  if not has_project_access:
    return [
      CheckResult(
        f"API: {api}",
        CheckStatus.SKIP,
        "Skipped (requires: GCP project access)",
      )
      for api in REQUIRED_APIS
    ]

  # Fetch all enabled APIs in one call.
  try:
    client = service_usage_v1.ServiceUsageClient()
    enabled = set()
    request = service_usage_v1.ListServicesRequest(
      parent=f"projects/{project}",
      filter="state:ENABLED",
    )
    for service in client.list_services(request=request):
      # service.config.name is the API name (e.g. "compute.googleapis.com").
      enabled.add(service.config.name)
  except Exception as exc:
    return [
      CheckResult(
        f"API: {api}",
        CheckStatus.WARN,
        f"Could not list enabled APIs: {exc}",
      )
      for api in REQUIRED_APIS
    ]

  results = []
  for api in REQUIRED_APIS:
    if api in enabled:
      results.append(CheckResult(f"API: {api}", CheckStatus.PASS, "Enabled"))
    else:
      results.append(
        CheckResult(
          f"API: {api}",
          CheckStatus.FAIL,
          "Not enabled",
          f"Run: gcloud services enable {api} --project {project}\n"
          f"Or: 'kinetic up' enables all required APIs automatically",
        )
      )
  return results


# ---------------------------------------------------------------------------
# Group 6: GCP Resources (service accounts, AR, storage, networking)
# ---------------------------------------------------------------------------


def _check_service_account(project, sa_id, display_label):
  """Check if a GCP service account exists."""
  email = f"{sa_id}@{project}.iam.gserviceaccount.com"
  name = f"projects/{project}/serviceAccounts/{email}"
  try:
    client = iam_admin_v1.IAMClient()
    client.get_service_account(name=name)
    return CheckResult(display_label, CheckStatus.PASS, email)
  except google_exceptions.NotFound:
    return CheckResult(
      display_label,
      CheckStatus.FAIL,
      f"Not found: {email}",
      "Run: kinetic up (creates all service accounts)",
    )
  except Exception as exc:
    return CheckResult(
      display_label,
      CheckStatus.WARN,
      f"Could not check: {exc}",
    )


def _check_ar_repo(project, cluster_name, ar_location):
  """Check Artifact Registry repository exists."""
  repo_id = f"kn-{cluster_name}"
  repo_name = (
    f"projects/{project}/locations/{ar_location}/repositories/{repo_id}"
  )
  try:
    client = artifactregistry_v1.ArtifactRegistryClient()
    client.get_repository(
      request=artifactregistry_v1.GetRepositoryRequest(name=repo_name),
    )
    return CheckResult(
      "Artifact Registry",
      CheckStatus.PASS,
      f"{ar_location}-docker.pkg.dev/{project}/{repo_id}",
    )
  except google_exceptions.NotFound:
    return CheckResult(
      "Artifact Registry",
      CheckStatus.FAIL,
      f"Repository '{repo_id}' not found in {ar_location}",
      "Run: kinetic up (creates the container registry)",
    )
  except Exception as exc:
    return CheckResult(
      "Artifact Registry",
      CheckStatus.WARN,
      f"Could not check: {exc}",
    )


def _check_storage_bucket(project, bucket_name, label):
  """Check a GCS bucket exists."""
  try:
    client = storage.Client(project=project)
    client.get_bucket(bucket_name)
    return CheckResult(label, CheckStatus.PASS, f"gs://{bucket_name}")
  except google_exceptions.NotFound:
    return CheckResult(
      label,
      CheckStatus.FAIL,
      f"Bucket not found: {bucket_name}",
      "Run: kinetic up (creates required storage buckets)",
    )
  except Exception as exc:
    return CheckResult(
      label,
      CheckStatus.WARN,
      f"Could not check: {exc}",
    )


def _check_vpc_network(project, cluster_name):
  """Check VPC network exists."""
  network_name = f"kn-{cluster_name}"
  try:
    client = compute_v1.NetworksClient()
    client.get(project=project, network=network_name)
    return CheckResult("VPC network", CheckStatus.PASS, network_name)
  except google_exceptions.NotFound:
    return CheckResult(
      "VPC network",
      CheckStatus.FAIL,
      f"Network '{network_name}' not found",
      "Run: kinetic up (creates VPC and Cloud NAT)",
    )
  except Exception as exc:
    return CheckResult(
      "VPC network",
      CheckStatus.WARN,
      f"Could not check: {exc}",
    )


def _check_cloud_nat(project, cluster_name, zone):
  """Check Cloud NAT exists (required for private node outbound traffic)."""
  region = zone_to_region(zone)
  router_name = f"kn-{cluster_name}-router"
  nat_name = f"kn-{cluster_name}-nat"
  try:
    client = compute_v1.RoutersClient()
    router = client.get(project=project, region=region, router=router_name)
    for nat in router.nats:
      if nat.name == nat_name:
        return CheckResult(
          "Cloud NAT",
          CheckStatus.PASS,
          f"{nat_name} (router: {router_name})",
        )
    return CheckResult(
      "Cloud NAT",
      CheckStatus.FAIL,
      f"NAT '{nat_name}' not found on router '{router_name}'",
      "Run: kinetic up (creates Cloud Router and NAT)\n"
      "Without NAT, private nodes cannot pull images or reach the internet",
    )
  except google_exceptions.NotFound:
    return CheckResult(
      "Cloud NAT",
      CheckStatus.FAIL,
      f"Router '{router_name}' not found in {region}",
      "Run: kinetic up (creates Cloud Router and NAT)\n"
      "Without NAT, private nodes cannot pull images or reach the internet",
    )
  except Exception as exc:
    return CheckResult(
      "Cloud NAT",
      CheckStatus.WARN,
      f"Could not check: {exc}",
    )


def _check_gcp_resources(has_project_access, project, zone, cluster_name):
  """Check GCP resources created by kinetic up."""
  if not has_project_access:
    skip = "Skipped (requires: GCP project access)"
    return [
      CheckResult(name, CheckStatus.SKIP, skip)
      for name in [
        "Node service account",
        "Build service account",
        "Artifact Registry",
        "Jobs bucket",
        "Builds bucket",
        "VPC network",
        "Cloud NAT",
      ]
    ]

  ar_location = zone_to_ar_location(zone)
  return [
    _check_service_account(
      project, f"kn-{cluster_name}-nodes", "Node service account"
    ),
    _check_service_account(
      project, f"kn-{cluster_name}-builds", "Build service account"
    ),
    _check_ar_repo(project, cluster_name, ar_location),
    _check_storage_bucket(
      project, f"{project}-kn-{cluster_name}-jobs", "Jobs bucket"
    ),
    _check_storage_bucket(
      project, f"{project}-kn-{cluster_name}-builds", "Builds bucket"
    ),
    _check_vpc_network(project, cluster_name),
    _check_cloud_nat(project, cluster_name, zone),
  ]


# ---------------------------------------------------------------------------
# Group 7: Infrastructure
# ---------------------------------------------------------------------------


def _check_pulumi_state(project, cluster_name):
  """Check Pulumi state directory for existing stacks."""
  stacks_dir = os.path.join(STATE_DIR, ".pulumi", "stacks", "kinetic")
  if not os.path.isdir(stacks_dir):
    return CheckResult(
      "Pulumi state",
      CheckStatus.WARN,
      f"State directory not found: {STATE_DIR}",
      f"Run: kinetic up --project {project}",
    )

  expected = f"{project}-{cluster_name}.json"
  try:
    files = [file for file in os.listdir(stacks_dir) if file.endswith(".json")]
  except OSError:
    files = []

  if expected in files:
    stack_name = f"{project}-{cluster_name}"
    return CheckResult(
      "Pulumi state", CheckStatus.PASS, f"Stack found: {stack_name}"
    )

  available = [file.removesuffix(".json") for file in files]
  if available:
    stacks_str = ", ".join(available)
    return CheckResult(
      "Pulumi state",
      CheckStatus.WARN,
      f"Stack '{project}-{cluster_name}' not found. Available: {stacks_str}",
      f"Check --cluster or KINETIC_CLUSTER value.\n"
      f"Or run: kinetic up --project {project} --cluster {cluster_name}",
    )
  return CheckResult(
    "Pulumi state",
    CheckStatus.WARN,
    "No stacks found",
    f"Run: kinetic up --project {project}",
  )


def _check_gke_cluster(project, zone, cluster_name):
  """Check GKE cluster exists and is RUNNING."""
  try:
    client = container_v1.ClusterManagerClient()
    name = f"projects/{project}/locations/{zone}/clusters/{cluster_name}"
    cluster = client.get_cluster(name=name)
    # Status enum: PROVISIONING=1, RUNNING=2, RECONCILING=3, etc.
    status = container_v1.Cluster.Status(cluster.status).name
    if status == "RUNNING":
      return CheckResult("GKE cluster", CheckStatus.PASS, "RUNNING")
    if status in ("PROVISIONING", "RECONCILING"):
      return CheckResult(
        "GKE cluster",
        CheckStatus.WARN,
        status,
        "Cluster is being updated. Wait a few minutes and retry.",
      )
    return CheckResult(
      "GKE cluster",
      CheckStatus.FAIL,
      status or "Unknown status",
      f"Check console: https://console.cloud.google.com/"
      f"kubernetes/list?project={project}",
    )
  except google_exceptions.NotFound:
    return CheckResult(
      "GKE cluster",
      CheckStatus.FAIL,
      "Cluster not found",
      f"Run: kinetic up --project {project} --zone {zone} "
      f"--cluster {cluster_name}",
    )
  except Exception as exc:
    return CheckResult(
      "GKE cluster",
      CheckStatus.WARN,
      f"Could not check: {exc}",
    )


def _check_infra(has_project_access, project, zone, cluster_name):
  """Run infrastructure checks."""
  results = []

  # Pulumi state can always be checked (filesystem only).
  if project:
    results.append(_check_pulumi_state(project, cluster_name))
  else:
    results.append(
      CheckResult(
        "Pulumi state",
        CheckStatus.SKIP,
        "Skipped (requires: Project ID)",
      )
    )

  if not has_project_access:
    results.append(
      CheckResult(
        "GKE cluster",
        CheckStatus.SKIP,
        "Skipped (requires: GCP project access)",
      )
    )
    return results

  results.append(_check_gke_cluster(project, zone, cluster_name))
  return results


# ---------------------------------------------------------------------------
# Group 8: Kubernetes
# ---------------------------------------------------------------------------


def _check_kubeconfig(project, zone, cluster_name):
  """Check kubeconfig context (read-only, no reconfiguration)."""
  try:
    from kubernetes import config as k8s_config
  except ImportError:
    return CheckResult(
      "kubeconfig context",
      CheckStatus.SKIP,
      "kubernetes Python package not installed",
    )

  expected = f"gke_{project}_{zone}_{cluster_name}"
  creds_cmd = (
    f"gcloud container clusters get-credentials {cluster_name} "
    f"--zone={zone} --project={project}"
  )

  try:
    k8s_config.load_kube_config()
    _, active = k8s_config.list_kube_config_contexts()
    if not active:
      return CheckResult(
        "kubeconfig context",
        CheckStatus.FAIL,
        "No active kubeconfig context",
        f"Run: {creds_cmd}",
      )
    active_cluster = active.get("context", {}).get("cluster", "")
    if active_cluster == expected:
      return CheckResult("kubeconfig context", CheckStatus.PASS, expected)
    return CheckResult(
      "kubeconfig context",
      CheckStatus.WARN,
      f"Active: '{active_cluster}' (expected: '{expected}')",
      f"Run: {creds_cmd}\n"
      f"Or switch context: kubectl config use-context {expected}",
    )
  except k8s_config.ConfigException:
    return CheckResult(
      "kubeconfig context",
      CheckStatus.FAIL,
      "No valid kubeconfig found",
      f"Run: {creds_cmd}",
    )


def _check_k8s_connectivity():
  """Check Kubernetes cluster reachability."""
  try:
    result = subprocess.run(
      ["kubectl", "cluster-info"],
      capture_output=True,
      timeout=_SUBPROCESS_TIMEOUT,
    )
    if result.returncode == 0:
      return CheckResult(
        "Cluster connectivity", CheckStatus.PASS, "Cluster reachable"
      )
    return CheckResult(
      "Cluster connectivity",
      CheckStatus.FAIL,
      "Cannot reach cluster",
      "If timeout: check network/VPN connection\n"
      "If unauthorized: re-run gcloud container clusters get-credentials\n"
      "If cluster is private: ensure you're on an authorized network",
    )
  except subprocess.TimeoutExpired:
    return CheckResult(
      "Cluster connectivity",
      CheckStatus.FAIL,
      "Timed out connecting to cluster",
      "Check network/VPN connection\n"
      "If cluster is private: ensure you're on an authorized network",
    )


def _check_node_pools(project, zone, cluster_name):
  """Check GKE node pool health.

  Returns:
      Tuple of (CheckResult, has_gpu_pools) so callers can decide
      whether GPU-specific checks (e.g. NVIDIA drivers) apply.
  """
  healthy_statuses = {"RUNNING", "RECONCILING", "PROVISIONING"}
  try:
    client = container_v1.ClusterManagerClient()
    parent = f"projects/{project}/locations/{zone}/clusters/{cluster_name}"
    response = client.list_node_pools(parent=parent)
    pools = list(response.node_pools)
  except Exception as exc:
    return (
      CheckResult(
        "Node pools",
        CheckStatus.WARN,
        f"Could not list node pools: {exc}",
      ),
      False,
    )

  if not pools:
    return (
      CheckResult(
        "Node pools",
        CheckStatus.WARN,
        "No node pools found",
        "Run: kinetic pool add --accelerator <spec>",
      ),
      False,
    )

  # Count accelerator pools (exclude default pool).
  accel_pools = [pool for pool in pools if pool.config.accelerators]
  # Detect GPU pools by checking for nvidia accelerator types.
  has_gpu_pools = any(
    "nvidia" in acc.accelerator_type.lower()
    for pool in accel_pools
    for acc in pool.config.accelerators
  )
  unhealthy = []
  for pool in pools:
    status_name = container_v1.NodePool.Status(pool.status).name
    if status_name not in healthy_statuses:
      unhealthy.append(f"{pool.name} ({status_name})")

  msg_parts = [f"{len(pools)} pool(s)"]
  if accel_pools:
    msg_parts.append(f"{len(accel_pools)} with accelerators")
  else:
    msg_parts.append("no accelerator pools")

  if unhealthy:
    return (
      CheckResult(
        "Node pools",
        CheckStatus.WARN,
        f"{', '.join(msg_parts)}. Unhealthy: {', '.join(unhealthy)}",
        "Delete and re-add unhealthy pools: kinetic pool remove/add",
      ),
      has_gpu_pools,
    )

  return (
    CheckResult("Node pools", CheckStatus.PASS, ", ".join(msg_parts)),
    has_gpu_pools,
  )


def _check_lws_crd():
  """Check if the LeaderWorkerSet CRD is installed."""
  try:
    result = subprocess.run(
      ["kubectl", "get", "crd", "leaderworkersets.leaderworkerset.x-k8s.io"],
      capture_output=True,
      timeout=_SUBPROCESS_TIMEOUT,
    )
    if result.returncode == 0:
      return CheckResult("LWS CRD", CheckStatus.PASS, "Installed")
    return CheckResult(
      "LWS CRD",
      CheckStatus.WARN,
      "Not installed (needed for multi-host TPU workloads)",
      "Run: kinetic up (installs LWS in post-deploy)\n"
      f"Or manually: kubectl apply --server-side -f {LWS_INSTALL_URL}",
    )
  except subprocess.TimeoutExpired:
    return CheckResult("LWS CRD", CheckStatus.WARN, "Timed out")


def _check_kinetic_ksa():
  """Check kinetic Kubernetes service account with WIF annotation."""
  try:
    result = subprocess.run(
      [
        "kubectl",
        "get",
        "serviceaccount",
        KINETIC_KSA_NAME,
        "-n",
        "default",
        "-o",
        "json",
      ],
      capture_output=True,
      text=True,
      timeout=_SUBPROCESS_TIMEOUT,
    )
    if result.returncode != 0:
      return CheckResult(
        "Kinetic KSA",
        CheckStatus.FAIL,
        f"ServiceAccount '{KINETIC_KSA_NAME}' not found in default namespace",
        "Run: kinetic up (creates the KSA with Workload Identity binding)",
      )
    try:
      sa = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
      return CheckResult(
        "Kinetic KSA", CheckStatus.WARN, "Could not parse KSA data"
      )

    annotations = sa.get("metadata", {}).get("annotations", {})
    wif_annotation = annotations.get("iam.gke.io/gcp-service-account", "")
    if wif_annotation:
      return CheckResult(
        "Kinetic KSA",
        CheckStatus.PASS,
        f"Workload Identity \u2192 {wif_annotation}",
      )
    return CheckResult(
      "Kinetic KSA",
      CheckStatus.WARN,
      "KSA exists but missing Workload Identity annotation",
      "Run: kinetic up (configures Workload Identity binding)\n"
      "Without WIF, pods cannot access GCS or Artifact Registry",
    )
  except subprocess.TimeoutExpired:
    return CheckResult("Kinetic KSA", CheckStatus.WARN, "Timed out")


def _check_nvidia_drivers(has_gpu_pools):
  """Check NVIDIA GPU driver DaemonSet is running."""
  if not has_gpu_pools:
    return CheckResult(
      "NVIDIA GPU drivers",
      CheckStatus.SKIP,
      "Skipped (no GPU node pools)",
    )
  try:
    result = subprocess.run(
      ["kubectl", "get", "daemonset", "-n", "kube-system", "-o", "json"],
      capture_output=True,
      text=True,
      timeout=_SUBPROCESS_TIMEOUT,
    )
    if result.returncode != 0:
      return CheckResult(
        "NVIDIA GPU drivers", CheckStatus.WARN, "Could not list DaemonSets"
      )

    try:
      data = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
      return CheckResult(
        "NVIDIA GPU drivers",
        CheckStatus.WARN,
        "Could not parse DaemonSet data",
      )

    nvidia_ds = [
      ds
      for ds in data.get("items", [])
      if "nvidia" in ds.get("metadata", {}).get("name", "").lower()
    ]

    if not nvidia_ds:
      return CheckResult(
        "NVIDIA GPU drivers",
        CheckStatus.WARN,
        "No NVIDIA DaemonSet found (required for GPU workloads)",
        "Run: kinetic up (installs NVIDIA driver DaemonSet)\n"
        "Or manually: kubectl apply -f <nvidia-driver-installer-url>",
      )

    ds = nvidia_ds[0]
    name = ds.get("metadata", {}).get("name", "nvidia-driver")
    ds_status = ds.get("status", {})
    desired = ds_status.get("desiredNumberScheduled", 0)
    ready = ds_status.get("numberReady", 0)
    if desired > 0 and ready < desired:
      return CheckResult(
        "NVIDIA GPU drivers",
        CheckStatus.WARN,
        f"{name}: {ready}/{desired} nodes ready",
        "Some GPU nodes may not have drivers installed yet.\n"
        "Check: kubectl get pods -n kube-system | grep nvidia",
      )
    return CheckResult(
      "NVIDIA GPU drivers",
      CheckStatus.PASS,
      f"{name}: {ready}/{desired} nodes ready",
    )
  except subprocess.TimeoutExpired:
    return CheckResult("NVIDIA GPU drivers", CheckStatus.WARN, "Timed out")


def _check_pending_pods():
  """Check for pods stuck in Pending state (scheduling issues)."""
  try:
    result = subprocess.run(
      [
        "kubectl",
        "get",
        "pods",
        "--all-namespaces",
        "--field-selector=status.phase=Pending",
        "-o",
        "json",
      ],
      capture_output=True,
      text=True,
      timeout=_SUBPROCESS_TIMEOUT,
    )
    if result.returncode != 0:
      return CheckResult(
        "Pending pods", CheckStatus.WARN, "Could not list pods"
      )

    try:
      data = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
      return CheckResult(
        "Pending pods", CheckStatus.WARN, "Could not parse pod data"
      )

    pending = data.get("items", [])
    if not pending:
      return CheckResult("Pending pods", CheckStatus.PASS, "No pending pods")

    # Categorize by scheduling reason.
    reasons = {}
    for pod in pending:
      ns = pod.get("metadata", {}).get("namespace", "?")
      name = pod.get("metadata", {}).get("name", "?")
      reason = "Unknown"
      for cond in pod.get("status", {}).get("conditions", []):
        if cond.get("type") == "PodScheduled" and cond.get("status") == "False":
          reason = cond.get("reason", "Unknown")
          break
      reasons.setdefault(reason, []).append(f"{ns}/{name}")

    parts = []
    hints = []
    for reason, pods in reasons.items():
      parts.append(f"{len(pods)} {reason}")
      if reason == "Unschedulable":
        hints.append(
          "Unschedulable pods: cluster may lack resources or "
          "matching node pools.\n"
          "Check: kubectl describe pod <name> -n <namespace>\n"
          "Add capacity: kinetic pool add --accelerator <spec>"
        )

    has_unschedulable = "Unschedulable" in reasons
    status = CheckStatus.WARN if has_unschedulable else CheckStatus.PASS
    msg = f"{len(pending)} pending: {', '.join(parts)}"
    hint = "\n".join(hints) if hints else ""

    return CheckResult("Pending pods", status, msg, hint)
  except subprocess.TimeoutExpired:
    return CheckResult("Pending pods", CheckStatus.WARN, "Timed out")


def _check_node_conditions():
  """Check node health conditions (disk/memory/PID pressure, NotReady)."""
  try:
    result = subprocess.run(
      ["kubectl", "get", "nodes", "-o", "json"],
      capture_output=True,
      text=True,
      timeout=_SUBPROCESS_TIMEOUT,
    )
    if result.returncode != 0:
      return CheckResult(
        "Node health", CheckStatus.WARN, "Could not list nodes"
      )

    try:
      data = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
      return CheckResult(
        "Node health", CheckStatus.WARN, "Could not parse node data"
      )

    nodes = data.get("items", [])
    if not nodes:
      return CheckResult("Node health", CheckStatus.WARN, "No nodes found")

    pressure_conditions = {"DiskPressure", "MemoryPressure", "PIDPressure"}
    issues = []
    not_ready = []
    for node in nodes:
      name = node.get("metadata", {}).get("name", "?")
      for cond in node.get("status", {}).get("conditions", []):
        ctype = cond.get("type", "")
        cstatus = cond.get("status", "")
        if ctype == "Ready" and cstatus != "True":
          not_ready.append(name)
        elif ctype in pressure_conditions and cstatus == "True":
          issues.append(f"{name}: {ctype}")

    if not_ready:
      names = ", ".join(not_ready[:3])
      suffix = f" (+{len(not_ready) - 3} more)" if len(not_ready) > 3 else ""
      return CheckResult(
        "Node health",
        CheckStatus.FAIL,
        f"{len(not_ready)} node(s) NotReady: {names}{suffix}",
        "Check: kubectl describe node <name>\n"
        "Node may be starting up, or there may be a kubelet issue",
      )

    if issues:
      return CheckResult(
        "Node health",
        CheckStatus.WARN,
        "; ".join(issues[:5]),
        "Nodes under resource pressure may evict pods.\n"
        "Check: kubectl describe node <name> | grep -A5 Conditions",
      )

    return CheckResult(
      "Node health",
      CheckStatus.PASS,
      f"{len(nodes)} node(s), all healthy",
    )
  except subprocess.TimeoutExpired:
    return CheckResult("Node health", CheckStatus.WARN, "Timed out")


def _check_warning_events():
  """Check for recent warning events in the cluster."""
  try:
    result = subprocess.run(
      [
        "kubectl",
        "get",
        "events",
        "--all-namespaces",
        "--field-selector=type=Warning",
        "--sort-by=.lastTimestamp",
        "-o",
        "json",
      ],
      capture_output=True,
      text=True,
      timeout=_SUBPROCESS_TIMEOUT,
    )
    if result.returncode != 0:
      return CheckResult(
        "Cluster events", CheckStatus.WARN, "Could not list events"
      )

    try:
      data = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
      return CheckResult(
        "Cluster events", CheckStatus.WARN, "Could not parse event data"
      )

    events = data.get("items", [])
    if not events:
      return CheckResult(
        "Cluster events", CheckStatus.PASS, "No warning events"
      )

    # Summarize by reason.
    reason_counts = {}
    for evt in events:
      reason = evt.get("reason", "Unknown")
      reason_counts[reason] = reason_counts.get(reason, 0) + 1

    top = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    summary = ", ".join(f"{r} ({c})" for r, c in top)

    concerning = {
      "FailedScheduling",
      "FailedMount",
      "BackOff",
      "OOMKilling",
      "FailedAttachVolume",
      "FailedCreatePodSandBox",
      "Unhealthy",
      "Evicted",
      "NodeNotReady",
      "FreeDiskSpaceFailed",
    }
    has_concerning = any(r in concerning for r in reason_counts)

    status = CheckStatus.WARN if has_concerning else CheckStatus.PASS
    hint = ""
    if has_concerning:
      hint = (
        "Check details: kubectl get events --all-namespaces "
        "--field-selector=type=Warning --sort-by=.lastTimestamp\n"
        "FailedScheduling: insufficient resources or no matching nodes\n"
        "OOMKilling: pods exceeding memory limits\n"
        "BackOff: container crash-looping"
      )

    return CheckResult(
      "Cluster events",
      status,
      f"{len(events)} warning(s): {summary}",
      hint,
    )
  except subprocess.TimeoutExpired:
    return CheckResult("Cluster events", CheckStatus.WARN, "Timed out")


def _check_quota(project, zone):
  """Check GCP compute quota for the region (heuristic)."""
  region = zone_to_region(zone)
  try:
    client = compute_v1.RegionsClient()
    region_info = client.get(project=project, region=region)
  except Exception as exc:
    return CheckResult(
      "GCP quota", CheckStatus.WARN, f"Could not fetch quota info: {exc}"
    )

  gpu_keywords = ("GPU", "NVIDIA", "TPU")
  quota_list = []
  for q in region_info.quotas:
    if not any(kw in q.metric for kw in gpu_keywords):
      continue
    if q.limit > 0:
      quota_list.append((q.metric, q.usage, q.limit))

  # Sort: non-zero usage first, then by name
  quota_list.sort(key=lambda x: (x[1] == 0, x[0]))

  warnings = []
  info_lines = []
  max_len = max(len(metric) for metric, _, _ in quota_list) if quota_list else 0
  for metric, usage, limit in quota_list:
    info_lines.append(
      f"  {metric.ljust(max_len)}: {int(usage):>3}/{int(limit):>3}"
    )
    if limit - usage <= 0:
      warnings.append(f"{metric} exhausted")

  details = (
    "\n".join(info_lines) if info_lines else "  No accelerator quotas found"
  )

  if warnings:
    return CheckResult(
      "GCP quota",
      CheckStatus.WARN,
      f"Checked accelerator quotas in {region} (some exhausted)\n{details}",
      "View and request increases at: "
      f"https://console.cloud.google.com/iam-admin/quotas?project={project}",
    )

  return CheckResult(
    "GCP quota",
    CheckStatus.PASS,
    f"Checked accelerator quotas in {region}\n{details}",
  )


def _check_kubernetes(
  has_kubectl, has_gke_cluster, project, zone, cluster_name
):
  """Run Kubernetes checks."""
  results = []

  # kubeconfig
  if not project:
    results.append(
      CheckResult(
        "kubeconfig context",
        CheckStatus.SKIP,
        "Skipped (requires: Project ID)",
      )
    )
  elif not has_kubectl:
    results.append(
      CheckResult(
        "kubeconfig context",
        CheckStatus.SKIP,
        "Skipped (requires: kubectl)",
      )
    )
  else:
    results.append(_check_kubeconfig(project, zone, cluster_name))

  kubeconfig_ok = results[-1].status == CheckStatus.PASS if results else False

  # Cluster connectivity
  if not has_kubectl or not kubeconfig_ok:
    skip_msg = (
      "Skipped (requires: kubectl)"
      if not has_kubectl
      else ("Skipped (requires: kubeconfig context)")
    )
    results.append(
      CheckResult("Cluster connectivity", CheckStatus.SKIP, skip_msg)
    )
  else:
    results.append(_check_k8s_connectivity())

  connectivity_ok = (
    results[-1].status == CheckStatus.PASS if len(results) >= 2 else False
  )

  # Node pools — can check via gcloud even without kubectl
  has_gpu_pools = False
  if not has_gke_cluster:
    results.append(
      CheckResult(
        "Node pools",
        CheckStatus.SKIP,
        "Skipped (requires: GKE cluster)",
      )
    )
  else:
    pool_result, has_gpu_pools = _check_node_pools(project, zone, cluster_name)
    results.append(pool_result)

  # LWS CRD
  if not has_kubectl or not connectivity_ok:
    results.append(
      CheckResult(
        "LWS CRD",
        CheckStatus.SKIP,
        "Skipped (requires: cluster connectivity)",
      )
    )
  else:
    results.append(_check_lws_crd())

  # Kinetic KSA with Workload Identity
  if not has_kubectl or not connectivity_ok:
    results.append(
      CheckResult(
        "Kinetic KSA",
        CheckStatus.SKIP,
        "Skipped (requires: cluster connectivity)",
      )
    )
  else:
    results.append(_check_kinetic_ksa())

  # NVIDIA GPU drivers (only relevant when GPU pools exist)
  if not has_kubectl or not connectivity_ok:
    results.append(
      CheckResult(
        "NVIDIA GPU drivers",
        CheckStatus.SKIP,
        "Skipped (requires: cluster connectivity)",
      )
    )
  else:
    results.append(_check_nvidia_drivers(has_gpu_pools))

  # Node health conditions
  if not has_kubectl or not connectivity_ok:
    results.append(
      CheckResult(
        "Node health",
        CheckStatus.SKIP,
        "Skipped (requires: cluster connectivity)",
      )
    )
  else:
    results.append(_check_node_conditions())

  # Pending / unschedulable pods
  if not has_kubectl or not connectivity_ok:
    results.append(
      CheckResult(
        "Pending pods",
        CheckStatus.SKIP,
        "Skipped (requires: cluster connectivity)",
      )
    )
  else:
    results.append(_check_pending_pods())

  # Cluster warning events
  if not has_kubectl or not connectivity_ok:
    results.append(
      CheckResult(
        "Cluster events",
        CheckStatus.SKIP,
        "Skipped (requires: cluster connectivity)",
      )
    )
  else:
    results.append(_check_warning_events())

  # GCP quota
  if not has_gke_cluster or not project:
    results.append(
      CheckResult(
        "GCP quota",
        CheckStatus.SKIP,
        "Skipped (requires: GKE cluster)",
      )
    )
  else:
    results.append(_check_quota(project, zone))

  return results


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------


def _print_results(groups):
  """Render grouped results table and fix hints.

  Args:
      groups: List of ``(section_name, [CheckResult, ...])`` tuples.
  """
  all_results = [r for _, checks in groups for r in checks]

  console.print()
  table = Table(
    show_header=True,
    header_style="bold",
    border_style="blue",
    title_style="bold blue",
    pad_edge=True,
    show_lines=False,
  )
  table.add_column("Check", min_width=30)
  table.add_column("Status", min_width=8, justify="center")
  table.add_column("Details")

  for section_name, checks in groups:
    # Section header row.
    table.add_row()
    table.add_row(f"[bold blue]{section_name}[/bold blue]", "", "")
    for r in checks:
      # Dim the entire row for SKIP results.
      if r.status == CheckStatus.SKIP:
        table.add_row(
          f"  [dim]{r.name}[/dim]",
          _STATUS_ICON[r.status],
          f"[dim]{r.message}[/dim]",
        )
      else:
        table.add_row(
          f"  {r.name}",
          _STATUS_ICON[r.status],
          r.message,
        )

  console.print(table)

  # Fix hints for FAIL/WARN items.
  hints = [
    r
    for r in all_results
    if r.fix_hint and r.status in (CheckStatus.FAIL, CheckStatus.WARN)
  ]
  if hints:
    console.print()
    console.print("[bold]Fix suggestions:[/bold]")
    for i, r in enumerate(hints, 1):
      icon = "\u2718" if r.status == CheckStatus.FAIL else "\u25b2"
      style = "red" if r.status == CheckStatus.FAIL else "yellow"
      console.print(f"\n  [{style}]{icon} {i}. {r.name}[/{style}]")
      for line in r.fix_hint.splitlines():
        console.print(f"     {line}")

  # Summary.
  console.print()
  total = len(all_results)
  pass_count = sum(1 for r in all_results if r.status == CheckStatus.PASS)
  fail_count = sum(1 for r in all_results if r.status == CheckStatus.FAIL)
  warn_count = sum(1 for r in all_results if r.status == CheckStatus.WARN)
  skip_count = sum(1 for r in all_results if r.status == CheckStatus.SKIP)

  parts = [f"{total} checks:"]
  if pass_count:
    parts.append(f"[green]{pass_count} passed[/green]")
  if fail_count:
    parts.append(f"[red]{fail_count} failed[/red]")
  if warn_count:
    parts.append(f"[yellow]{warn_count} warnings[/yellow]")
  if skip_count:
    parts.append(f"[dim]{skip_count} skipped[/dim]")
  console.print(", ".join(parts))

  if not fail_count and not warn_count:
    console.print("[green]All checks passed![/green]")
  console.print()


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


def _emit_progress(panel, section, results):
  """Emit progress lines to the live panel for a completed group."""
  panel.on_output(f"[{section}]")
  for r in results:
    panel.on_output(f"  {_PROGRESS_ICON[r.status]} {r.name}")


@click.command()
@common_options
def doctor(project, zone, cluster_name):
  """Check environment, credentials, and infrastructure health."""
  banner("kinetic Doctor")

  # Resolve values without prompting.
  project = project or get_default_project()
  zone = zone or get_default_zone()
  cluster_name = cluster_name or get_default_cluster_name()

  groups = []

  with LiveOutputPanel(
    "Running diagnostics",
    transient=True,
    show_subtitle=False,
    max_lines=10,
  ) as panel:
    # Group 1: Local tools.
    panel.on_output("Checking local tools...")
    tool_results = _check_local_tools()
    groups.append((_SECTIONS[0], tool_results))
    _emit_progress(panel, _SECTIONS[0], tool_results)
    has_gcloud = tool_results[0].status == CheckStatus.PASS
    has_kubectl = tool_results[1].status == CheckStatus.PASS

    # Group 2: Authentication.
    panel.on_output("Checking authentication...")
    auth_results = _check_auth(has_gcloud)
    groups.append((_SECTIONS[1], auth_results))
    _emit_progress(panel, _SECTIONS[1], auth_results)
    has_adc = auth_results[0].status == CheckStatus.PASS

    # Group 3: Configuration.
    config_results = _check_config(project, zone, cluster_name)
    groups.append((_SECTIONS[2], config_results))
    _emit_progress(panel, _SECTIONS[2], config_results)

    # Group 4: GCP project.
    panel.on_output("Checking GCP project access...")
    project_results = _check_gcp_project(has_gcloud, has_adc, project)
    groups.append((_SECTIONS[3], project_results))
    _emit_progress(panel, _SECTIONS[3], project_results)
    has_project_access = project_results[0].status == CheckStatus.PASS

    # Group 5: GCP APIs.
    panel.on_output("Checking GCP APIs...")
    api_results = _check_apis(has_project_access, project)
    groups.append((_SECTIONS[4], api_results))
    _emit_progress(panel, _SECTIONS[4], api_results)

    # Group 6: GCP Resources.
    panel.on_output("Checking GCP resources...")
    resource_results = _check_gcp_resources(
      has_project_access, project, zone, cluster_name
    )
    groups.append((_SECTIONS[5], resource_results))
    _emit_progress(panel, _SECTIONS[5], resource_results)

    # Group 7: Infrastructure.
    panel.on_output("Checking infrastructure...")
    infra_results = _check_infra(
      has_project_access, project, zone, cluster_name
    )
    groups.append((_SECTIONS[6], infra_results))
    _emit_progress(panel, _SECTIONS[6], infra_results)
    has_gke_cluster = any(
      r.name == "GKE cluster" and r.status == CheckStatus.PASS
      for r in infra_results
    )

    # Group 8: Kubernetes.
    panel.on_output("Checking Kubernetes...")
    k8s_results = _check_kubernetes(
      has_kubectl, has_gke_cluster, project, zone, cluster_name
    )
    groups.append((_SECTIONS[7], k8s_results))
    _emit_progress(panel, _SECTIONS[7], k8s_results)

    # Mark panel as error if any check failed (turns border yellow).
    all_results = [r for _, checks in groups for r in checks]
    if any(r.status == CheckStatus.FAIL for r in all_results):
      panel.mark_error()

  _print_results(groups)

  all_results = [r for _, checks in groups for r in checks]
  if any(r.status == CheckStatus.FAIL for r in all_results):
    sys.exit(1)
