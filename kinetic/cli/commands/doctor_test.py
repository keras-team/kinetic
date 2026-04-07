"""Tests for kinetic.cli.commands.doctor — environment diagnostics."""

import contextlib
import json
import subprocess
from types import SimpleNamespace
from unittest import mock

import google.auth.exceptions
from absl.testing import absltest
from click.testing import CliRunner
from google.api_core import exceptions as google_exceptions

from kinetic.cli.commands.doctor import (
  CheckResult,
  CheckStatus,
  doctor,
)

_MODULE = "kinetic.cli.commands.doctor"

# Shared CLI args.
_CLI_ARGS = [
  "--project",
  "test-project",
  "--zone",
  "us-central2-b",
  "--cluster",
  "test-cluster",
]


def _mock_which(binaries_present):
  """Return a shutil.which mock that resolves only the given binaries."""

  def fake_which(binary):
    if binary in binaries_present:
      return f"/usr/bin/{binary}"
    return None

  return fake_which


def _all_tools_which(binary):
  """shutil.which that finds all required binaries."""
  known = {"gcloud", "kubectl", "gke-gcloud-auth-plugin"}
  return f"/usr/bin/{binary}" if binary in known else None


def _mock_adc_pass():
  """Return a context manager that mocks google.auth.default to succeed."""
  mock_creds = mock.MagicMock()
  return mock.patch(
    f"{_MODULE}.google.auth.default",
    return_value=(mock_creds, "test-project"),
  )


def _make_service_mock(config_name):
  """Create a mock service object for ServiceUsageClient.list_services."""
  svc = SimpleNamespace()
  svc.config = SimpleNamespace(name=config_name)
  return svc


def _make_node_pool(name, status_value, accelerators=None):
  """Create a mock NodePool object for ClusterManagerClient.list_node_pools.

  Args:
      name: Pool name string.
      status_value: Integer status (2=RUNNING, 5=ERROR, etc.).
      accelerators: List of dicts with 'accelerator_type' key, or None.
  """
  accel_objs = []
  if accelerators:
    for a in accelerators:
      accel_objs.append(SimpleNamespace(accelerator_type=a["accelerator_type"]))
  config = SimpleNamespace(accelerators=accel_objs)
  return SimpleNamespace(name=name, status=status_value, config=config)


def _make_quota(metric, usage, limit):
  """Create a mock Quota object for RegionsClient.get."""
  return SimpleNamespace(metric=metric, usage=usage, limit=limit)


def _sdk_patches(
  *,
  project_ok=True,
  billing_ok=True,
  apis_enabled=None,
  sa_ok=True,
  ar_ok=True,
  buckets_ok=True,
  vpc_ok=True,
  nat_ok=True,
  cluster_status=2,  # 2=RUNNING
  node_pools=None,
  quotas=None,
):
  """Return a dict of mock.patch context managers for all SDK clients.

  All default to a healthy/passing state so individual tests can
  override only the parameter they care about.
  """
  if apis_enabled is None:
    apis_enabled = [
      "compute.googleapis.com",
      "cloudbuild.googleapis.com",
      "artifactregistry.googleapis.com",
      "storage.googleapis.com",
      "container.googleapis.com",
      "secretmanager.googleapis.com",
      "iam.googleapis.com",
    ]
  if node_pools is None:
    node_pools = [_make_node_pool("default-pool", 2)]  # RUNNING=2
  if quotas is None:
    quotas = []

  patches = {}

  # resourcemanager — project access
  rm_client = mock.MagicMock()
  if project_ok:
    rm_client.get_project.return_value = SimpleNamespace(
      name="projects/test-project"
    )
  else:
    rm_client.get_project.side_effect = google_exceptions.NotFound("not found")
  patches["rm"] = mock.patch(
    f"{_MODULE}.resourcemanager_v3.ProjectsClient", return_value=rm_client
  )

  # billing
  billing_client = mock.MagicMock()
  billing_client.get_project_billing_info.return_value = SimpleNamespace(
    billing_enabled=billing_ok
  )
  patches["billing"] = mock.patch(
    f"{_MODULE}.billing_v1.CloudBillingClient", return_value=billing_client
  )

  # service usage — APIs
  su_client = mock.MagicMock()
  su_client.list_services.return_value = [
    _make_service_mock(api) for api in apis_enabled
  ]
  patches["su"] = mock.patch(
    f"{_MODULE}.service_usage_v1.ServiceUsageClient", return_value=su_client
  )

  # IAM — service accounts
  iam_client = mock.MagicMock()
  if sa_ok:
    iam_client.get_service_account.return_value = SimpleNamespace(
      email="sa@test.iam"
    )
  else:
    iam_client.get_service_account.side_effect = google_exceptions.NotFound(
      "not found"
    )
  patches["iam"] = mock.patch(
    f"{_MODULE}.iam_admin_v1.IAMClient", return_value=iam_client
  )

  # Artifact Registry
  ar_client = mock.MagicMock()
  if ar_ok:
    ar_client.get_repository.return_value = SimpleNamespace(name="repo")
  else:
    ar_client.get_repository.side_effect = google_exceptions.NotFound(
      "not found"
    )
  patches["ar"] = mock.patch(
    f"{_MODULE}.artifactregistry_v1.ArtifactRegistryClient",
    return_value=ar_client,
  )

  # Storage — buckets
  storage_client = mock.MagicMock()
  if buckets_ok:
    storage_client.get_bucket.return_value = SimpleNamespace(name="bucket")
  else:
    storage_client.get_bucket.side_effect = google_exceptions.NotFound(
      "not found"
    )
  patches["storage"] = mock.patch(
    f"{_MODULE}.storage.Client", return_value=storage_client
  )

  # Compute — VPC
  networks_client = mock.MagicMock()
  if vpc_ok:
    networks_client.get.return_value = SimpleNamespace(name="kn-test-cluster")
  else:
    networks_client.get.side_effect = google_exceptions.NotFound("not found")
  patches["networks"] = mock.patch(
    f"{_MODULE}.compute_v1.NetworksClient", return_value=networks_client
  )

  # Compute — Cloud NAT (via router)
  routers_client = mock.MagicMock()
  if nat_ok:
    nat_obj = SimpleNamespace(name="kn-test-cluster-nat")
    routers_client.get.return_value = SimpleNamespace(nats=[nat_obj])
  else:
    routers_client.get.side_effect = google_exceptions.NotFound("not found")
  patches["routers"] = mock.patch(
    f"{_MODULE}.compute_v1.RoutersClient", return_value=routers_client
  )

  # Container — GKE cluster
  container_client = mock.MagicMock()
  if cluster_status is None:
    container_client.get_cluster.side_effect = google_exceptions.NotFound(
      "not found"
    )
    container_client.list_node_pools.return_value = SimpleNamespace(
      node_pools=[]
    )
  else:
    container_client.get_cluster.return_value = SimpleNamespace(
      status=cluster_status
    )
    container_client.list_node_pools.return_value = SimpleNamespace(
      node_pools=node_pools
    )
  patches["container"] = mock.patch(
    f"{_MODULE}.container_v1.ClusterManagerClient",
    return_value=container_client,
  )

  # Compute — quota
  regions_client = mock.MagicMock()
  regions_client.get.return_value = SimpleNamespace(quotas=quotas)
  patches["regions"] = mock.patch(
    f"{_MODULE}.compute_v1.RegionsClient", return_value=regions_client
  )

  return patches


# ---------------------------------------------------------------------------
# subprocess mock (only for gcloud account + kubectl commands now)
# ---------------------------------------------------------------------------


def _make_run_side_effect(
  *,
  account="user@example.com",
  k8s_ok=True,
  lws_ok=True,
):
  """Build a subprocess.run side-effect for remaining subprocess calls.

  Only handles gcloud account and kubectl commands — ADC uses
  google.auth.default (mocked separately) and all GCP resource checks
  use SDK clients mocked via _sdk_patches().
  """

  def side_effect(args, **kwargs):
    cmd = args if isinstance(args, list) else list(args)
    text = kwargs.get("text", False)

    # Account check.
    if "get-value" in cmd and "account" in cmd:
      out = account if account else "(unset)"
      if text:
        return subprocess.CompletedProcess(
          args=cmd, returncode=0, stdout=out, stderr=""
        )
      return subprocess.CompletedProcess(
        args=cmd, returncode=0, stdout=out.encode(), stderr=b""
      )

    # kubectl cluster-info.
    if "cluster-info" in cmd:
      rc = 0 if k8s_ok else 1
      return subprocess.CompletedProcess(
        args=cmd, returncode=rc, stdout=b"", stderr=b""
      )

    # kubectl get crd (LWS).
    if "get" in cmd and "crd" in cmd:
      rc = 0 if lws_ok else 1
      return subprocess.CompletedProcess(
        args=cmd, returncode=rc, stdout=b"", stderr=b""
      )

    # kubectl get serviceaccount (KSA check).
    if "get" in cmd and "serviceaccount" in cmd:
      ksa_data = json.dumps(
        {
          "metadata": {
            "name": "kinetic",
            "annotations": {
              "iam.gke.io/gcp-service-account": (
                "kn-test-cluster-nodes@test-project.iam.gserviceaccount.com"
              )
            },
          }
        }
      )
      if text:
        return subprocess.CompletedProcess(
          args=cmd, returncode=0, stdout=ksa_data, stderr=""
        )
      return subprocess.CompletedProcess(
        args=cmd, returncode=0, stdout=ksa_data.encode(), stderr=b""
      )

    # kubectl get daemonset (NVIDIA drivers check).
    if "get" in cmd and "daemonset" in cmd:
      ds_data = json.dumps({"items": []})
      if text:
        return subprocess.CompletedProcess(
          args=cmd, returncode=0, stdout=ds_data, stderr=""
        )
      return subprocess.CompletedProcess(
        args=cmd, returncode=0, stdout=ds_data.encode(), stderr=b""
      )

    # kubectl get pods (pending pods check).
    if "get" in cmd and "pods" in cmd:
      pods_data = json.dumps({"items": []})
      if text:
        return subprocess.CompletedProcess(
          args=cmd, returncode=0, stdout=pods_data, stderr=""
        )
      return subprocess.CompletedProcess(
        args=cmd, returncode=0, stdout=pods_data.encode(), stderr=b""
      )

    # kubectl get nodes (node health check).
    if "get" in cmd and "nodes" in cmd:
      nodes_data = json.dumps(
        {
          "items": [
            {
              "metadata": {"name": "node-1"},
              "status": {
                "conditions": [
                  {"type": "Ready", "status": "True"},
                  {"type": "DiskPressure", "status": "False"},
                  {"type": "MemoryPressure", "status": "False"},
                  {"type": "PIDPressure", "status": "False"},
                ]
              },
            }
          ]
        }
      )
      if text:
        return subprocess.CompletedProcess(
          args=cmd, returncode=0, stdout=nodes_data, stderr=""
        )
      return subprocess.CompletedProcess(
        args=cmd, returncode=0, stdout=nodes_data.encode(), stderr=b""
      )

    # kubectl get events (warning events check).
    if "get" in cmd and "events" in cmd:
      events_data = json.dumps({"items": []})
      if text:
        return subprocess.CompletedProcess(
          args=cmd, returncode=0, stdout=events_data, stderr=""
        )
      return subprocess.CompletedProcess(
        args=cmd, returncode=0, stdout=events_data.encode(), stderr=b""
      )

    # Default fallback.
    return subprocess.CompletedProcess(
      args=cmd, returncode=0, stdout=b"", stderr=b""
    )

  return side_effect


# ---------------------------------------------------------------------------
# Helper: enter all SDK patches + common mocks at once
# ---------------------------------------------------------------------------


def _enter_patches(
  stack,
  sdk_overrides=None,
  run_overrides=None,
  pulumi_dir_exists=True,
  pulumi_files=None,
  which_fn=None,
):
  """Enter all patches into an ExitStack and return the mock_kube handle.

  Args:
      stack: contextlib.ExitStack to register patches in.
      sdk_overrides: kwargs forwarded to _sdk_patches().
      run_overrides: kwargs forwarded to _make_run_side_effect().
      pulumi_dir_exists: Whether the Pulumi state dir exists.
      pulumi_files: Files returned by os.listdir for the stacks dir.
      which_fn: Override for shutil.which. Defaults to _all_tools_which.
  """
  sdk = _sdk_patches(**(sdk_overrides or {}))
  run_se = _make_run_side_effect(**(run_overrides or {}))
  if pulumi_files is None:
    pulumi_files = ["test-project-test-cluster.json"]

  stack.enter_context(
    mock.patch(
      f"{_MODULE}.shutil.which", side_effect=which_fn or _all_tools_which
    )
  )
  stack.enter_context(
    mock.patch(f"{_MODULE}.subprocess.run", side_effect=run_se)
  )
  stack.enter_context(_mock_adc_pass())
  stack.enter_context(
    mock.patch(f"{_MODULE}.os.path.isdir", return_value=pulumi_dir_exists)
  )
  stack.enter_context(
    mock.patch(f"{_MODULE}.os.listdir", return_value=pulumi_files)
  )
  mock_kube = stack.enter_context(mock.patch(f"{_MODULE}._check_kubeconfig"))
  for p in sdk.values():
    stack.enter_context(p)

  return mock_kube


class DoctorAllPassTest(absltest.TestCase):
  """All checks pass — exit code 0."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()

  def test_all_pass(self):
    with contextlib.ExitStack() as stack:
      mock_kube = _enter_patches(stack)
      mock_kube.return_value = CheckResult(
        "kubeconfig context",
        CheckStatus.PASS,
        "gke_test-project_us-central2-b_test-cluster",
      )
      result = self.runner.invoke(doctor, _CLI_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("All checks passed", result.output)


class DoctorGcloudMissingTest(absltest.TestCase):
  """gcloud missing — FAIL + downstream SKIP."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()

  def test_gcloud_missing(self):
    with (
      mock.patch(
        f"{_MODULE}.shutil.which",
        side_effect=_mock_which({"kubectl", "gke-gcloud-auth-plugin"}),
      ),
      mock.patch(
        f"{_MODULE}.google.auth.default",
        side_effect=google.auth.exceptions.DefaultCredentialsError("none"),
      ),
    ):
      result = self.runner.invoke(doctor, _CLI_ARGS)

    self.assertEqual(result.exit_code, 1, result.output)
    self.assertIn("FAIL", result.output)
    self.assertIn("gcloud CLI", result.output)
    # ADC check runs (FAIL), gcloud account check is SKIP.
    self.assertIn("SKIP", result.output)


class DoctorKubectlMissingTest(absltest.TestCase):
  """kubectl missing — FAIL for kubectl, other checks still run."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()

  def test_kubectl_missing(self):
    with contextlib.ExitStack() as stack:
      _enter_patches(
        stack,
        which_fn=_mock_which({"gcloud", "gke-gcloud-auth-plugin"}),
      )
      result = self.runner.invoke(doctor, _CLI_ARGS)

    self.assertEqual(result.exit_code, 1, result.output)
    self.assertIn("FAIL", result.output)
    # Auth checks should still run (gcloud is present).
    self.assertIn("Application Default Credentials", result.output)


class DoctorAdcNotConfiguredTest(absltest.TestCase):
  """ADC not configured — FAIL with fix hints."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()

  def test_adc_fail(self):
    with (
      mock.patch(f"{_MODULE}.shutil.which", side_effect=_all_tools_which),
      mock.patch(
        f"{_MODULE}.subprocess.run",
        side_effect=_make_run_side_effect(),
      ),
      mock.patch(
        f"{_MODULE}.google.auth.default",
        side_effect=google.auth.exceptions.DefaultCredentialsError("none"),
      ),
    ):
      result = self.runner.invoke(doctor, _CLI_ARGS)

    self.assertEqual(result.exit_code, 1, result.output)
    self.assertIn("FAIL", result.output)
    self.assertIn("gcloud auth application-default login", result.output)
    self.assertIn("GOOGLE_APPLICATION_CREDENTIALS", result.output)


class DoctorProjectNotSetTest(absltest.TestCase):
  """Project not set — WARN, downstream checks SKIP."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()

  def test_no_project(self):
    with (
      mock.patch(f"{_MODULE}.shutil.which", side_effect=_all_tools_which),
      mock.patch(
        f"{_MODULE}.subprocess.run", side_effect=_make_run_side_effect()
      ),
      _mock_adc_pass(),
      mock.patch(f"{_MODULE}.get_default_project", return_value=None),
    ):
      # Invoke without --project flag.
      result = self.runner.invoke(
        doctor,
        [
          "--zone",
          "us-central2-b",
          "--cluster",
          "test-cluster",
        ],
      )

    # WARN, not FAIL, so exit 0.
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("WARN", result.output)
    self.assertIn("SKIP", result.output)


class DoctorBillingNotEnabledTest(absltest.TestCase):
  """Billing not enabled — FAIL (tests billing_enabled=False branch)."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()

  def test_billing_disabled(self):
    with contextlib.ExitStack() as stack:
      _enter_patches(stack, sdk_overrides={"billing_ok": False})
      result = self.runner.invoke(doctor, _CLI_ARGS)

    self.assertEqual(result.exit_code, 1, result.output)
    self.assertIn("Billing", result.output)
    self.assertIn("\u2718", result.output)  # ✘ icon


class DoctorApiNotEnabledTest(absltest.TestCase):
  """One API not enabled — FAIL with fix hint."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()

  def test_missing_api(self):
    enabled = [
      "compute.googleapis.com",
      # cloudbuild missing
      "artifactregistry.googleapis.com",
      "storage.googleapis.com",
      "container.googleapis.com",
    ]
    with contextlib.ExitStack() as stack:
      mock_kube = _enter_patches(stack, sdk_overrides={"apis_enabled": enabled})
      mock_kube.return_value = CheckResult(
        "kubeconfig context", CheckStatus.PASS, "ok"
      )
      result = self.runner.invoke(doctor, _CLI_ARGS)

    self.assertEqual(result.exit_code, 1, result.output)
    self.assertIn("cloudbuild.googleapis.com", result.output)
    self.assertIn("FAIL", result.output)
    self.assertIn("gcloud services enable", result.output)


class DoctorResourceNotFoundTest(absltest.TestCase):
  """SDK NotFound → exit 1 for each resource type (parameterized).

  Each case tests the same trivial pattern: SDK raises NotFound →
  check returns FAIL → doctor exits 1.  One parameterized test covers
  all resources instead of one class per resource.
  """

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()

  # (sdk_override_key, expected_text_in_output)
  _CASES = [
    ("project_ok", "GCP project access"),
    ("sa_ok", "service account"),
    ("ar_ok", "Artifact Registry"),
    ("buckets_ok", "bucket"),
    ("vpc_ok", "VPC network"),
    ("nat_ok", "Cloud NAT"),
    ("cluster_status", "GKE cluster"),
  ]

  def test_not_found_causes_fail(self):
    for override_key, expected_text in self._CASES:
      with self.subTest(resource=override_key):
        # cluster_status=None triggers NotFound; others use False.
        value = None if override_key == "cluster_status" else False
        with contextlib.ExitStack() as stack:
          mock_kube = _enter_patches(stack, sdk_overrides={override_key: value})
          mock_kube.return_value = CheckResult(
            "kubeconfig context", CheckStatus.PASS, "ok"
          )
          result = self.runner.invoke(doctor, _CLI_ARGS)

        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn(
          expected_text,
          result.output.lower() if expected_text.islower() else result.output,
        )


class DoctorNoPulumiStateTest(absltest.TestCase):
  """No Pulumi state — WARN with available stacks."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()

  def test_no_state_dir(self):
    with contextlib.ExitStack() as stack:
      mock_kube = _enter_patches(stack, pulumi_dir_exists=False)
      mock_kube.return_value = CheckResult(
        "kubeconfig context", CheckStatus.PASS, "ok"
      )
      result = self.runner.invoke(doctor, _CLI_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("WARN", result.output)
    self.assertIn("State directory not found", result.output)

  def test_wrong_stack_shows_available(self):
    with contextlib.ExitStack() as stack:
      mock_kube = _enter_patches(
        stack,
        pulumi_files=["other-project-other-cluster.json"],
      )
      mock_kube.return_value = CheckResult(
        "kubeconfig context", CheckStatus.PASS, "ok"
      )
      result = self.runner.invoke(doctor, _CLI_ARGS)

    self.assertIn("Available:", result.output)
    self.assertIn("other-project-other-cluster", result.output)


class DoctorGkeClusterNotRunningTest(absltest.TestCase):
  """GKE cluster status enum mapping — PROVISIONING is WARN, not FAIL."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()

  def test_cluster_provisioning(self):
    # PROVISIONING = 1
    with contextlib.ExitStack() as stack:
      mock_kube = _enter_patches(stack, sdk_overrides={"cluster_status": 1})
      mock_kube.return_value = CheckResult(
        "kubeconfig context", CheckStatus.PASS, "ok"
      )
      result = self.runner.invoke(doctor, _CLI_ARGS)

    # PROVISIONING is WARN, not FAIL.
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("PROVISIONING", result.output)


class DoctorNodePoolUnhealthyTest(absltest.TestCase):
  """Unhealthy node pool — WARN."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()

  def test_unhealthy_pool(self):
    pools = [
      _make_node_pool("default-pool", 2),  # RUNNING=2
      _make_node_pool(
        "gpu-pool",
        6,  # ERROR=6
        accelerators=[{"accelerator_type": "nvidia-tesla-t4"}],
      ),
    ]
    with contextlib.ExitStack() as stack:
      mock_kube = _enter_patches(stack, sdk_overrides={"node_pools": pools})
      mock_kube.return_value = CheckResult(
        "kubeconfig context", CheckStatus.PASS, "ok"
      )
      result = self.runner.invoke(doctor, _CLI_ARGS)

    self.assertIn("gpu-pool", result.output)
    self.assertIn("ERROR", result.output)


class DoctorKubeconfigMismatchTest(absltest.TestCase):
  """kubeconfig mismatch — WARN."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()

  def test_wrong_context(self):
    with contextlib.ExitStack() as stack:
      mock_kube = _enter_patches(stack)
      mock_kube.return_value = CheckResult(
        "kubeconfig context",
        CheckStatus.WARN,
        "Active: 'gke_other' (expected: 'gke_test-project_us-central2-b_test-cluster')",
        "Run: gcloud container clusters get-credentials ...",
      )
      result = self.runner.invoke(doctor, _CLI_ARGS)

    # WARN, not FAIL.
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("WARN", result.output)


class DoctorLwsMissingTest(absltest.TestCase):
  """LWS CRD missing — WARN (not FAIL)."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()

  def test_lws_missing(self):
    with contextlib.ExitStack() as stack:
      mock_kube = _enter_patches(stack, run_overrides={"lws_ok": False})
      mock_kube.return_value = CheckResult(
        "kubeconfig context", CheckStatus.PASS, "ok"
      )
      result = self.runner.invoke(doctor, _CLI_ARGS)

    # LWS missing is WARN, so exit 0.
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("LWS CRD", result.output)
    self.assertIn("WARN", result.output)


class DoctorExitCodeTest(absltest.TestCase):
  """Exit code is 1 on any FAIL, 0 on only WARN/SKIP."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()

  def test_exit_1_on_fail(self):
    """Any FAIL → exit 1."""
    with (
      mock.patch(f"{_MODULE}.shutil.which", side_effect=_mock_which(set())),
      mock.patch(
        f"{_MODULE}.google.auth.default",
        side_effect=google.auth.exceptions.DefaultCredentialsError("none"),
      ),
    ):
      result = self.runner.invoke(doctor, _CLI_ARGS)

    self.assertEqual(result.exit_code, 1, result.output)

  def test_exit_0_on_warn_only(self):
    """Only WARN/SKIP → exit 0."""
    with (
      mock.patch(f"{_MODULE}.shutil.which", side_effect=_all_tools_which),
      mock.patch(
        f"{_MODULE}.subprocess.run", side_effect=_make_run_side_effect()
      ),
      _mock_adc_pass(),
      mock.patch(f"{_MODULE}.get_default_project", return_value=None),
    ):
      result = self.runner.invoke(
        doctor,
        [
          "--zone",
          "us-central2-b",
          "--cluster",
          "test-cluster",
        ],
      )

    self.assertEqual(result.exit_code, 0, result.output)


if __name__ == "__main__":
  absltest.main()
