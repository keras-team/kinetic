"""Tests for kinetic.cli.commands.doctor — environment diagnostics."""

import json
import subprocess
from unittest import mock

from absl.testing import absltest
from click.testing import CliRunner

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


def _mock_run_success(text_output=""):
  """Return a CompletedProcess with returncode 0."""
  return subprocess.CompletedProcess(
    args=[], returncode=0, stdout=text_output.encode(), stderr=b""
  )


def _mock_run_failure():
  """Return a CompletedProcess with returncode 1."""
  return subprocess.CompletedProcess(
    args=[], returncode=1, stdout=b"", stderr=b"error"
  )


def _all_tools_which(binary):
  """shutil.which that finds all required binaries."""
  known = {"gcloud", "kubectl", "gke-gcloud-auth-plugin"}
  return f"/usr/bin/{binary}" if binary in known else None


def _make_run_side_effect(
  *,
  adc_ok=True,
  account="user@example.com",
  project_ok=True,
  billing_ok=True,
  apis_enabled=None,
  cluster_status="RUNNING",
  node_pools=None,
  k8s_ok=True,
  lws_ok=True,
  quota_data=None,
):
  """Build a subprocess.run side-effect that routes by command args."""
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
    node_pools = [{"name": "default-pool", "status": "RUNNING", "config": {}}]
  if quota_data is None:
    quota_data = {"quotas": []}

  def side_effect(args, **kwargs):
    cmd = args if isinstance(args, list) else list(args)
    text = kwargs.get("text", False)

    # ADC check.
    if "application-default" in cmd and "print-access-token" in cmd:
      rc = 0 if adc_ok else 1
      stdout = b"token" if adc_ok else b""
      return subprocess.CompletedProcess(
        args=cmd, returncode=rc, stdout=stdout, stderr=b""
      )

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

    # Billing (must be before generic project describe since both contain "projects" + "describe").
    if "billing" in cmd and "describe" in cmd:
      val = "True" if billing_ok else "False"
      if text:
        return subprocess.CompletedProcess(
          args=cmd, returncode=0, stdout=val, stderr=""
        )
      return subprocess.CompletedProcess(
        args=cmd, returncode=0, stdout=val.encode(), stderr=b""
      )

    # Project describe.
    if "projects" in cmd and "describe" in cmd:
      rc = 0 if project_ok else 1
      return subprocess.CompletedProcess(
        args=cmd, returncode=rc, stdout=b"", stderr=b""
      )

    # Enabled APIs.
    if "services" in cmd and "list" in cmd:
      out = "\n".join(apis_enabled)
      if text:
        return subprocess.CompletedProcess(
          args=cmd, returncode=0, stdout=out, stderr=""
        )
      return subprocess.CompletedProcess(
        args=cmd, returncode=0, stdout=out.encode(), stderr=b""
      )

    # Cluster describe.
    if "clusters" in cmd and "describe" in cmd:
      if cluster_status is None:
        return subprocess.CompletedProcess(
          args=cmd, returncode=1, stdout=b"", stderr=b"not found"
        )
      if text:
        return subprocess.CompletedProcess(
          args=cmd, returncode=0, stdout=cluster_status, stderr=""
        )
      return subprocess.CompletedProcess(
        args=cmd, returncode=0, stdout=cluster_status.encode(), stderr=b""
      )

    # Node pools list.
    if "node-pools" in cmd and "list" in cmd:
      out = json.dumps(node_pools)
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
              "iam.gke.io/gcp-service-account": "kn-test-cluster-nodes@test-project.iam.gserviceaccount.com"
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

    # Quota.
    if "regions" in cmd and "describe" in cmd:
      out = json.dumps(quota_data)
      if text:
        return subprocess.CompletedProcess(
          args=cmd, returncode=0, stdout=out, stderr=""
        )
      return subprocess.CompletedProcess(
        args=cmd, returncode=0, stdout=out.encode(), stderr=b""
      )

    # Default fallback.
    return subprocess.CompletedProcess(
      args=cmd, returncode=0, stdout=b"", stderr=b""
    )

  return side_effect


class DoctorAllPassTest(absltest.TestCase):
  """All checks pass — exit code 0."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()

  def test_all_pass(self):
    with (
      mock.patch(f"{_MODULE}.shutil.which", side_effect=_all_tools_which),
      mock.patch(
        f"{_MODULE}.subprocess.run", side_effect=_make_run_side_effect()
      ),
      mock.patch(f"{_MODULE}.os.path.isdir", return_value=True),
      mock.patch(
        f"{_MODULE}.os.listdir", return_value=["test-project-test-cluster.json"]
      ),
      mock.patch(f"{_MODULE}._check_kubeconfig") as mock_kube,
    ):
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
    with mock.patch(
      f"{_MODULE}.shutil.which",
      side_effect=_mock_which({"kubectl", "gke-gcloud-auth-plugin"}),
    ):
      result = self.runner.invoke(doctor, _CLI_ARGS)

    self.assertEqual(result.exit_code, 1, result.output)
    self.assertIn("FAIL", result.output)
    self.assertIn("gcloud CLI", result.output)
    # Auth should be skipped.
    self.assertIn("SKIP", result.output)


class DoctorKubectlMissingTest(absltest.TestCase):
  """kubectl missing — FAIL for kubectl, other checks still run."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()

  def test_kubectl_missing(self):
    with (
      mock.patch(
        f"{_MODULE}.shutil.which",
        side_effect=_mock_which({"gcloud", "gke-gcloud-auth-plugin"}),
      ),
      mock.patch(
        f"{_MODULE}.subprocess.run", side_effect=_make_run_side_effect()
      ),
      mock.patch(f"{_MODULE}.os.path.isdir", return_value=True),
      mock.patch(
        f"{_MODULE}.os.listdir", return_value=["test-project-test-cluster.json"]
      ),
    ):
      result = self.runner.invoke(doctor, _CLI_ARGS)

    self.assertEqual(result.exit_code, 1, result.output)
    self.assertIn("FAIL", result.output)
    # Auth checks should still run (gcloud is present).
    self.assertIn("gcloud auth (ADC)", result.output)


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
        side_effect=_make_run_side_effect(adc_ok=False),
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


class DoctorProjectNotAccessibleTest(absltest.TestCase):
  """Project not accessible — FAIL with kinetic up in hint."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()

  def test_project_not_found(self):
    with (
      mock.patch(f"{_MODULE}.shutil.which", side_effect=_all_tools_which),
      mock.patch(
        f"{_MODULE}.subprocess.run",
        side_effect=_make_run_side_effect(project_ok=False),
      ),
    ):
      result = self.runner.invoke(doctor, _CLI_ARGS)

    self.assertEqual(result.exit_code, 1, result.output)
    self.assertIn("FAIL", result.output)
    self.assertIn("kinetic up", result.output)


class DoctorBillingNotEnabledTest(absltest.TestCase):
  """Billing not enabled — FAIL."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()

  def test_billing_disabled(self):
    with (
      mock.patch(f"{_MODULE}.shutil.which", side_effect=_all_tools_which),
      mock.patch(
        f"{_MODULE}.subprocess.run",
        side_effect=_make_run_side_effect(billing_ok=False),
      ),
    ):
      result = self.runner.invoke(doctor, _CLI_ARGS)

    self.assertEqual(result.exit_code, 1, result.output)
    self.assertIn("Billing", result.output)
    self.assertIn("FAIL", result.output)


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
    with (
      mock.patch(f"{_MODULE}.shutil.which", side_effect=_all_tools_which),
      mock.patch(
        f"{_MODULE}.subprocess.run",
        side_effect=_make_run_side_effect(apis_enabled=enabled),
      ),
      mock.patch(f"{_MODULE}.os.path.isdir", return_value=True),
      mock.patch(
        f"{_MODULE}.os.listdir", return_value=["test-project-test-cluster.json"]
      ),
      mock.patch(f"{_MODULE}._check_kubeconfig") as mock_kube,
    ):
      mock_kube.return_value = CheckResult(
        "kubeconfig context", CheckStatus.PASS, "ok"
      )
      result = self.runner.invoke(doctor, _CLI_ARGS)

    self.assertEqual(result.exit_code, 1, result.output)
    self.assertIn("cloudbuild.googleapis.com", result.output)
    self.assertIn("FAIL", result.output)
    self.assertIn("gcloud services enable", result.output)


class DoctorNoPulumiStateTest(absltest.TestCase):
  """No Pulumi state — WARN with available stacks."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()

  def test_no_state_dir(self):
    with (
      mock.patch(f"{_MODULE}.shutil.which", side_effect=_all_tools_which),
      mock.patch(
        f"{_MODULE}.subprocess.run", side_effect=_make_run_side_effect()
      ),
      mock.patch(f"{_MODULE}.os.path.isdir", return_value=False),
      mock.patch(f"{_MODULE}._check_kubeconfig") as mock_kube,
    ):
      mock_kube.return_value = CheckResult(
        "kubeconfig context", CheckStatus.PASS, "ok"
      )
      result = self.runner.invoke(doctor, _CLI_ARGS)

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("WARN", result.output)
    self.assertIn("State directory not found", result.output)

  def test_wrong_stack_shows_available(self):
    with (
      mock.patch(f"{_MODULE}.shutil.which", side_effect=_all_tools_which),
      mock.patch(
        f"{_MODULE}.subprocess.run", side_effect=_make_run_side_effect()
      ),
      mock.patch(f"{_MODULE}.os.path.isdir", return_value=True),
      mock.patch(
        f"{_MODULE}.os.listdir",
        return_value=["other-project-other-cluster.json"],
      ),
      mock.patch(f"{_MODULE}._check_kubeconfig") as mock_kube,
    ):
      mock_kube.return_value = CheckResult(
        "kubeconfig context", CheckStatus.PASS, "ok"
      )
      result = self.runner.invoke(doctor, _CLI_ARGS)

    self.assertIn("Available:", result.output)
    self.assertIn("other-project-other-cluster", result.output)


class DoctorGkeClusterNotRunningTest(absltest.TestCase):
  """GKE cluster not running — FAIL with status-specific message."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()

  def test_cluster_not_found(self):
    with (
      mock.patch(f"{_MODULE}.shutil.which", side_effect=_all_tools_which),
      mock.patch(
        f"{_MODULE}.subprocess.run",
        side_effect=_make_run_side_effect(cluster_status=None),
      ),
      mock.patch(f"{_MODULE}.os.path.isdir", return_value=True),
      mock.patch(
        f"{_MODULE}.os.listdir", return_value=["test-project-test-cluster.json"]
      ),
    ):
      result = self.runner.invoke(doctor, _CLI_ARGS)

    self.assertEqual(result.exit_code, 1, result.output)
    self.assertIn("Cluster not found", result.output)

  def test_cluster_provisioning(self):
    with (
      mock.patch(f"{_MODULE}.shutil.which", side_effect=_all_tools_which),
      mock.patch(
        f"{_MODULE}.subprocess.run",
        side_effect=_make_run_side_effect(cluster_status="PROVISIONING"),
      ),
      mock.patch(f"{_MODULE}.os.path.isdir", return_value=True),
      mock.patch(
        f"{_MODULE}.os.listdir", return_value=["test-project-test-cluster.json"]
      ),
      mock.patch(f"{_MODULE}._check_kubeconfig") as mock_kube,
    ):
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
      {"name": "default-pool", "status": "RUNNING", "config": {}},
      {
        "name": "gpu-pool",
        "status": "ERROR",
        "config": {"accelerators": [{"type": "nvidia-tesla-t4"}]},
      },
    ]
    with (
      mock.patch(f"{_MODULE}.shutil.which", side_effect=_all_tools_which),
      mock.patch(
        f"{_MODULE}.subprocess.run",
        side_effect=_make_run_side_effect(node_pools=pools),
      ),
      mock.patch(f"{_MODULE}.os.path.isdir", return_value=True),
      mock.patch(
        f"{_MODULE}.os.listdir", return_value=["test-project-test-cluster.json"]
      ),
      mock.patch(f"{_MODULE}._check_kubeconfig") as mock_kube,
    ):
      mock_kube.return_value = CheckResult(
        "kubeconfig context", CheckStatus.PASS, "ok"
      )
      result = self.runner.invoke(doctor, _CLI_ARGS)

    self.assertIn("gpu-pool", result.output)
    self.assertIn("ERROR", result.output)
    self.assertIn("WARN", result.output)


class DoctorKubeconfigMismatchTest(absltest.TestCase):
  """kubeconfig mismatch — WARN."""

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()

  def test_wrong_context(self):
    with (
      mock.patch(f"{_MODULE}.shutil.which", side_effect=_all_tools_which),
      mock.patch(
        f"{_MODULE}.subprocess.run", side_effect=_make_run_side_effect()
      ),
      mock.patch(f"{_MODULE}.os.path.isdir", return_value=True),
      mock.patch(
        f"{_MODULE}.os.listdir", return_value=["test-project-test-cluster.json"]
      ),
      mock.patch(f"{_MODULE}._check_kubeconfig") as mock_kube,
    ):
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
    with (
      mock.patch(f"{_MODULE}.shutil.which", side_effect=_all_tools_which),
      mock.patch(
        f"{_MODULE}.subprocess.run",
        side_effect=_make_run_side_effect(lws_ok=False),
      ),
      mock.patch(f"{_MODULE}.os.path.isdir", return_value=True),
      mock.patch(
        f"{_MODULE}.os.listdir", return_value=["test-project-test-cluster.json"]
      ),
      mock.patch(f"{_MODULE}._check_kubeconfig") as mock_kube,
    ):
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
    with mock.patch(f"{_MODULE}.shutil.which", side_effect=_mock_which(set())):
      result = self.runner.invoke(doctor, _CLI_ARGS)

    self.assertEqual(result.exit_code, 1, result.output)

  def test_exit_0_on_warn_only(self):
    """Only WARN/SKIP → exit 0."""
    with (
      mock.patch(f"{_MODULE}.shutil.which", side_effect=_all_tools_which),
      mock.patch(
        f"{_MODULE}.subprocess.run", side_effect=_make_run_side_effect()
      ),
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
