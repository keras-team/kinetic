"""Tests for kinetic.credentials — shared credential checks."""

import subprocess
from unittest import mock

from absl.testing import absltest
from kubernetes.config import ConfigException

from kinetic import credentials

_MODULE = "kinetic.credentials"


class TestEnsureGcloud(absltest.TestCase):
  def test_missing(self):
    with (
      mock.patch("shutil.which", return_value=None),
      self.assertRaisesRegex(RuntimeError, "gcloud CLI not found"),
    ):
      credentials.ensure_gcloud()


class TestEnsureGkeAuthPlugin(absltest.TestCase):
  def test_missing_install_succeeds(self):
    with (
      mock.patch("shutil.which", return_value=None),
      mock.patch(f"{_MODULE}.subprocess.run") as mock_run,
    ):
      credentials.ensure_gke_auth_plugin()
      mock_run.assert_called_once()
      args = mock_run.call_args[0][0]
      self.assertIn("gke-gcloud-auth-plugin", args)
      self.assertIn("--quiet", args)

  def test_missing_install_fails(self):
    with (
      mock.patch("shutil.which", return_value=None),
      mock.patch(
        f"{_MODULE}.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "gcloud"),
      ),
      self.assertRaisesRegex(
        RuntimeError, "Failed to install gke-gcloud-auth-plugin"
      ),
    ):
      credentials.ensure_gke_auth_plugin()


class TestEnsureAdc(absltest.TestCase):
  def test_token_success(self):
    with mock.patch(f"{_MODULE}.subprocess.run") as mock_run:
      mock_run.return_value.returncode = 0
      credentials.ensure_adc()
      self.assertEqual(mock_run.call_count, 1)

  def test_token_failure_triggers_login(self):
    with mock.patch(f"{_MODULE}.subprocess.run") as mock_run:
      token_result = mock.MagicMock()
      token_result.returncode = 1
      # First call returns failure, second call (login) succeeds.
      mock_run.side_effect = [token_result, mock.DEFAULT]

      credentials.ensure_adc()

      self.assertEqual(mock_run.call_count, 2)
      login_call = mock_run.call_args_list[1]
      self.assertIn("login", login_call[0][0])

  def test_login_failure_raises(self):
    with mock.patch(f"{_MODULE}.subprocess.run") as mock_run:
      token_result = mock.MagicMock()
      token_result.returncode = 1
      mock_run.side_effect = [
        token_result,
        subprocess.CalledProcessError(1, "gcloud"),
      ]

      with self.assertRaisesRegex(
        RuntimeError, "Failed to configure Application Default"
      ):
        credentials.ensure_adc()


class TestEnsureKubeconfig(absltest.TestCase):
  def _mock_active_context(self, cluster_name):
    """Return mock values for list_kube_config_contexts."""
    ctx = {"context": {"cluster": cluster_name}, "name": cluster_name}
    return [ctx], ctx

  def test_correct_cluster_context(self):
    """When the active context matches the expected cluster, no reconfigure."""
    with (
      mock.patch(f"{_MODULE}.config.load_kube_config"),
      mock.patch(
        f"{_MODULE}.config.list_kube_config_contexts",
        return_value=self._mock_active_context(
          "gke_my-proj_us-central1-a_my-cluster"
        ),
      ),
      mock.patch(f"{_MODULE}._configure_kubeconfig") as mock_configure,
    ):
      credentials.ensure_kubeconfig("my-proj", "us-central1-a", "my-cluster")
      mock_configure.assert_not_called()

  def test_wrong_cluster_context_triggers_reconfigure(self):
    """When the active context doesn't match, reconfigure."""
    with (
      mock.patch(f"{_MODULE}.config.load_kube_config"),
      mock.patch(
        f"{_MODULE}.config.list_kube_config_contexts",
        return_value=self._mock_active_context(
          "gke_other-proj_us-west1-b_other-cluster"
        ),
      ),
      mock.patch(f"{_MODULE}._configure_kubeconfig") as mock_configure,
    ):
      credentials.ensure_kubeconfig("my-proj", "us-central1-a", "my-cluster")
      mock_configure.assert_called_once_with(
        "my-cluster", "us-central1-a", "my-proj"
      )

  def test_no_kubeconfig_triggers_configure(self):
    """When no kubeconfig exists, configure from scratch."""
    with (
      mock.patch(
        f"{_MODULE}.config.load_kube_config",
        side_effect=ConfigException("no config"),
      ),
      mock.patch(f"{_MODULE}._configure_kubeconfig") as mock_configure,
    ):
      credentials.ensure_kubeconfig("my-proj", "us-central1-a", "my-cluster")
      mock_configure.assert_called_once_with(
        "my-cluster", "us-central1-a", "my-proj"
      )

  def test_configure_failure_raises(self):
    with (
      mock.patch(
        f"{_MODULE}.config.load_kube_config",
        side_effect=ConfigException("no config"),
      ),
      mock.patch(
        f"{_MODULE}.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "gcloud"),
      ),
      self.assertRaisesRegex(RuntimeError, "Failed to configure kubeconfig"),
    ):
      credentials.ensure_kubeconfig("my-proj", "us-central1-a", "my-cluster")


if __name__ == "__main__":
  absltest.main()
