"""Tests for kinetic.credentials — shared credential checks."""

import subprocess
from unittest import mock

import google.auth.exceptions
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
  def test_adc_found_and_refreshable(self):
    """google.auth.default() succeeds and refresh works — no subprocess."""
    mock_creds = mock.MagicMock()
    with (
      mock.patch(
        f"{_MODULE}.google.auth.default", return_value=(mock_creds, "p")
      ),
      mock.patch(f"{_MODULE}.subprocess.run") as mock_run,
    ):
      credentials.ensure_adc()
      mock_creds.refresh.assert_called_once()
      mock_run.assert_not_called()

  def test_adc_not_found_gcloud_available(self):
    """No ADC, gcloud present — falls back to interactive login."""
    with (
      mock.patch(
        f"{_MODULE}.google.auth.default",
        side_effect=google.auth.exceptions.DefaultCredentialsError("none"),
      ),
      mock.patch("shutil.which", return_value="/usr/bin/gcloud"),
      mock.patch(f"{_MODULE}.subprocess.run") as mock_run,
    ):
      credentials.ensure_adc()
      mock_run.assert_called_once()
      self.assertIn("login", mock_run.call_args[0][0])

  def test_adc_not_found_no_gcloud(self):
    """No ADC, no gcloud — raises with GOOGLE_APPLICATION_CREDENTIALS hint."""
    with (
      mock.patch(
        f"{_MODULE}.google.auth.default",
        side_effect=google.auth.exceptions.DefaultCredentialsError("none"),
      ),
      mock.patch("shutil.which", return_value=None),
      self.assertRaisesRegex(RuntimeError, "GOOGLE_APPLICATION_CREDENTIALS"),
    ):
      credentials.ensure_adc()

  def test_adc_found_refresh_fails_gcloud_login(self):
    """ADC found but refresh fails, gcloud present — falls back to login."""
    mock_creds = mock.MagicMock()
    mock_creds.refresh.side_effect = google.auth.exceptions.RefreshError(
      "expired"
    )
    with (
      mock.patch(
        f"{_MODULE}.google.auth.default", return_value=(mock_creds, "p")
      ),
      mock.patch("shutil.which", return_value="/usr/bin/gcloud"),
      mock.patch(f"{_MODULE}.subprocess.run") as mock_run,
    ):
      credentials.ensure_adc()
      mock_run.assert_called_once()
      self.assertIn("login", mock_run.call_args[0][0])

  def test_adc_found_refresh_fails_no_gcloud(self):
    """ADC found but refresh fails, no gcloud — raises."""
    mock_creds = mock.MagicMock()
    mock_creds.refresh.side_effect = google.auth.exceptions.RefreshError(
      "expired"
    )
    with (
      mock.patch(
        f"{_MODULE}.google.auth.default", return_value=(mock_creds, "p")
      ),
      mock.patch("shutil.which", return_value=None),
      self.assertRaisesRegex(RuntimeError, "GOOGLE_APPLICATION_CREDENTIALS"),
    ):
      credentials.ensure_adc()

  def test_login_failure_raises(self):
    """No ADC, gcloud present but login fails — raises."""
    with (
      mock.patch(
        f"{_MODULE}.google.auth.default",
        side_effect=google.auth.exceptions.DefaultCredentialsError("none"),
      ),
      mock.patch("shutil.which", return_value="/usr/bin/gcloud"),
      mock.patch(
        f"{_MODULE}.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "gcloud"),
      ),
      self.assertRaisesRegex(
        RuntimeError, "Failed to configure Application Default"
      ),
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
