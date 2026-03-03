"""Tests for keras_remote.cli.prerequisites_check — tool availability checks."""

from unittest import mock

import click
from absl.testing import absltest

from keras_remote.cli.prerequisites_check import (
  check_gcloud,
  check_gcloud_auth,
  check_gke_auth_plugin,
  check_kubectl,
)

_MODULE = "keras_remote.cli.prerequisites_check"


class TestToolChecks(absltest.TestCase):
  """Tests for CLI-only tool checks (kubectl)."""

  def test_kubectl_present(self):
    with mock.patch("shutil.which", return_value="/usr/bin/kubectl"):
      check_kubectl()

  def test_kubectl_missing(self):
    with (
      mock.patch("shutil.which", return_value=None),
      self.assertRaisesRegex(click.ClickException, "kubectl not found"),
    ):
      check_kubectl()


class TestDelegatedChecks(absltest.TestCase):
  """Tests for checks that delegate to keras_remote.credentials."""

  def test_check_gcloud_delegates(self):
    with mock.patch(f"{_MODULE}.credentials.ensure_gcloud"):
      check_gcloud()

  def test_check_gcloud_converts_error(self):
    with (
      mock.patch(
        f"{_MODULE}.credentials.ensure_gcloud",
        side_effect=RuntimeError("gcloud CLI not found"),
      ),
      self.assertRaisesRegex(click.ClickException, "gcloud CLI not found"),
    ):
      check_gcloud()

  def test_check_gke_auth_plugin_delegates(self):
    with mock.patch(f"{_MODULE}.credentials.ensure_gke_auth_plugin"):
      check_gke_auth_plugin()

  def test_check_gke_auth_plugin_converts_error(self):
    with (
      mock.patch(
        f"{_MODULE}.credentials.ensure_gke_auth_plugin",
        side_effect=RuntimeError("Failed to install gke-gcloud-auth-plugin"),
      ),
      self.assertRaisesRegex(
        click.ClickException,
        "Failed to install gke-gcloud-auth-plugin",
      ),
    ):
      check_gke_auth_plugin()

  def test_check_gcloud_auth_delegates(self):
    with mock.patch(f"{_MODULE}.credentials.ensure_adc"):
      check_gcloud_auth()

  def test_check_gcloud_auth_converts_error(self):
    with (
      mock.patch(
        f"{_MODULE}.credentials.ensure_adc",
        side_effect=RuntimeError("Failed to configure Application Default"),
      ),
      self.assertRaisesRegex(
        click.ClickException,
        "Failed to configure Application Default",
      ),
    ):
      check_gcloud_auth()


if __name__ == "__main__":
  absltest.main()
