"""Tests for kinetic.cli.infra.post_deploy."""

from unittest import mock

from absl.testing import absltest

from kinetic.cli.infra import post_deploy


class TestPostDeploy(absltest.TestCase):
  @mock.patch("kinetic.cli.infra.post_deploy.subprocess.run")
  @mock.patch("kinetic.cli.infra.post_deploy.invalidate_credential_cache")
  def test_configure_kubectl_args(self, mock_invalidate, mock_run):
    post_deploy.configure_kubectl("my-cluster", "us-central1-a", "my-proj")
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    self.assertIn("--", args)
    idx_get = args.index("get-credentials")
    idx_delim = args.index("--")
    idx_cluster = args.index("my-cluster")
    self.assertLess(idx_get, idx_delim)
    self.assertLess(idx_delim, idx_cluster)


if __name__ == "__main__":
  absltest.main()
