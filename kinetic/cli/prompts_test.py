"""Tests for kinetic.cli.prompts."""

import json
from unittest import mock

from absl.testing import absltest

from kinetic.cli import prompts


class TestPrompts(absltest.TestCase):
  @mock.patch("kinetic.cli.prompts.subprocess.run")
  def test_project_exists_args(self, mock_run):
    mock_run.return_value.returncode = 0
    prompts._project_exists("my-proj")
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    self.assertIn("--", args)
    self.assertIn("my-proj", args)
    idx_delim = args.index("--")
    idx_project = args.index("my-proj")
    self.assertLess(idx_delim, idx_project)

  @mock.patch("kinetic.cli.prompts.subprocess.run")
  def test_create_project_args(self, mock_run):
    mock_run.return_value.returncode = 0
    prompts._create_project("my-proj")
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    self.assertIn("--", args)
    self.assertIn("my-proj", args)
    idx_delim = args.index("--")
    idx_project = args.index("my-proj")
    self.assertLess(idx_delim, idx_project)

  @mock.patch("kinetic.cli.prompts.subprocess.run")
  @mock.patch("click.confirm", return_value=True)
  def test_link_billing_account_args(self, mock_confirm, mock_run):
    # Mock first call (list billing accounts)
    mock_acct = {
      "name": "billingAccounts/123",
      "displayName": "My Billing Account",
    }
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = json.dumps([mock_acct])

    # _link_billing_account calls subprocess.run twice (list and link).

    prompts._link_billing_account("my-proj")

    self.assertEqual(mock_run.call_count, 2)

    # Check second call args
    args = mock_run.call_args_list[1][0][0]
    self.assertIn("--", args)
    self.assertIn("my-proj", args)
    idx_delim = args.index("--")
    idx_project = args.index("my-proj")
    self.assertLess(idx_delim, idx_project)


if __name__ == "__main__":
  absltest.main()
