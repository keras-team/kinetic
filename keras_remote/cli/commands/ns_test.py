"""Tests for keras_remote.cli.commands.ns — namespace CLI commands."""

from unittest import mock

import click
from absl.testing import absltest, parameterized

from keras_remote.cli.commands.ns import validate_namespace_name


class TestValidateNamespaceName(parameterized.TestCase):
  @parameterized.named_parameters(
    dict(testcase_name="simple", name="team-nlp"),
    dict(testcase_name="short", name="a"),
    dict(testcase_name="digits", name="team1"),
    dict(testcase_name="max_length", name="a" * 27),
    dict(testcase_name="mixed", name="abc-123-xyz"),
  )
  def test_valid_names(self, name):
    validate_namespace_name(name)  # should not raise

  @parameterized.named_parameters(
    dict(testcase_name="too_long", name="a" * 28, match="at most 27"),
    dict(testcase_name="uppercase", name="Team", match="lowercase"),
    dict(testcase_name="underscore", name="team_nlp", match="lowercase"),
    dict(
      testcase_name="starts_with_hyphen", name="-team", match="start and end"
    ),
    dict(testcase_name="ends_with_hyphen", name="team-", match="start and end"),
    dict(testcase_name="space", name="team nlp", match="lowercase"),
  )
  def test_invalid_names(self, name, match):
    with self.assertRaisesRegex(click.BadParameter, match):
      validate_namespace_name(name)

  @parameterized.named_parameters(
    dict(testcase_name="default", name="default"),
    dict(testcase_name="kube_system", name="kube-system"),
    dict(testcase_name="kube_public", name="kube-public"),
    dict(testcase_name="kube_node_lease", name="kube-node-lease"),
  )
  def test_reserved_names(self, name):
    with self.assertRaisesRegex(click.BadParameter, "reserved"):
      validate_namespace_name(name)


class TestNsCreate(absltest.TestCase):
  def test_creates_namespace_config(self):
    """Verify ns create builds correct NamespaceConfig and calls update."""
    with (
      mock.patch(
        "keras_remote.cli.commands.ns._load_state",
        return_value=("proj", "us-central1-a", "cluster", [], []),
      ),
      mock.patch(
        "keras_remote.cli.commands.ns._apply_update",
        return_value=True,
      ) as mock_update,
    ):
      from click.testing import CliRunner

      from keras_remote.cli.commands.ns import ns

      runner = CliRunner()
      result = runner.invoke(
        ns,
        [
          "create",
          "team-nlp",
          "--members",
          "alice@co.com,bob@co.com",
          "--gpus",
          "8",
          "--max-jobs",
          "10",
          "-y",
        ],
      )
      self.assertEqual(result.exit_code, 0, msg=result.output)

      # Check _apply_update was called with correct namespace config
      call_args = mock_update.call_args
      namespaces = call_args[0][4]
      self.assertLen(namespaces, 1)
      self.assertEqual(namespaces[0].name, "team-nlp")
      self.assertEqual(namespaces[0].members, ["alice@co.com", "bob@co.com"])
      self.assertEqual(namespaces[0].gpus, 8)
      self.assertEqual(namespaces[0].max_jobs, 10)


class TestNsCreateIgnoreIamErrors(absltest.TestCase):
  def test_k8s_ok_iam_fails_shows_warnings(self):
    """Phase 1 (K8s) succeeds, phase 2 (IAM) fails → Created With Warnings."""
    with (
      mock.patch(
        "keras_remote.cli.commands.ns._load_state",
        return_value=("proj", "us-central1-a", "cluster", [], []),
      ),
      mock.patch(
        "keras_remote.cli.commands.ns._apply_update",
        side_effect=[True, False],  # K8s ok, IAM fails
      ) as mock_update,
    ):
      from click.testing import CliRunner

      from keras_remote.cli.commands.ns import ns

      runner = CliRunner()
      result = runner.invoke(
        ns,
        ["create", "team-nlp", "--ignore-iam-errors", "-y"],
      )
      self.assertEqual(result.exit_code, 0, msg=result.output)
      self.assertIn("Created With Warnings", result.output)

      # Phase 1 called with skip_iam=True, phase 2 without
      self.assertEqual(mock_update.call_count, 2)
      self.assertTrue(mock_update.call_args_list[0].kwargs["skip_iam"])
      self.assertFalse(
        mock_update.call_args_list[1].kwargs.get("skip_iam", False)
      )

  def test_k8s_fails_aborts(self):
    """Phase 1 (K8s) fails → Creation Failed, no phase 2."""
    with (
      mock.patch(
        "keras_remote.cli.commands.ns._load_state",
        return_value=("proj", "us-central1-a", "cluster", [], []),
      ),
      mock.patch(
        "keras_remote.cli.commands.ns._apply_update",
        return_value=False,
      ) as mock_update,
    ):
      from click.testing import CliRunner

      from keras_remote.cli.commands.ns import ns

      runner = CliRunner()
      result = runner.invoke(
        ns,
        ["create", "team-nlp", "--ignore-iam-errors", "-y"],
      )
      self.assertEqual(result.exit_code, 0, msg=result.output)
      self.assertIn("Creation Failed", result.output)
      # Only phase 1 was attempted
      self.assertEqual(mock_update.call_count, 1)

  def test_hard_failure_without_flag(self):
    """Without --ignore-iam-errors, update failure reports hard failure."""
    with (
      mock.patch(
        "keras_remote.cli.commands.ns._load_state",
        return_value=("proj", "us-central1-a", "cluster", [], []),
      ),
      mock.patch(
        "keras_remote.cli.commands.ns._apply_update",
        return_value=False,
      ),
    ):
      from click.testing import CliRunner

      from keras_remote.cli.commands.ns import ns

      runner = CliRunner()
      result = runner.invoke(
        ns,
        ["create", "team-nlp", "-y"],
      )
      self.assertEqual(result.exit_code, 0, msg=result.output)
      self.assertIn("Creation Failed", result.output)

  def test_both_phases_succeed(self):
    """Both phases succeed → normal Namespace Created."""
    with (
      mock.patch(
        "keras_remote.cli.commands.ns._load_state",
        return_value=("proj", "us-central1-a", "cluster", [], []),
      ),
      mock.patch(
        "keras_remote.cli.commands.ns._apply_update",
        return_value=True,
      ),
    ):
      from click.testing import CliRunner

      from keras_remote.cli.commands.ns import ns

      runner = CliRunner()
      result = runner.invoke(
        ns,
        ["create", "team-nlp", "--ignore-iam-errors", "-y"],
      )
      self.assertEqual(result.exit_code, 0, msg=result.output)
      self.assertIn("Namespace Created", result.output)
      self.assertNotIn("Warnings", result.output)

  def test_retry_existing_namespace_with_flag(self):
    """When namespace exists in state and --ignore-iam-errors is set, retry."""
    from keras_remote.cli.config import NamespaceConfig

    existing = [NamespaceConfig(name="team-nlp", members=["alice@co.com"])]
    with (
      mock.patch(
        "keras_remote.cli.commands.ns._load_state",
        return_value=("proj", "us-central1-a", "cluster", [], existing),
      ),
      mock.patch(
        "keras_remote.cli.commands.ns._apply_update",
        return_value=True,
      ) as mock_update,
    ):
      from click.testing import CliRunner

      from keras_remote.cli.commands.ns import ns

      runner = CliRunner()
      result = runner.invoke(
        ns,
        ["create", "team-nlp", "--ignore-iam-errors", "-y"],
      )
      self.assertEqual(result.exit_code, 0, msg=result.output)
      self.assertIn("Namespace Created", result.output)
      self.assertIn("retrying", result.output.lower())

      # Should reuse existing config, not add a duplicate
      namespaces = mock_update.call_args[0][4]
      self.assertLen(namespaces, 1)
      self.assertEqual(namespaces[0].name, "team-nlp")

  def test_reject_existing_namespace_without_flag(self):
    """When namespace exists in state without --ignore-iam-errors, reject."""
    from keras_remote.cli.config import NamespaceConfig

    existing = [NamespaceConfig(name="team-nlp")]
    with mock.patch(
      "keras_remote.cli.commands.ns._load_state",
      return_value=("proj", "us-central1-a", "cluster", [], existing),
    ):
      from click.testing import CliRunner

      from keras_remote.cli.commands.ns import ns

      runner = CliRunner()
      result = runner.invoke(ns, ["create", "team-nlp", "-y"])
      self.assertNotEqual(result.exit_code, 0)
      self.assertIn("already exists", result.output)


class TestNsDelete(absltest.TestCase):
  def test_removes_namespace(self):
    """Verify ns delete removes the namespace from config."""
    from keras_remote.cli.config import NamespaceConfig

    existing = [
      NamespaceConfig(name="team-nlp"),
      NamespaceConfig(name="team-cv"),
    ]
    with (
      mock.patch(
        "keras_remote.cli.commands.ns._load_state",
        return_value=("proj", "us-central1-a", "cluster", [], existing),
      ),
      mock.patch(
        "keras_remote.cli.commands.ns._apply_update",
        return_value=True,
      ) as mock_update,
    ):
      from click.testing import CliRunner

      from keras_remote.cli.commands.ns import ns

      runner = CliRunner()
      result = runner.invoke(ns, ["delete", "team-nlp", "-y"])
      self.assertEqual(result.exit_code, 0, msg=result.output)

      namespaces = mock_update.call_args[0][4]
      self.assertLen(namespaces, 1)
      self.assertEqual(namespaces[0].name, "team-cv")


class TestNsAddMember(absltest.TestCase):
  def test_adds_member(self):
    """Verify add-member appends to namespace members."""
    from keras_remote.cli.config import NamespaceConfig

    existing = [NamespaceConfig(name="team-nlp", members=["alice@co.com"])]
    with (
      mock.patch(
        "keras_remote.cli.commands.ns._load_state",
        return_value=("proj", "us-central1-a", "cluster", [], existing),
      ),
      mock.patch(
        "keras_remote.cli.commands.ns._apply_update",
        return_value=True,
      ) as mock_update,
    ):
      from click.testing import CliRunner

      from keras_remote.cli.commands.ns import ns

      runner = CliRunner()
      result = runner.invoke(
        ns,
        ["add-member", "team-nlp", "--member", "bob@co.com", "-y"],
      )
      self.assertEqual(result.exit_code, 0, msg=result.output)

      namespaces = mock_update.call_args[0][4]
      self.assertIn("bob@co.com", namespaces[0].members)
      self.assertIn("alice@co.com", namespaces[0].members)


class TestNsRemoveMember(absltest.TestCase):
  def test_removes_member(self):
    """Verify remove-member removes from namespace members."""
    from keras_remote.cli.config import NamespaceConfig

    existing = [
      NamespaceConfig(name="team-nlp", members=["alice@co.com", "bob@co.com"])
    ]
    with (
      mock.patch(
        "keras_remote.cli.commands.ns._load_state",
        return_value=("proj", "us-central1-a", "cluster", [], existing),
      ),
      mock.patch(
        "keras_remote.cli.commands.ns._apply_update",
        return_value=True,
      ) as mock_update,
    ):
      from click.testing import CliRunner

      from keras_remote.cli.commands.ns import ns

      runner = CliRunner()
      result = runner.invoke(
        ns,
        ["remove-member", "team-nlp", "--member", "alice@co.com", "-y"],
      )
      self.assertEqual(result.exit_code, 0, msg=result.output)

      namespaces = mock_update.call_args[0][4]
      self.assertEqual(namespaces[0].members, ["bob@co.com"])


if __name__ == "__main__":
  absltest.main()
