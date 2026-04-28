"""Tests for kinetic.cli.infra.stack_manager — backend URL routing."""

from unittest import mock

from absl.testing import absltest

from kinetic.cli.config import InfraConfig
from kinetic.cli.infra import stack_manager


class GetStackBackendRoutingTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    # Stub out everything Pulumi-related so the test only exercises the
    # backend URL branching logic.
    self.mock_makedirs = self.enterContext(
      mock.patch.object(stack_manager.os, "makedirs")
    )
    self.mock_pulumi_cmd = self.enterContext(
      mock.patch.object(stack_manager.auto, "PulumiCommand")
    )
    self.mock_project_settings = self.enterContext(
      mock.patch.object(stack_manager.auto, "ProjectSettings")
    )
    self.mock_project_backend = self.enterContext(
      mock.patch.object(stack_manager.auto, "ProjectBackend")
    )
    self.mock_create_or_select_stack = self.enterContext(
      mock.patch.object(stack_manager.auto, "create_or_select_stack")
    )
    self.mock_local_workspace = self.enterContext(
      mock.patch.object(stack_manager.auto, "LocalWorkspaceOptions")
    )
    self.mock_ensure_gcs = self.enterContext(
      mock.patch.object(stack_manager, "ensure_gcs_backend")
    )

  def test_local_default_does_not_call_ensure_gcs(self):
    config = InfraConfig(project="p", zone="z", cluster_name="c")

    stack_manager.get_stack(lambda: None, config)

    self.mock_ensure_gcs.assert_not_called()
    self.mock_makedirs.assert_called_once()
    backend_url_kwarg = self.mock_project_backend.call_args.kwargs["url"]
    self.assertTrue(backend_url_kwarg.startswith("file://"))

  def test_explicit_file_url(self):
    config = InfraConfig(
      project="p",
      zone="z",
      cluster_name="c",
      state_backend_url="file:///tmp/custom",
    )

    stack_manager.get_stack(lambda: None, config)

    self.mock_ensure_gcs.assert_not_called()
    self.assertEqual(
      self.mock_project_backend.call_args.kwargs["url"], "file:///tmp/custom"
    )

  def test_gcs_url_calls_ensure_gcs_with_project(self):
    config = InfraConfig(
      project="kinetic-proj",
      zone="z",
      cluster_name="c",
      state_backend_url="gs://team-state",
    )

    stack_manager.get_stack(lambda: None, config)

    # Bucket is created under the kinetic project — not whatever ADC
    # happens to default to.
    self.mock_ensure_gcs.assert_called_once_with(
      "gs://team-state", project="kinetic-proj"
    )
    self.mock_makedirs.assert_not_called()
    self.assertEqual(
      self.mock_project_backend.call_args.kwargs["url"], "gs://team-state"
    )


if __name__ == "__main__":
  absltest.main()
