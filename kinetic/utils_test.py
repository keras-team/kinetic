"""Shared test utilities for colocated unit tests."""

from unittest import mock


def create_mock_storage_client():
  """Create a mock google.cloud.storage.Client."""
  mock_client = mock.MagicMock()
  patcher = mock.patch("google.cloud.storage.Client", return_value=mock_client)
  return mock_client, patcher


def create_mock_kube_config():
  """Create a mock kubernetes config loading patcher."""
  return mock.patch("kinetic.backend.gke_client._load_kube_config")


def create_mock_batch_v1():
  """Create a mock kubernetes BatchV1Api."""
  mock_api = mock.MagicMock()
  patcher = mock.patch("kubernetes.client.BatchV1Api", return_value=mock_api)
  return mock_api, patcher
