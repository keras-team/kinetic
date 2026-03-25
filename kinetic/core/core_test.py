"""Tests for kinetic.core.core — run/submit decorators and env var capture."""

import os
from unittest import mock
from unittest.mock import MagicMock

from absl.testing import absltest

from kinetic.core.core import run


class TestEnvVarCapture(absltest.TestCase):
  def test_exact_match(self):
    env = {**os.environ, "MY_VAR": "my_val"}
    mock_handle = MagicMock()
    mock_handle.result.return_value = None
    with (
      mock.patch.dict(os.environ, {"MY_VAR": "my_val"}),
      mock.patch("kinetic.core.core.submit_remote", return_value=mock_handle),
      mock.patch(
        "kinetic.core.core.JobContext.from_params", return_value=MagicMock()
      ) as mock_from_params,
    ):

      @run(accelerator="cpu", capture_env_vars=["MY_VAR"])
      def func():
        pass

      func()
      env_vars = mock_from_params.call_args[0][7]
      self.assertEqual(env_vars, {"MY_VAR": "my_val"})

  def test_wildcard_pattern(self):
    env = {
      k: v
      for k, v in os.environ.items()
      if k not in ("PREFIX_A", "PREFIX_B", "OTHER")
    }
    env.update({"PREFIX_A": "1", "PREFIX_B": "2", "OTHER": "3"})
    mock_handle = MagicMock()
    mock_handle.result.return_value = None
    with (
      mock.patch.dict(os.environ, env, clear=True),
      mock.patch("kinetic.core.core.submit_remote", return_value=mock_handle),
      mock.patch(
        "kinetic.core.core.JobContext.from_params", return_value=MagicMock()
      ) as mock_from_params,
    ):

      @run(accelerator="cpu", capture_env_vars=["PREFIX_*"])
      def func():
        pass

      func()
      env_vars = mock_from_params.call_args[0][7]
      self.assertIn("PREFIX_A", env_vars)
      self.assertIn("PREFIX_B", env_vars)
      self.assertNotIn("OTHER", env_vars)

  def test_missing_var_skipped(self):
    env = {k: v for k, v in os.environ.items() if k != "NONEXISTENT"}
    mock_handle = MagicMock()
    mock_handle.result.return_value = None
    with (
      mock.patch.dict(os.environ, env, clear=True),
      mock.patch("kinetic.core.core.submit_remote", return_value=mock_handle),
      mock.patch(
        "kinetic.core.core.JobContext.from_params", return_value=MagicMock()
      ) as mock_from_params,
    ):

      @run(accelerator="cpu", capture_env_vars=["NONEXISTENT"])
      def func():
        pass

      func()
      env_vars = mock_from_params.call_args[0][7]
      self.assertEqual(env_vars, {})

  def test_mixed_exact_and_wildcard(self):
    env = {
      k: v
      for k, v in os.environ.items()
      if k not in ("EXACT_VAR", "WILD_A", "WILD_B")
    }
    env.update({"EXACT_VAR": "exact", "WILD_A": "a", "WILD_B": "b"})
    mock_handle = MagicMock()
    mock_handle.result.return_value = None
    with (
      mock.patch.dict(os.environ, env, clear=True),
      mock.patch("kinetic.core.core.submit_remote", return_value=mock_handle),
      mock.patch(
        "kinetic.core.core.JobContext.from_params", return_value=MagicMock()
      ) as mock_from_params,
    ):

      @run(
        accelerator="cpu",
        capture_env_vars=["EXACT_VAR", "WILD_*"],
      )
      def func():
        pass

      func()
      env_vars = mock_from_params.call_args[0][7]
      self.assertEqual(
        env_vars, {"EXACT_VAR": "exact", "WILD_A": "a", "WILD_B": "b"}
      )


class TestExecuteOnBackendDefaults(absltest.TestCase):
  def test_cluster_from_env(self):
    """When cluster=None, falls back to KINETIC_CLUSTER env var."""
    mock_handle = MagicMock()
    mock_handle.result.return_value = 42
    with (
      mock.patch.dict(
        os.environ,
        {
          "KINETIC_CLUSTER": "env-cluster",
          "KINETIC_PROJECT": "proj",
        },
      ),
      mock.patch(
        "kinetic.core.core.submit_remote",
        return_value=mock_handle,
      ) as mock_submit,
      mock.patch(
        "kinetic.core.core.JobContext.from_params",
        return_value=MagicMock(),
      ),
    ):

      @run(accelerator="cpu", cluster=None)
      def func():
        pass

      func()

      backend = mock_submit.call_args[0][1]
      self.assertEqual(backend.cluster, "env-cluster")

  def test_namespace_from_env(self):
    """When namespace=None, falls back to KINETIC_NAMESPACE env var."""
    mock_handle = MagicMock()
    mock_handle.result.return_value = 42
    with (
      mock.patch.dict(
        os.environ,
        {
          "KINETIC_NAMESPACE": "custom-ns",
          "KINETIC_PROJECT": "proj",
        },
      ),
      mock.patch(
        "kinetic.core.core.submit_remote",
        return_value=mock_handle,
      ) as mock_submit,
      mock.patch(
        "kinetic.core.core.JobContext.from_params",
        return_value=MagicMock(),
      ),
    ):

      @run(accelerator="cpu", namespace=None)
      def func():
        pass

      func()

      backend = mock_submit.call_args[0][1]
      self.assertEqual(backend.namespace, "custom-ns")


class TestSubmitOnBackend(absltest.TestCase):
  def test_run_calls_result_on_handle(self):
    """run() is submit() + result() — calls .result() on the returned handle."""
    mock_handle = MagicMock()
    mock_handle.result.return_value = 123
    with (
      mock.patch.dict(os.environ, {"KINETIC_PROJECT": "proj"}),
      mock.patch(
        "kinetic.core.core.submit_remote",
        return_value=mock_handle,
      ),
      mock.patch(
        "kinetic.core.core.JobContext.from_params",
        return_value=MagicMock(),
      ),
    ):

      @run(accelerator="cpu")
      def func():
        pass

      result = func()

      self.assertEqual(result, 123)
      mock_handle.result.assert_called_once_with()


if __name__ == "__main__":
  absltest.main()
