"""Tests for kinetic.core.core — run decorator and env var capture."""

import os
from unittest import mock
from unittest.mock import MagicMock

from absl.testing import absltest

from kinetic.core.core import run


class TestRunDecorator(absltest.TestCase):
  def test_decorator_preserves_wrapped_function(self):
    @run(accelerator="cpu")
    def my_func():
      """My docstring."""

    self.assertTrue(callable(my_func))
    self.assertEqual(my_func.__name__, "my_func")
    self.assertEqual(my_func.__doc__, "My docstring.")


class TestEnvVarCapture(absltest.TestCase):
  def test_exact_match(self):
    with (
      mock.patch.dict(os.environ, {"MY_VAR": "my_val"}),
      mock.patch("kinetic.core.core._execute_on_gke") as mock_exec,
    ):

      @run(accelerator="cpu", capture_env_vars=["MY_VAR"])
      def func():
        pass

      func()
      call_args = mock_exec.call_args
      env_vars = call_args[0][-3]  # env_vars arg (before volumes and spot)
      self.assertEqual(env_vars, {"MY_VAR": "my_val"})

  def test_wildcard_pattern(self):
    env = {
      k: v
      for k, v in os.environ.items()
      if k not in ("PREFIX_A", "PREFIX_B", "OTHER")
    }
    env.update({"PREFIX_A": "1", "PREFIX_B": "2", "OTHER": "3"})
    with (
      mock.patch.dict(os.environ, env, clear=True),
      mock.patch("kinetic.core.core._execute_on_gke") as mock_exec,
    ):

      @run(accelerator="cpu", capture_env_vars=["PREFIX_*"])
      def func():
        pass

      func()
      env_vars = mock_exec.call_args[0][-3]
      self.assertIn("PREFIX_A", env_vars)
      self.assertIn("PREFIX_B", env_vars)
      self.assertNotIn("OTHER", env_vars)

  def test_missing_var_skipped(self):
    env = {k: v for k, v in os.environ.items() if k != "NONEXISTENT"}
    with (
      mock.patch.dict(os.environ, env, clear=True),
      mock.patch("kinetic.core.core._execute_on_gke") as mock_exec,
    ):

      @run(accelerator="cpu", capture_env_vars=["NONEXISTENT"])
      def func():
        pass

      func()
      env_vars = mock_exec.call_args[0][-3]
      self.assertEqual(env_vars, {})

  def test_none_capture(self):
    with mock.patch("kinetic.core.core._execute_on_gke") as mock_exec:

      @run(accelerator="cpu", capture_env_vars=None)
      def func():
        pass

      func()
      env_vars = mock_exec.call_args[0][-3]
      self.assertEqual(env_vars, {})

  def test_mixed_exact_and_wildcard(self):
    env = {
      k: v
      for k, v in os.environ.items()
      if k not in ("EXACT_VAR", "WILD_A", "WILD_B")
    }
    env.update({"EXACT_VAR": "exact", "WILD_A": "a", "WILD_B": "b"})
    with (
      mock.patch.dict(os.environ, env, clear=True),
      mock.patch("kinetic.core.core._execute_on_gke") as mock_exec,
    ):

      @run(
        accelerator="cpu",
        capture_env_vars=["EXACT_VAR", "WILD_*"],
      )
      def func():
        pass

      func()
      env_vars = mock_exec.call_args[0][-3]
      self.assertEqual(
        env_vars, {"EXACT_VAR": "exact", "WILD_A": "a", "WILD_B": "b"}
      )


class TestExecuteOnGkeDefaults(absltest.TestCase):
  def test_cluster_from_env(self):
    """When cluster=None, falls back to KINETIC_CLUSTER env var."""
    with (
      mock.patch.dict(
        os.environ,
        {
          "KINETIC_CLUSTER": "env-cluster",
          "KINETIC_PROJECT": "proj",
        },
      ),
      mock.patch(
        "kinetic.core.core.execute_remote",
        return_value=42,
      ) as mock_exec,
      mock.patch(
        "kinetic.core.core.JobContext.from_params",
        return_value=MagicMock(),
      ),
    ):

      @run(accelerator="cpu", cluster=None)
      def func():
        pass

      func()

      call_args = mock_exec.call_args
      backend = call_args[0][1]
      self.assertEqual(backend.cluster, "env-cluster")

  def test_namespace_from_env(self):
    """When namespace=None, falls back to KINETIC_NAMESPACE env var."""
    with (
      mock.patch.dict(
        os.environ,
        {
          "KINETIC_NAMESPACE": "custom-ns",
          "KINETIC_PROJECT": "proj",
        },
      ),
      mock.patch(
        "kinetic.core.core.execute_remote",
        return_value=42,
      ) as mock_exec,
      mock.patch(
        "kinetic.core.core.JobContext.from_params",
        return_value=MagicMock(),
      ),
    ):

      @run(accelerator="cpu", namespace=None)
      def func():
        pass

      func()

      backend = mock_exec.call_args[0][1]
      self.assertEqual(backend.namespace, "custom-ns")


if __name__ == "__main__":
  absltest.main()
