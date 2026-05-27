"""Tests for kinetic.core.core — run/submit decorators and env var capture."""

import json
import os
import tempfile
from unittest import mock
from unittest.mock import MagicMock

from absl.testing import absltest

from kinetic.cli.profiles import resolve_infra
from kinetic.constants import DEFAULT_CLUSTER_NAME, DEFAULT_ZONE
from kinetic.core.core import run


def _isolate_profile_env(extra=None):
  """Build an env dict that disables on-disk profile loading.

  Tests should not be affected by whatever profile the developer happens
  to have saved at ~/.kinetic/profiles.json. Pointing KINETIC_PROFILES_FILE
  at a nonexistent path makes `resolve_active()` return None.
  """
  env = {"KINETIC_PROFILES_FILE": "/nonexistent/kinetic-profiles.json"}
  if extra:
    env.update(extra)
  return env


class TestEnvVarCapture(absltest.TestCase):
  def test_exact_match(self):
    mock_handle = MagicMock()
    mock_handle.result.return_value = None
    with (
      mock.patch.dict(
        os.environ,
        _isolate_profile_env({"MY_VAR": "my_val", "KINETIC_PROJECT": "p"}),
      ),
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
    env = _isolate_profile_env(
      {
        "PREFIX_A": "1",
        "PREFIX_B": "2",
        "OTHER": "3",
        "KINETIC_PROJECT": "p",
      }
    )
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
    env = _isolate_profile_env({"KINETIC_PROJECT": "p"})
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
    env = _isolate_profile_env(
      {
        "EXACT_VAR": "exact",
        "WILD_A": "a",
        "WILD_B": "b",
        "KINETIC_PROJECT": "p",
      }
    )
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
        _isolate_profile_env(
          {
            "KINETIC_CLUSTER": "env-cluster",
            "KINETIC_PROJECT": "proj",
          }
        ),
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
        _isolate_profile_env(
          {
            "KINETIC_NAMESPACE": "custom-ns",
            "KINETIC_PROJECT": "proj",
          }
        ),
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


class TestResolveInfra(absltest.TestCase):
  """Unit tests for the precedence chain in `resolve_infra`."""

  def _stage_profile(self, **fields):
    """Write a one-profile store and return env dict pointing at it."""
    fd, path = tempfile.mkstemp(suffix=".json", prefix="kinetic-profiles-")
    os.close(fd)
    self.addCleanup(os.unlink, path)
    payload = {"current": "p", "profiles": {"p": fields}}
    with open(path, "w", encoding="utf-8") as f:
      json.dump(payload, f)
    return {"KINETIC_PROFILES_FILE": path}

  def test_precedence_kwarg_beats_env_beats_profile_beats_default(self):
    """Each layer wins over the layers below for the field it sets."""
    env = self._stage_profile(
      project="prof-proj",
      zone="prof-zone",
      cluster="prof-cluster",
      namespace="prof-ns",
    )
    # Sets env for cluster only; kwarg only for namespace.
    env["KINETIC_CLUSTER"] = "env-cluster"
    with mock.patch.dict(os.environ, env, clear=True):
      out = resolve_infra(
        project=None, zone=None, cluster=None, namespace="kwarg-ns"
      )
    self.assertEqual(
      out,
      {
        "project": "prof-proj",  # profile (no kwarg / env)
        "zone": "prof-zone",  # profile (no kwarg / env)
        "cluster": "env-cluster",  # env beats profile
        "namespace": "kwarg-ns",  # kwarg beats env / profile
      },
    )

  def test_built_in_defaults_when_nothing_set(self):
    """Without a profile or env vars, falls through to built-in defaults."""
    env = {
      "KINETIC_PROFILES_FILE": "/nonexistent/kinetic-profiles.json",
      "KINETIC_PROJECT": "fallback-proj",  # required, no default
    }
    with mock.patch.dict(os.environ, env, clear=True):
      out = resolve_infra(project=None, zone=None, cluster=None, namespace=None)
    self.assertEqual(out["zone"], DEFAULT_ZONE)
    self.assertEqual(out["cluster"], DEFAULT_CLUSTER_NAME)
    self.assertEqual(out["namespace"], "default")
    self.assertEqual(out["project"], "fallback-proj")


class TestProfileEndToEnd(absltest.TestCase):
  """Smoke test that an on-disk profile actually reaches the backend."""

  def test_profile_flows_through_run_to_backend(self):
    fd, path = tempfile.mkstemp(suffix=".json", prefix="kinetic-profiles-")
    os.close(fd)
    self.addCleanup(os.unlink, path)
    with open(path, "w", encoding="utf-8") as f:
      json.dump(
        {
          "current": "dev",
          "profiles": {
            "dev": {
              "project": "prof-project",
              "zone": "europe-west4-a",
              "cluster": "prof-cluster",
              "namespace": "prof-ns",
            }
          },
        },
        f,
      )

    mock_handle = MagicMock()
    mock_handle.result.return_value = None
    with (
      mock.patch.dict(os.environ, {"KINETIC_PROFILES_FILE": path}, clear=True),
      mock.patch(
        "kinetic.core.core.submit_remote", return_value=mock_handle
      ) as mock_submit,
      mock.patch(
        "kinetic.core.core.JobContext.from_params", return_value=MagicMock()
      ),
    ):

      @run(accelerator="cpu")
      def func():
        pass

      func()

      backend = mock_submit.call_args[0][1]
      self.assertEqual(backend.cluster, "prof-cluster")
      self.assertEqual(backend.namespace, "prof-ns")


class TestDebugRequiresInteractiveTerminal(absltest.TestCase):
  """run(debug=True) requires a TTY so the user can attach a debugger."""

  def test_run_debug_raises_when_stdin_not_tty(self):
    mock_handle = MagicMock()
    with (
      mock.patch.dict(
        os.environ,
        _isolate_profile_env({"KINETIC_PROJECT": "proj"}),
        clear=False,
      ),
      mock.patch(
        "kinetic.core.core.submit_remote",
        return_value=mock_handle,
      ),
      mock.patch(
        "kinetic.core.core.JobContext.from_params",
        return_value=MagicMock(),
      ),
      mock.patch("sys.stdin.isatty", return_value=False),
    ):
      # Ensure the override env var is not set.
      os.environ.pop("KINETIC_NO_TTY_DEBUG", None)

      @run(accelerator="cpu", debug=True)
      def func():
        pass

      with self.assertRaisesRegex(
        RuntimeError, "debug=True requires an interactive terminal"
      ):
        func()

      # The debug attach path must not have been invoked.
      mock_handle.debug_attach.assert_not_called()

  def test_run_debug_allowed_when_stdin_is_tty(self):
    mock_handle = MagicMock()
    mock_handle.result.return_value = 7
    with (
      mock.patch.dict(
        os.environ,
        _isolate_profile_env({"KINETIC_PROJECT": "proj"}),
        clear=False,
      ),
      mock.patch(
        "kinetic.core.core.submit_remote",
        return_value=mock_handle,
      ),
      mock.patch(
        "kinetic.core.core.JobContext.from_params",
        return_value=MagicMock(),
      ),
      mock.patch("sys.stdin.isatty", return_value=True),
      mock.patch("kinetic.core.core.cleanup_port_forward"),
    ):

      @run(accelerator="cpu", debug=True)
      def func():
        pass

      result = func()

      self.assertEqual(result, 7)
      mock_handle.debug_attach.assert_called_once()

  def test_run_debug_override_env_var_bypasses_tty_check(self):
    mock_handle = MagicMock()
    mock_handle.result.return_value = 7
    with (
      mock.patch.dict(
        os.environ,
        _isolate_profile_env(
          {"KINETIC_PROJECT": "proj", "KINETIC_NO_TTY_DEBUG": "1"}
        ),
        clear=False,
      ),
      mock.patch(
        "kinetic.core.core.submit_remote",
        return_value=mock_handle,
      ),
      mock.patch(
        "kinetic.core.core.JobContext.from_params",
        return_value=MagicMock(),
      ),
      mock.patch("sys.stdin.isatty", return_value=False),
      mock.patch("kinetic.core.core.cleanup_port_forward"),
    ):

      @run(accelerator="cpu", debug=True)
      def func():
        pass

      result = func()

      self.assertEqual(result, 7)
      mock_handle.debug_attach.assert_called_once()

  def test_run_without_debug_skips_tty_check(self):
    """debug=False should not require a TTY even when stdin is piped."""
    mock_handle = MagicMock()
    mock_handle.result.return_value = 42
    with (
      mock.patch.dict(
        os.environ,
        _isolate_profile_env({"KINETIC_PROJECT": "proj"}),
        clear=False,
      ),
      mock.patch(
        "kinetic.core.core.submit_remote",
        return_value=mock_handle,
      ),
      mock.patch(
        "kinetic.core.core.JobContext.from_params",
        return_value=MagicMock(),
      ),
      mock.patch("sys.stdin.isatty", return_value=False),
    ):

      @run(accelerator="cpu")
      def func():
        pass

      result = func()

      self.assertEqual(result, 42)
      mock_handle.debug_attach.assert_not_called()


class TestSubmitOnBackend(absltest.TestCase):
  def test_run_calls_result_on_handle(self):
    """run() is submit() + result() — calls .result() on the returned handle."""
    mock_handle = MagicMock()
    mock_handle.result.return_value = 123
    with (
      mock.patch.dict(
        os.environ, _isolate_profile_env({"KINETIC_PROJECT": "proj"})
      ),
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
      mock_handle.result.assert_called_once_with(stream_logs=True)


class TestRemoteCallableDescriptor(absltest.TestCase):
  def test_instance_method_sync(self):
    mock_handle = MagicMock()
    mock_handle.result.return_value = 42
    with (
      mock.patch.dict(
        os.environ, _isolate_profile_env({"KINETIC_PROJECT": "proj"})
      ),
      mock.patch("kinetic.core.core.submit_remote", return_value=mock_handle),
      mock.patch(
        "kinetic.core.core.JobContext.from_params", return_value=MagicMock()
      ) as mock_from_params,
    ):

      class Trainer:
        @run(accelerator="cpu")
        def train(self, lr):
          return lr

      trainer = Trainer()
      trainer.train(0.01)
      args = mock_from_params.call_args[0][1]
      self.assertEqual(len(args), 2)
      self.assertEqual(args[0], trainer)
      self.assertEqual(args[1], 0.01)

  def test_instance_method_async(self):
    mock_handle = MagicMock()
    with (
      mock.patch.dict(
        os.environ, _isolate_profile_env({"KINETIC_PROJECT": "proj"})
      ),
      mock.patch("kinetic.core.core.submit_remote", return_value=mock_handle),
      mock.patch(
        "kinetic.core.core.JobContext.from_params", return_value=MagicMock()
      ) as mock_from_params,
    ):

      class Trainer:
        @run(accelerator="cpu")
        def train(self, lr):
          return lr

      trainer = Trainer()
      trainer.train.run_async(0.01)
      args = mock_from_params.call_args[0][1]
      self.assertEqual(len(args), 2)
      self.assertEqual(args[0], trainer)
      self.assertEqual(args[1], 0.01)


if __name__ == "__main__":
  absltest.main()
