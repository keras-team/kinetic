"""Tests for kinetic.core.core — run/submit decorators and env var capture."""

import functools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from unittest import mock
from unittest.mock import MagicMock

import cloudpickle
from absl.testing import absltest

from kinetic.cli.profiles import resolve_infra
from kinetic.constants import DEFAULT_CLUSTER_NAME, DEFAULT_ZONE
from kinetic.core.core import _capture_env, _safe_func_name, run


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
      ) as mock_submit,
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

      # Nothing may reach the cluster: a submitted debug job that nobody
      # can attach to sits there for the whole attach window, then runs.
      mock_submit.assert_not_called()

      # The debug attach path must not have been invoked.
      mock_handle.debug_attach.assert_not_called()

  def test_run_async_debug_submits_without_tty(self):
    """Only the blocking path needs a TTY; run_async() attaches later."""
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
      ) as mock_submit,
      mock.patch(
        "kinetic.core.core.JobContext.from_params",
        return_value=MagicMock(),
      ),
      mock.patch("sys.stdin.isatty", return_value=False),
    ):
      os.environ.pop("KINETIC_NO_TTY_DEBUG", None)

      @run(accelerator="cpu", debug=True)
      def func():
        pass

      handle = func.run_async()

      self.assertIs(handle, mock_handle)
      mock_submit.assert_called_once()

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


class TestEnvCaptureBlocklist(absltest.TestCase):
  """Wildcards must not sweep in process-critical or secret-looking vars."""

  def test_wildcard_skips_process_critical_vars(self):
    env = {
      "PATH": "/opt/homebrew/bin",
      "HOME": "/Users/alice",
      "PYTHONPATH": "/Users/alice/repo",
      "LD_LIBRARY_PATH": "/opt/homebrew/lib",
      "VIRTUAL_ENV": "/Users/alice/.venv",
      "KERAS_BACKEND": "torch",
      "MY_FLAG": "on",
    }
    with mock.patch.dict(os.environ, env, clear=True):
      captured = _capture_env(["*"])
    self.assertEqual(captured, {"MY_FLAG": "on"})

  def test_prefix_wildcard_skips_blocklisted_name(self):
    with mock.patch.dict(
      os.environ, {"KERAS_BACKEND": "torch", "KERAS_HOME": "/k"}, clear=True
    ):
      captured = _capture_env(["KERAS*"])
    self.assertEqual(captured, {"KERAS_HOME": "/k"})

  def test_explicit_name_overrides_blocklist(self):
    with mock.patch.dict(
      os.environ, {"KERAS_BACKEND": "torch", "OTHER": "1"}, clear=True
    ):
      captured = _capture_env(["KERAS_BACKEND"])
    self.assertEqual(captured, {"KERAS_BACKEND": "torch"})

  def test_explicit_name_wins_even_alongside_wildcard(self):
    with mock.patch.dict(
      os.environ, {"KERAS_BACKEND": "torch", "KERAS_HOME": "/k"}, clear=True
    ):
      captured = _capture_env(["KERAS*", "KERAS_BACKEND"])
    self.assertEqual(captured, {"KERAS_BACKEND": "torch", "KERAS_HOME": "/k"})

  def test_captured_names_are_logged_without_values(self):
    with (
      mock.patch.dict(os.environ, {"MY_FLAG": "supersecretvalue"}, clear=True),
      mock.patch("kinetic.core.core.logging.info") as mock_info,
    ):
      _capture_env(["MY_FLAG"])
    messages = [
      call[0][0] % call[0][1:] if len(call[0]) > 1 else call[0][0]
      for call in mock_info.call_args_list
    ]
    joined = "\n".join(messages)
    self.assertIn("MY_FLAG", joined)
    self.assertNotIn("supersecretvalue", joined)

  def test_secret_looking_names_warn(self):
    env = {
      "AWS_SECRET_ACCESS_KEY": "wJalr",
      "HF_TOKEN": "hf_fake",
      "MY_FLAG": "on",
    }
    with (
      mock.patch.dict(os.environ, env, clear=True),
      mock.patch("kinetic.core.core.logging.warning") as mock_warning,
    ):
      captured = _capture_env(["AWS_*", "HF_TOKEN", "MY_FLAG"])
    self.assertEqual(len(captured), 3)
    mock_warning.assert_called_once()
    named = mock_warning.call_args[0][1]
    self.assertIn("AWS_SECRET_ACCESS_KEY", named)
    self.assertIn("HF_TOKEN", named)
    self.assertNotIn("MY_FLAG", named)
    # The warning fires for explicitly named credentials too, so its text
    # must say it is informational rather than a rejected capture.
    self.assertIn("informational", mock_warning.call_args[0][0])

  def test_no_warning_without_secret_names(self):
    with (
      mock.patch.dict(os.environ, {"MY_FLAG": "on"}, clear=True),
      mock.patch("kinetic.core.core.logging.warning") as mock_warning,
    ):
      _capture_env(["MY_FLAG"])
    mock_warning.assert_not_called()


class TestDecorationGuards(absltest.TestCase):
  """Bad decoration targets fail at decoration time, not on the pod."""

  def test_classmethod_object_rejected(self):
    with self.assertRaisesRegex(TypeError, r"below\s+@classmethod"):

      class _Trainer:
        @run(accelerator="cpu")
        @classmethod
        def train(cls):
          pass

  def test_staticmethod_object_rejected(self):
    with self.assertRaisesRegex(TypeError, r"below\s+@staticmethod"):

      class _Trainer:
        @run(accelerator="cpu")
        @staticmethod
        def train():
          pass

  def test_staticmethod_above_run_is_supported(self):
    class Trainer:
      @staticmethod
      @run(accelerator="cpu")
      def train(x):
        return x

    self.assertTrue(hasattr(Trainer.train, "run_async"))
    self.assertTrue(hasattr(Trainer().train, "run_async"))

  def test_lru_cache_rejected(self):
    with self.assertRaisesRegex(TypeError, "lru_cache"):

      @run(accelerator="cpu")
      @functools.lru_cache(maxsize=8)
      def cached(x):
        return x * 2

  def test_functools_cache_rejected(self):
    with self.assertRaisesRegex(TypeError, "lru_cache"):

      @run(accelerator="cpu")
      @functools.cache
      def cached(x):
        return x * 2

  def test_non_callable_rejected(self):
    with self.assertRaisesRegex(TypeError, "expected a callable"):
      run(accelerator="cpu")(42)

  def test_callable_without_settable_name_rejected(self):
    class Slotted:
      __slots__ = ()

      def __call__(self):
        return 1

    with self.assertRaisesRegex(TypeError, "__name__"):
      run(accelerator="cpu")(Slotted())


class TestDisplayNameSafety(absltest.TestCase):
  """Callables without __name__ must not crash the submit path."""

  def test_safe_func_name_unwraps_partial(self):
    def raw_train(model, lr):
      return (model, lr)

    self.assertEqual(
      _safe_func_name(functools.partial(raw_train, "gemma")), "raw_train"
    )

  def test_safe_func_name_falls_back_to_sanitized_repr(self):
    class Weird:
      def __call__(self):
        return 1

      def __repr__(self):
        return "<Weird object at 0x %s>" % ("y" * 200)

    name = _safe_func_name(Weird())
    self.assertLessEqual(len(name), 40)
    self.assertRegex(name, r"^[A-Za-z0-9_.-]+$")

  def test_partial_gets_a_name_and_submits(self):
    def raw_train(model, lr):
      return (model, lr)

    mock_handle = MagicMock()
    mock_handle.result.return_value = "ok"
    with (
      mock.patch.dict(
        os.environ, _isolate_profile_env({"KINETIC_PROJECT": "proj"})
      ),
      mock.patch("kinetic.core.core.submit_remote", return_value=mock_handle),
      mock.patch(
        "kinetic.core.core.JobContext.from_params", return_value=MagicMock()
      ) as mock_from_params,
    ):
      trainer = run(accelerator="cpu")(functools.partial(raw_train, "gemma"))
      self.assertEqual(trainer(0.1), "ok")

    submitted_func = mock_from_params.call_args[0][0]
    # JobContext.__post_init__ builds display_name from func.__name__.
    self.assertEqual(submitted_func.__name__, "raw_train")

  def test_callable_instance_gets_a_name_and_submits(self):
    class Trainer:
      def __call__(self, lr):
        return lr

    mock_handle = MagicMock()
    mock_handle.result.return_value = "ok"
    with (
      mock.patch.dict(
        os.environ, _isolate_profile_env({"KINETIC_PROJECT": "proj"})
      ),
      mock.patch("kinetic.core.core.submit_remote", return_value=mock_handle),
      mock.patch(
        "kinetic.core.core.JobContext.from_params", return_value=MagicMock()
      ) as mock_from_params,
    ):
      decorated = run(accelerator="cpu")(Trainer())
      self.assertEqual(decorated(0.1), "ok")

    submitted_func = mock_from_params.call_args[0][0]
    self.assertIsInstance(submitted_func.__name__, str)
    self.assertTrue(submitted_func.__name__)


@run(accelerator="cpu")
def _module_level_decorated(x):
  return x * 3


class TestRemoteCallablePickling(absltest.TestCase):
  """A decorated callable pickles as the plain wrapped function."""

  def test_module_level_decorated_pickles_as_plain_function(self):
    restored = cloudpickle.loads(cloudpickle.dumps(_module_level_decorated))
    self.assertFalse(hasattr(restored, "run_async"))
    self.assertEqual(restored(2), 6)

  def test_reduce_preserves_the_wrapped_function(self):
    reconstructor, args = _module_level_decorated.__reduce__()
    self.assertIs(reconstructor(*args), _module_level_decorated._func)

  def test_instance_of_class_with_decorated_method_loads_without_kinetic(self):
    tmp_dir = tempfile.mkdtemp(prefix="kinetic-reduce-")
    self.addCleanup(shutil.rmtree, tmp_dir, True)
    payload_path = os.path.join(tmp_dir, "payload.pkl")
    dump_path = os.path.join(tmp_dir, "dump.py")
    load_path = os.path.join(tmp_dir, "load.py")

    with open(dump_path, "w", encoding="utf-8") as f:
      f.write(
        textwrap.dedent(
          """
          import sys

          import cloudpickle

          import kinetic


          class Trainer:
            def __init__(self, lr):
              self.lr = lr

            @kinetic.run(accelerator="cpu")
            def train(self, extra):
              return self.lr + extra


          with open(sys.argv[1], "wb") as out:
            cloudpickle.dump((Trainer(1.5), Trainer.train), out)
          """
        )
      )
    with open(load_path, "w", encoding="utf-8") as f:
      f.write(
        textwrap.dedent(
          """
          import pickle
          import sys


          class _BlockKinetic:
            def find_spec(self, name, path=None, target=None):
              if name == "kinetic" or name.startswith("kinetic."):
                raise ModuleNotFoundError("No module named 'kinetic'")
              return None


          sys.meta_path.insert(0, _BlockKinetic())
          with open(sys.argv[1], "rb") as src:
            instance, func = pickle.load(src)
          assert "kinetic" not in sys.modules
          print(type(instance.train).__name__, instance.train(0.5),
                func(instance, 0.5))
          """
        )
      )

    repo_root = os.path.dirname(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    dump_env = dict(os.environ, PYTHONPATH=repo_root)
    subprocess.run(
      [sys.executable, dump_path, payload_path],
      check=True,
      cwd=tmp_dir,
      env=dump_env,
    )

    load_env = dict(os.environ)
    load_env.pop("PYTHONPATH", None)
    loaded = subprocess.run(
      [sys.executable, load_path, payload_path],
      check=True,
      cwd=tmp_dir,
      env=load_env,
      capture_output=True,
      text=True,
    )
    self.assertEqual(loaded.stdout.split(), ["method", "2.0", "2.0"])


if __name__ == "__main__":
  absltest.main()
