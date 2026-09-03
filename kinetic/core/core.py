from __future__ import annotations

import functools
import operator
import os
import re
import sys
import warnings
from typing import Any, Callable, Generic, ParamSpec, TypeVar, overload

from absl import logging

from kinetic.backend.execution import (
  GKEBackend,
  JobContext,
  PathwaysBackend,
  submit_remote,
)
from kinetic.cli.profiles import resolve_infra
from kinetic.collections import BatchHandle
from kinetic.collections import map as collections_map
from kinetic.core import accelerators
from kinetic.data import Data
from kinetic.debug import cleanup_port_forward
from kinetic.jobs import JobHandle

# Parameter signature and return type of the user's decorated function.
# These let @kinetic.run preserve the original call signature on the
# returned RemoteCallable so type checkers can resolve `.run_async(...)`.
P = ParamSpec("P")
R = TypeVar("R")


def _validate_volumes(volumes):
  """Validate the optional volumes mapping."""
  if volumes is None:
    return
  if not isinstance(volumes, dict):
    raise TypeError(f"volumes must be a dict, got {type(volumes).__name__}")
  for mount_path, data_obj in volumes.items():
    if not isinstance(mount_path, str) or not mount_path.startswith("/"):
      raise ValueError(
        f"Volume mount path must be an absolute path "
        f"(start with '/'), got: {mount_path!r}"
      )
    if not isinstance(data_obj, Data):
      raise TypeError(
        f"Volume value for {mount_path!r} must be a Data "
        f"instance, got {type(data_obj).__name__}"
      )


# The pod applies captured values over the image's own environment, so a
# wildcard that sweeps in client-side process configuration (a macOS PATH, a
# local VIRTUAL_ENV, KERAS_BACKEND=torch) breaks the job before user code
# runs. Wildcards never capture these; an exact name still does.
_WILDCARD_ENV_BLOCKLIST = frozenset(
  {
    "PATH",
    "HOME",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "SHELL",
    "TMPDIR",
    "TEMP",
    "TMP",
    "HOSTNAME",
    "USER",
    "LOGNAME",
    "SSH_AUTH_SOCK",
    "KUBERNETES_SERVICE_HOST",
    "KERAS_BACKEND",
  }
)

_SECRET_NAME_PATTERN = re.compile(
  r"TOKEN|SECRET|KEY|PASSWORD|CREDENTIAL", re.IGNORECASE
)

# Kept conservative so a sanitized repr is safe to embed in a display name.
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def _capture_env(capture_env_vars):
  """Capture requested environment variables for remote execution.

  Args:
    capture_env_vars: Names to capture. A trailing ``*`` makes the entry a
      prefix pattern (``"AWS_*"``); ``"*"`` matches everything.

  Returns:
    Dict of captured variable names to their current values. Wildcard
    matches skip process-critical variables (`_WILDCARD_ENV_BLOCKLIST`);
    naming such a variable exactly still captures it.
  """
  env_vars = {}
  if not capture_env_vars:
    return env_vars

  explicit = {p for p in capture_env_vars if not p.endswith("*")}
  skipped = set()

  for pattern in capture_env_vars:
    if pattern.endswith("*"):
      prefix = pattern[:-1]
      for name, value in os.environ.items():
        if not name.startswith(prefix):
          continue
        if name in _WILDCARD_ENV_BLOCKLIST and name not in explicit:
          skipped.add(name)
          continue
        env_vars[name] = value
    elif pattern in os.environ:
      env_vars[pattern] = os.environ[pattern]

  if skipped:
    logging.info(
      "capture_env_vars: not capturing process-critical variables matched by "
      "a wildcard: %s. Overriding the pod's own values for these breaks the "
      "runtime; list a name exactly in capture_env_vars if you really need it.",
      ", ".join(sorted(skipped)),
    )
  if env_vars:
    logging.info(
      "capture_env_vars: capturing %d environment variable(s): %s",
      len(env_vars),
      ", ".join(sorted(env_vars)),
    )
  secrets = sorted(n for n in env_vars if _SECRET_NAME_PATTERN.search(n))
  if secrets:
    logging.warning(
      "capture_env_vars is shipping credential-looking variables (%s). Their "
      "values are stored in plaintext inside the job payload in the job "
      "bucket, which is readable by every job pod in the cluster. Artifacts "
      "are deleted only when a job's result is collected; otherwise they "
      "remain until the bucket retention period expires. This notice is "
      "informational and appears even for variables you listed explicitly; "
      "if shipping this credential is intentional, no action is needed.",
      ", ".join(secrets),
    )
  return env_vars


def _unwrap_partial(func):
  """Return the innermost callable behind a chain of functools.partial."""
  while isinstance(func, functools.partial):
    func = func.func
  return func


def _safe_func_name(func):
  """Build a display-safe name for an arbitrary callable.

  functools.partial objects and callable instances have no ``__name__``;
  fall back to a sanitized, truncated repr rather than raising.
  """
  target = _unwrap_partial(func)
  name = getattr(target, "__name__", None)
  if isinstance(name, str) and name:
    return name
  sanitized = _UNSAFE_NAME_CHARS.sub("-", repr(target)).strip("-")[:40]
  return sanitized or type(target).__name__


def _validate_decorated_callable(func):
  """Reject callables that cannot survive submission, at decoration time.

  Each of these otherwise fails much later — after credentials, image
  build, upload and node scheduling — with an error that names no user
  symbol.
  """
  if isinstance(func, (classmethod, staticmethod)):
    kind = type(func).__name__
    raise TypeError(
      f"@kinetic.run cannot wrap a {kind} object. Apply @kinetic.run below "
      f"@{kind}:\n"
      f"  @{kind}\n"
      f"  @kinetic.run(...)\n"
      f"  def my_func(...): ...\n"
      f"Applying it above @{kind} submits a job that cannot run on the pod."
    )
  # Deliberately type-specific guards: the generic unpicklable case is
  # already caught at submit time by save_payload's bisection (which names
  # the offending component). These exist to fail at decoration time with
  # remediation advice — "reorder your decorators" — that only a
  # type-specific check can give. Duck-typed so functools.cache and
  # third-party cache decorators with the same interface are caught too.
  if hasattr(func, "cache_clear") and hasattr(func, "cache_info"):
    raise TypeError(
      "@kinetic.run cannot wrap a functools.lru_cache/functools.cache "
      "wrapper: the cache object is not serializable, and a cache is "
      "meaningless remotely because every job runs in a fresh process. "
      "Decorate the undecorated function with @kinetic.run instead (and "
      "apply the cache to a local callable if you still want it locally)."
    )
  if not callable(func):
    raise TypeError(
      f"@kinetic.run expected a callable, got {type(func).__name__}"
    )


def _ensure_func_name(func):
  """Guarantee ``func.__name__`` exists before the function is submitted.

  The job display name is built from ``func.__name__`` at submit time, so a
  functools.partial or a callable instance used to raise AttributeError on
  the first call. Set a synthetic name in place when the object accepts
  attributes; refuse at decoration time when it does not.
  """
  if isinstance(getattr(func, "__name__", None), str):
    return func
  try:
    func.__name__ = _safe_func_name(func)
  except (AttributeError, TypeError) as e:
    raise TypeError(
      f"@kinetic.run received a {type(func).__name__} object with no "
      "__name__ attribute, and one cannot be set on it. Wrap the call in a "
      "plain function and decorate that instead:\n"
      "  @kinetic.run(...)\n"
      "  def my_job(...):\n"
      "    return the_callable(...)"
    ) from e
  return func


def _require_interactive_terminal():
  """Raise if stdin is not a TTY and KINETIC_NO_TTY_DEBUG is not set.

  ``run(debug=True)`` blocks waiting for a VS Code debugger to attach.
  Without a TTY (CI, cron, nohup, piped input), no one can attach and
  the job burns the whole attach window before falling through.  Called
  before the job is submitted so nothing lands on the cluster; fails
  fast with a clear message instead.  Set ``KINETIC_NO_TTY_DEBUG=1`` to
  override (useful for automated tests).
  """
  if os.environ.get("KINETIC_NO_TTY_DEBUG") == "1":
    return
  if not sys.stdin.isatty():
    raise RuntimeError(
      "debug=True requires an interactive terminal but stdin is not a TTY. "
      "Either remove debug=True, or call func.run_async() and attach with "
      "handle.debug_attach() from an interactive session, or set "
      "KINETIC_NO_TTY_DEBUG=1 to override."
    )


def _resolve_backend_name(accelerator, backend, spot=False):
  """Resolve the backend from explicit config or accelerator type."""
  if backend is not None:
    return backend

  try:
    accel_config = accelerators.parse_accelerator(accelerator, spot=spot)
    if (
      isinstance(accel_config, accelerators.TpuConfig)
      and accel_config.num_nodes > 1
    ):
      return "pathways"
  except ValueError:
    pass
  return "gke"


def _make_decorator(
  accelerator,
  container_image,
  base_image_repo,
  zone,
  project,
  capture_env_vars,
  cluster,
  backend,
  namespace,
  volumes,
  spot,
  sync,
  output_dir,
  debug,
):
  """Build a decorator that submits the wrapped function for remote execution.

  Args:
    sync: If True, block on result (`run()` semantics).
      If False, return a `JobHandle` immediately (`run_async()` semantics).
    debug: If True, enable debugpy remote debugging.
  """

  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      # Checked before anything is submitted: a blocking debug call that
      # nobody can attach to would otherwise leave a job on the cluster
      # that waits out the whole attach window and then runs anyway.
      if sync and debug:
        _require_interactive_terminal()

      env_vars = _capture_env(capture_env_vars)
      resolved_backend = _resolve_backend_name(accelerator, backend, spot=spot)

      if resolved_backend not in ("gke", "pathways"):
        raise ValueError(
          f"Unknown backend: {resolved_backend}. "
          "Use 'gke', 'pathways', or None for auto-detection"
        )

      infra = resolve_infra(
        project=project, zone=zone, cluster=cluster, namespace=namespace
      )

      ctx = JobContext.from_params(
        func,
        args,
        kwargs,
        accelerator,
        container_image,
        infra["zone"],
        infra["project"],
        env_vars,
        cluster_name=infra["cluster"],
        volumes=volumes,
        spot=spot,
        debug=debug,
        output_dir=output_dir,
        base_image_repo=base_image_repo,
      )

      if resolved_backend == "pathways":
        backend_inst = PathwaysBackend(
          cluster=infra["cluster"], namespace=infra["namespace"]
        )
      else:
        backend_inst = GKEBackend(
          cluster=infra["cluster"], namespace=infra["namespace"]
        )

      handle = submit_remote(ctx, backend_inst)

      if sync:
        if debug:
          pf_proc = handle.debug_attach(working_dir=ctx.working_dir)
          try:
            return handle.result(stream_logs=False, cleanup=False)
          finally:
            cleanup_port_forward(pf_proc)
        return handle.result(stream_logs=True)
      return handle

    return wrapper

  return decorator


class RemoteCallable(Generic[P, R]):
  """Wrapper class returned by @kinetic.run to handle sync and async calls.

  Generic over the decorated function's parameter signature (``P``) and
  return type (``R``) so type checkers can resolve `.run_async(...)` and
  preserve the original call signature.

  Supports instance methods via the descriptor protocol (__get__).
  """

  def __init__(
    self,
    func: Callable[P, R],
    sync_wrapper: Callable[..., Any],
    async_wrapper: Callable[..., JobHandle],
  ):
    self._func = func
    self._sync_wrapper = sync_wrapper
    self._async_wrapper = async_wrapper
    functools.update_wrapper(self, func)

  def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
    """Synchronous execution (blocks)."""
    return self._sync_wrapper(*args, **kwargs)

  def __reduce__(self):
    """Pickle as the plain wrapped function, dropping the kinetic wrapper.

    Anything that references a decorated callable — a class whose body
    defines one, or an instance of such a class passed as an argument —
    would otherwise pull `kinetic.*` module globals into the payload
    through the submit closures, and the job image does not contain the
    kinetic package, so the pod dies at unpickle time.

    Semantics on the pod: the attribute is the undecorated function, so
    calling it there runs the body in the pod process instead of
    submitting another job (nested submission is not supported).
    """
    # The reconstructor must be importable on the pod, which only has the
    # standard library plus the job's own dependencies — hence operator
    # rather than a kinetic-level identity helper.
    return (operator.itemgetter(0), ((self._func,),))

  def run_async(self, *args: P.args, **kwargs: P.kwargs) -> JobHandle:
    """Submit the decorated function for remote execution without blocking.

    Unlike calling the wrapper directly (which blocks until the remote job
    finishes), `run_async` returns as soon as the job is submitted. The
    returned `JobHandle` is a durable reference to the job: you can exit
    the process, then reattach from any machine with the handle's
    `job_id` to stream logs or collect the result.

    Args:
      *args: Positional arguments forwarded to the decorated function on
        the remote worker. Must match the function's signature and be
        picklable.
      **kwargs: Keyword arguments forwarded to the decorated function on
        the remote worker. Must be picklable.

    Returns:
      A `JobHandle` for the submitted job. Use `handle.job_id` to
      reattach later, `handle.result()` to block until completion and
      fetch the return value, and `handle.cancel()` to stop the job.

    Examples::

        @kinetic.run(accelerator="cpu")
        def train():
          ...

        # Fire and forget, then reattach from another process.
        handle = train.run_async()
        print(handle.job_id)  # later: kinetic.attach(handle.job_id)

        # Submit, do other work locally, then collect the result.
        handle = train.run_async()
        result = handle.result()
    """
    return self._async_wrapper(*args, **kwargs)

  def run_async_map(self, inputs, **kwargs) -> BatchHandle:
    """Fan the decorated function out over many inputs as parallel jobs.

    Each item in `inputs` is submitted as its own independent remote job
    (one accelerator per job), letting you sweep hyperparameters or shard
    a dataset across the cluster. This is a convenience wrapper around
    `kinetic.collections.map(self.run_async, inputs, **kwargs)`.

    Args:
      inputs: Iterable of inputs to fan out over. By default (`input_mode`
        of `"auto"`), each item is dispatched by type: a `dict` is passed
        as `**kwargs`, a `list`/`tuple` as `*args`, and any other value as
        a single positional argument.
      **kwargs: Options forwarded to `kinetic.collections.map`, such as
        `input_mode`, `max_concurrent`, `retries`, `fail_fast`,
        `cancel_running_on_fail`, `name`, `tags`, `project`, and
        `cluster`.

    Returns:
      A `BatchHandle` for observing, collecting, and cleaning up the
      group of jobs. Use `handle.results()` to gather every return value
      and `handle.wait()` to block until the batch finishes.

    Examples::

        @kinetic.run(accelerator="tpu-v6e-1")
        def train(lr):
          ...

        # Sweep over learning rates, one remote job per value.
        batch = train.run_async_map([{"lr": 0.1}, {"lr": 0.01}])
        losses = batch.results()

        # Cap concurrency so at most four jobs run at once.
        batch = train.run_async_map(grid, max_concurrent=4)
    """

    return collections_map(self._async_wrapper, inputs, **kwargs)

  @overload
  def __get__(self, instance: None, owner: Any) -> RemoteCallable[P, R]: ...

  @overload
  def __get__(
    self, instance: object, owner: Any
  ) -> _BoundRemoteCallable[R]: ...

  def __get__(
    self, instance, owner
  ) -> RemoteCallable[P, R] | _BoundRemoteCallable[R]:
    if instance is None:
      return self
    return _BoundRemoteCallable(self, instance)


class _BoundRemoteCallable(Generic[R]):
  """Proxy for RemoteCallable bound to an instance.

  Binding an instance method drops the leading ``self`` argument, which a
  bare ``ParamSpec`` cannot express, so the wrapped callable is typed with
  an open signature (``RemoteCallable[..., R]``) here.
  """

  def __init__(self, callable_: RemoteCallable[..., R], instance):
    self._c = callable_
    self._instance = instance

  def __call__(self, *args, **kwargs) -> R:
    return self._c(self._instance, *args, **kwargs)

  def run_async(self, *args, **kwargs) -> JobHandle:
    return self._c.run_async(self._instance, *args, **kwargs)

  def run_async_map(self, inputs, **kwargs) -> BatchHandle:
    def bound_async_wrapper(*a, **kw):
      return self._c._async_wrapper(self._instance, *a, **kw)

    from kinetic.collections import map as collections_map

    return collections_map(bound_async_wrapper, inputs, **kwargs)


def run(
  accelerator: str = "tpu-v5e-1",
  container_image: str | None = None,
  base_image_repo: str | None = None,
  zone: str | None = None,
  project: str | None = None,
  capture_env_vars: list[str] | None = None,
  cluster: str | None = None,
  backend: str | None = None,
  namespace: str | None = None,
  volumes: dict[str, Data] | None = None,
  spot: bool = False,
  output_dir: str | None = None,
  debug: bool = False,
) -> Callable[[Callable[P, R]], RemoteCallable[P, R]]:
  """Execute function on remote TPU/GPU.

  Args:
    accelerator: TPU/GPU type (e.g., 'tpu-v3-8', 'tpu-v5litepod-4', 'gpu-l4', 'gpu-a100')
    container_image: Controls the container image used for execution.
      `None` or `"bundled"` (default) builds a custom image with all
      dependencies baked in via Cloud Build.  `"prebuilt"` uses a
      prebuilt base image and installs user requirements at pod startup
      via `uv pip install`.  Any other string is treated as a custom
      container image URI.
    base_image_repo: Docker Hub repository for prebuilt base images
      (e.g., `"mycompany/kinetic"`). Defaults to `KINETIC_BASE_IMAGE_REPO`
      env var, then `"kinetic"`. Only used when `container_image` is
      `"prebuilt"`.
    zone: GCP zone. Falls back to KINETIC_ZONE, then the active profile's
      zone (from ~/.kinetic/profiles.json), then 'us-central1-a'.
    project: GCP project. Falls back to KINETIC_PROJECT, then the active
      profile's project, then GOOGLE_CLOUD_PROJECT.
    capture_env_vars: List of environment variable names or patterns (ending in `*`)
      to propagate to the remote environment. Defaults to None. Wildcard
      patterns never capture process-critical variables such as `PATH`,
      `HOME`, `PYTHONPATH`, `VIRTUAL_ENV` or `KERAS_BACKEND` (overriding
      the pod's own values breaks the runtime); name one exactly to
      capture it anyway. Captured values are stored in the job payload in
      the job bucket, so avoid sweeping credentials in with a wildcard.
    cluster: GKE cluster name. Falls back to KINETIC_CLUSTER, then the
      active profile's cluster, then the built-in default.
    backend: Backend to use ('gke' or 'pathways')
    namespace: Kubernetes namespace. Falls back to KINETIC_NAMESPACE, then
      the active profile's namespace, then 'default'.
    volumes: Dict mapping absolute mount paths to Data objects, e.g.
      `{"/data": Data("./dataset/")}`. Data is downloaded to these
      paths on the pod before function execution.
    spot: If True, use preemptible/spot VMs for the job.
    output_dir: GCS directory where job outputs should be saved.
      Propagated to the remote worker as the `KINETIC_OUTPUT_DIR`
      environment variable. Defaults to `gs://{bucket_name}/outputs/{job_id}`.
    debug: If True, enable debugpy remote debugging. The pod will start
      a debugpy server and wait for a VS Code debugger to attach before
      executing the function. Port-forwarding is set up automatically.

  Returns:
    A decorator that returns a wrapper function. When called, the wrapper
    executes the function remotely and blocks until completion (sync mode).
    The wrapper also has the following methods:

    - `run_async(*args, **kwargs)`: Submits the job for remote execution
      and returns a `JobHandle` immediately (async mode).
    - `run_async_map(inputs, **kwargs)`: Fans out across accelerators
      for a collection of inputs, returning a `BatchHandle`.

  Raises:
    TypeError: If the decorated object is a `classmethod`/`staticmethod`
      object (apply `@kinetic.run` below those), a `functools.lru_cache`
      wrapper, or not callable at all.
  """
  _validate_volumes(volumes)

  if debug and spot:
    warnings.warn(
      "debug=True with spot=True is not recommended — your debug "
      "session may be interrupted by preemption.",
      stacklevel=3,
    )

  def decorator(func: Callable[P, R]) -> RemoteCallable[P, R]:
    _validate_decorated_callable(func)
    func = _ensure_func_name(func)

    # Create the sync wrapper
    sync_decorator = _make_decorator(
      accelerator,
      container_image,
      base_image_repo,
      zone,
      project,
      capture_env_vars,
      cluster,
      backend,
      namespace,
      volumes,
      spot,
      sync=True,
      output_dir=output_dir,
      debug=debug,
    )
    sync_wrapper = sync_decorator(func)

    # Create the async wrapper
    async_decorator = _make_decorator(
      accelerator,
      container_image,
      base_image_repo,
      zone,
      project,
      capture_env_vars,
      cluster,
      backend,
      namespace,
      volumes,
      spot,
      sync=False,
      output_dir=output_dir,
      debug=debug,
    )
    async_wrapper = async_decorator(func)

    return RemoteCallable(func, sync_wrapper, async_wrapper)

  return decorator
