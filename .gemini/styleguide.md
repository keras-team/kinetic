When performing code reviews on pull requests, you must strictly adhere to the following principles in addition to the API design guidelines above:

1. **Question the Necessity of Changes**: Do not assume that the pull request changes are strictly necessary. Critically review the proposed changes to ensure they add real value. Point out any code that solving a non-existent problem or adding unnecessary complexity.

2. **Call out "AI Slop"**: Actively look for and identify "AI slop"—generic, overly verbose, or hallucinated code that lacks context or violates best practices. If you suspect the code is AI slop, explicitly call it out.

3. **Poke Holes in the Implementation**: Your goal is to critically test the logic. Actively search for and point out failing edge cases, race conditions, or unhandled exceptions in the implementation.

4. **Demand Robustness**: Do not accept fragile code. If the proposed code is not robust enough or lacks proper error handling, explicitly tell the author why the current approach is brittle and what must be done to reinforce it.

5. **Respect Existing Repo Patterns**: Before suggesting review comments (like asking users to add boilerplate or specific patterns), actively check for existing design patterns across the repository. Do not suggest adding useless code or structures that contradict or fall outside the established Keras repo coding style.

# Keras Remote API design guidelines

These guidelines are meant to help focus design discussions and help us create delightful developer experiences for remote execution.

These are meant as guidelines, not rules: each decision should be debated in its own unique context.

---

## Design end-to-end workflows, not individual functions and classes.

When developing APIs, start by designing end-to-end workflows.

- The goal is to arrive at workflows that feel like they are purposefully designed and well-optimized. **Features only exist to support a workflow.** No feature should exist to provide a capability "just in case".
- **Every design review document should prominently feature a code example of one or two end-to-end workflows.**
- Ask: **in what workflows will this be used?**
- Ask: **do users really need to configure this parameter**, or can we infer it from the context (e.g., defaulting to the current gcloud project)?

### Example:

Instead of exposing low-level cluster creation and job submission as separate steps that the user must manually stitch together:

```python
# Bad: Fragmented workflow
cluster = create_cluster("my-cluster")
job = submit_job(cluster, my_function)
result = wait_for_result(job)
```

We design for the end-to-end user intent ("run this function on a TPU"):

```python
# Good: Workflow-centric
@keras_remote.run(accelerator="v3-8")
def my_function():
    ...
```

---

## Carefully weigh whether a new feature should be included.

Every feature has a maintenance cost.

- **It should be broadly useful.** Avoid niche infrastructure options unless critical.
- **It should be a best practice.** We don't support every experimental scheduler or accelerator immediately.
- **It should have an owner.**

---

## Seek to minimize cognitive load for our users.

- **Automate everything.** Infrastructure provisioning, Docker builds, and auth should be invisible.
- **Minimize actions & choices.** Sensible defaults are key (e.g., default zone `us-central1-a`, default project from gcloud config).
- **Design simple consistent workflows.**

### Practical rules:

- **No API should deal with internal implementation details.** Users shouldn't need to know about Kubernetes pods, node pools, or internal IP addresses unless absolutely necessary for advanced debugging.
- **Introduce as few new concepts as possible.** The mental model should be "running a function on an accelerator", not "managing a distributed system".
- **Objects that do interchangeable things should have identical APIs.** Different backends should be interchangeable with minimal changes.
- **Plain Python types are preferable.** Use strings for accelerators (`"v3-8"`) rather than custom Enums.

### Naming:

- **The meaning of an argument should be clear.** `accelerator` is better than `machine_type` or `instance_config`.
- **Consistency is key.** modifying the `zone` argument should be consistent across all backends.
- **Avoid OverlyLongAndSpecificNamingPatterns.**

### Example:

```python
# Bad: Implementation details leaking into API
@keras_remote.run(
    tpu_topology="2x2",
    k8s_namespace="user-1",
    docker_image_repo="gcr.io/..."
)
```

```python
# Good: Focused on the user problem
@keras_remote.run(
    accelerator="v3-8",
    container_image="my-image"
)
```

---

## Balance expressivity vs. user-friendliness.

### Simple use cases should be simple, advanced use cases should be possible.

- **Don't increase cognitive load of common use cases.**
- **Make sure advanced users have a path.** For example, allow identifying a specific `cluster` or `service_account` for power users, but keep them optional.

### APIs should be strictly compartmentalized.

- The `keras_remote.run` decorator shouldn't handle model definition logic. It should only handle execution.

---

## Keep resource name resolution consistent across all usage paths.

Every configurable resource name (project, zone, cluster, namespace, etc.) must be resolvable through the same set of paths:

1. **Explicit parameter** to `@run()` (highest priority)
2. **Environment variable** (`KERAS_REMOTE_*`)
3. **CLI flag** (with Click's `envvar=` for automatic env var binding)
4. **Interactive prompt or sensible default** (lowest priority)

When adding a new configurable name:

- Add a parameter to `@run()` with env var fallback
- Add a `--flag` with `envvar=` to **every relevant CLI command** (not just `up` — also `down`, `status`, etc.)
- Add a row to `config show` so users can verify their configuration
- Ensure the env var fallback order is identical everywhere the name is resolved

This prevents confusing situations where a user sets an env var that works in one path but is silently ignored in another.

---

## CLI commands must be idempotent and use the centralized state module.

All Pulumi stack operations go through `cli/infra/state.py` — **no command file should call `stack.up()`, `stack.destroy()`, `stack.refresh()`, or import `create_program`/`get_stack` directly.**

The three entry points are:

- `load_state(project, zone, cluster_name)` → `StackState` — loads ALL state dimensions (refresh, node pools, etc.)
- `apply_update(config)` — runs `stack.up()` with a complete `InfraConfig`
- `apply_destroy(config)` — runs `stack.destroy()`

Every mutating CLI command follows this pattern:

1. `load_state()` — refresh stack and read all current state dimensions into `StackState`
2. Build `InfraConfig` — merge existing state with desired changes
3. `apply_update(config)` or `apply_destroy(config)` — apply the diff

When adding a **new state dimension** (e.g. namespaces), add it to `StackState` and `load_state()` — every command inherits it automatically, preventing accidental omissions that would cause Pulumi to delete resources.

All CLI commands must use the `common_options` decorator from `cli/options.py` for `--project`/`--zone`/`--cluster` flags — never define these inline.

This ensures:

- Re-running after partial failure is always safe
- Existing resources are never accidentally recreated (Pulumi tracks by URN)
- External drift is detected and corrected
- New state dimensions cannot be accidentally omitted by individual commands

---

## All infrastructure resources must be cluster-scoped.

Every resource managed by the CLI must include the cluster name in its identifier so that multiple clusters within the same GCP project are fully independent. The naming convention is `{project}-kr-{cluster_name}-{purpose}` for buckets and `kr-{cluster_name}` for Artifact Registry repos.

| Resource      | Name pattern                         |
| ------------- | ------------------------------------ |
| Pulumi stack  | `{project}-{cluster_name}`           |
| Jobs bucket   | `{project}-kr-{cluster_name}-jobs`   |
| Builds bucket | `{project}-kr-{cluster_name}-builds` |
| AR repository | `kr-{cluster_name}`                  |

The only exception is project-wide GCP API enablement, which is intentionally shared across clusters (`disable_on_destroy=False`).

When adding a new infrastructure resource, always scope it to the `(project, cluster_name)` pair. Runtime code (`JobContext`, `container_builder`) resolves the cluster name from `KERAS_REMOTE_CLUSTER` env var or default.

---

## Prefer graceful degradation over hard failures in CLI operations.

Partial failures in multi-step CLI operations should not abort the entire flow:

- If `stack.refresh()` fails, log a warning and continue with stale state
- If `stack.up()` fails, set a failure flag but still run post-deploy steps
- If a post-deploy step fails (kubectl, LWS, GPU drivers), log a warning and continue with remaining steps

The user can always re-run the same command to recover, since all operations are idempotent.

---

## Don't neglect error messages, docstrings, and documentation.

- **Catch user errors early.** Validate GCP project existence and quota before starting a long build.
- **Provide detailed feedback.**
  - Bad: `Error: 403 Forbidden`
  - Good: `Permission denied. Please ensure your account 'user@example.com' has the 'Storage Object Admin' role on bucket 'gs://my-bucket'.`
- **Show, don't tell.** Documentation should show code examples of running functions, not just list arguments.

### Error messages: a case study

Bad:

```
RuntimeError: Job failed.
```

Good:

```
RuntimeError: The remote job failed with exit code 1.
Logs from the worker:
...
ModuleNotFoundError: No module named 'tensorflow_datasets'
...
Tip: You can add 'tensorflow_datasets' to your requirements.txt file to install it on the remote worker.
```
