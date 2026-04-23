# Batched Jobs

`@kinetic.submit()` is the tool for a single long-running job. When you
need to run the **same function over many inputs**, such as a hyperparameter
sweep, one job per dataset shard, an evaluation grid — wiring that up
by hand means a loop that calls `submit()`, your own bookkeeping for
which handles are still live, your own error aggregation, your own
cleanup. `kinetic.map()` is that loop, done for you.

You hand it a `@kinetic.submit()`-decorated function and a list of
inputs. It returns a single `BatchHandle` that represents the whole
collection: one place to observe progress, collect results in input
order, handle failures, cancel siblings, and tear everything down. The
underlying jobs are independent Kinetic jobs — each one gets a real
`JobHandle`, runs on its own pod, and writes its own artifacts to GCS.

This page builds on the single-job workflow covered in
[Detached Jobs](async_jobs.md). Familiarity with `JobHandle` and the
`PENDING`/`RUNNING`/`SUCCEEDED`/`FAILED`/`NOT_FOUND` lifecycle is
assumed.

## A first fan-out

Pass a `@kinetic.submit()`-decorated function and a list of inputs to
`kinetic.map()`. It returns a `BatchHandle` immediately while jobs are
submitted in the background.

```python
import kinetic

@kinetic.submit(accelerator="tpu-v5e-1")
def train(lr):
    import keras
    model = keras.Sequential([keras.layers.Dense(64, activation="relu"),
                              keras.layers.Dense(1)])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    history = model.fit(x_train, y_train, epochs=10, verbose=0)
    return history.history["loss"][-1]

batch = kinetic.map(train, [0.001, 0.01, 0.1])
losses = batch.results()
print(losses)  # [0.32, 0.28, 0.41] — one result per input, in order
```

:::{note}
The first argument must be decorated with `@kinetic.submit()`, not
`@kinetic.run()`. `@kinetic.run()` blocks until the job finishes and
returns the result directly, so it cannot be used for fan-out.
:::

## Input modes

The `input_mode` parameter controls how each item in `inputs` is passed
to the submit function.

| `input_mode`       | Item type                         | How it's called | Example item                  |
| ------------------ | --------------------------------- | --------------- | ----------------------------- |
| `"auto"` (default) | `dict` with valid identifier keys | `fn(**item)`    | `{"lr": 0.01, "wd": 1e-4}`    |
| `"auto"` (default) | `list` or `tuple`                 | `fn(*item)`     | `[0.01, 32]`                  |
| `"auto"` (default) | anything else                     | `fn(item)`      | `0.01`                        |
| `"single"`         | any                               | `fn(item)`      | always passed as a single arg |
| `"args"`           | `list` or `tuple` (required)      | `fn(*item)`     | `[0.01, 32]`                  |
| `"kwargs"`         | `dict` (required)                 | `fn(**item)`    | `{"lr": 0.01}`                |

### Dict inputs (kwargs unpacking)

When using `"auto"` mode, dicts with valid Python identifier keys are
unpacked as keyword arguments:

```python
@kinetic.submit(accelerator="tpu-v5e-1")
def train(lr, batch_size):
    ...

configs = [
    {"lr": 0.001, "batch_size": 32},
    {"lr": 0.01,  "batch_size": 64},
]
batch = kinetic.map(train, configs)
```

### Preventing unpacking

If your function takes a list or dict as a single argument, use
`input_mode="single"` to prevent automatic unpacking:

```python
@kinetic.submit(accelerator="cpu")
def process(items):
    return sum(items)

batch = kinetic.map(process, [[1, 2, 3], [4, 5, 6]], input_mode="single")
```

:::{note}
In `"auto"` mode, dicts with non-identifier keys (like
`{"not-an-id": 1}`) or Python keywords (like `{"class": 1}`) are passed
as a single positional argument rather than unpacked. Use
`input_mode="kwargs"` or `input_mode="single"` if you need explicit
control.
:::

## Monitoring a batch

You can inspect progress at any time through the `BatchHandle`.

```python
# Per-job status
for idx, status in batch.statuses():
    print(f"Job {idx}: {status.value}")

# Aggregate counts
print(batch.status_counts())
# {'RUNNING': 2, 'SUCCEEDED': 1}
```

`statuses()` returns `(index, JobStatus)` pairs for each submitted job.
Slots that haven't been submitted yet (when using bounded concurrency)
are skipped. Job statuses follow the same lifecycle as single jobs —
see [Detached Jobs](async_jobs.md) for details on `PENDING`, `RUNNING`,
`SUCCEEDED`, `FAILED`, and `NOT_FOUND`.

## Collecting results

### `results()`

The simplest way to collect all results. By default it blocks until
every job finishes and returns results in input order.

```python
# Input order (default)
losses = batch.results()
# losses[0] corresponds to inputs[0], losses[1] to inputs[1], etc.
```

For faster access to early finishers, use `ordered=False` to collect in
completion order:

```python
losses = batch.results(ordered=False)
# Results appear in the order jobs finish, not input order
```

**Parameters:**

- **`timeout`** (`float | None`, default `None`): Maximum seconds to
  wait. Raises `TimeoutError` if exceeded.
- **`ordered`** (`bool`, default `True`): `True` returns results aligned
  with `inputs`. `False` returns results in the order jobs complete.
- **`cleanup`** (`bool`, default `True`): Delete each child's Kubernetes
  resources and GCS artifacts after downloading its result. The group
  manifest is preserved so `attach_batch()` still works.
- **`return_exceptions`** (`bool`, default `False`): When `True`, failed
  positions contain the exception object instead of raising
  `BatchError`. When `False`, any failure raises `BatchError`.

:::{important}
A `TimeoutError` does not cancel running jobs. They continue executing
on the cluster. Call `batch.cancel()` explicitly if you want to stop
them after a timeout.
:::

### `as_completed()`

For processing results incrementally as jobs finish, use the
`as_completed()` iterator. It yields `JobHandle` objects in completion
order.

```python
for job in batch.as_completed():
    result = job.result()
    print(f"{job.job_id} finished: {result}")
```

`as_completed()` streams results even while submission is still in
progress. With bounded concurrency, you can start processing the first
results before the last inputs have been submitted.

**Parameters:**

- **`poll_interval`** (`float`, default `5.0`): Seconds between status
  polls.
- **`timeout`** (`float | None`, default `None`): Maximum seconds to
  wait. Raises `TimeoutError` if exceeded.

## Handling failures

When any job fails and `return_exceptions=False` (the default),
`results()` raises a `BatchError`.

```python
try:
    results = batch.results()
except kinetic.BatchError as e:
    print(f"Batch {e.group_id}: {len(e.failures)} of {len(e.partial_results)} jobs failed")
    for job in e.failures:
        print(f"  {job.job_id}: {job.status().value}")
    # e.partial_results has results at successful positions, None at failed ones
```

`BatchError` provides three attributes:

- **`group_id`**: The batch identifier.
- **`failures`**: List of `JobHandle` objects for the failed jobs.
- **`partial_results`**: A list aligned with `inputs` where successful
  positions contain the result and failed positions contain `None`.

### Tolerating failures

Use `return_exceptions=True` to collect results without raising. Failed
positions contain the exception object.

```python
results = batch.results(return_exceptions=True)
for i, r in enumerate(results):
    if isinstance(r, Exception):
        print(f"Job {i} failed: {r}")
    else:
        print(f"Job {i}: {r}")
```

### Inspecting failures

`failures()` returns handles for jobs with status `FAILED`. It
intentionally excludes `NOT_FOUND` because that status is ambiguous —
a job may be `NOT_FOUND` because its Kubernetes resources were cleaned
up, not because it failed. Use `statuses()` for finer-grained
inspection.

```python
for job in batch.failures():
    print(f"{job.job_id}: {job.tail(n=20)}")
```

## Retries

The `retries` parameter specifies how many additional attempts a job
gets after failure. The total number of attempts per input is
`1 + retries`.

```python
batch = kinetic.map(train, configs, retries=2)
# Each job gets up to 3 attempts (1 initial + 2 retries)
```

- Retries are triggered when a job reaches `FAILED` or `NOT_FOUND`
  status.
- Before each retry, Kinetic cleans up the previous attempt's
  Kubernetes resources (GCS artifacts are preserved for debugging).
- The group manifest tracks the attempt count per job, so
  `attach_batch()` can distinguish retries from initial submissions.
- Submission errors (when the call to `submit_fn` itself raises) are
  not retried. These are typically packaging or configuration errors
  that would fail again.

:::{note}
When `retries > 0`, job submission runs in a background thread so
Kinetic can poll for failures and resubmit.
:::

## Concurrency control

By default, `kinetic.map()` limits the number of concurrently active
jobs to 64. Use `max_concurrent` to tune this.

```python
# At most 8 jobs running at once
batch = kinetic.map(train, configs, max_concurrent=8)
```

```python
# Submit all jobs immediately (no concurrency limit)
batch = kinetic.map(train, configs, max_concurrent=None)
```

- **Default:** `64`. New jobs are launched as running ones finish.
- **`None`:** All inputs are submitted immediately with no concurrency
  limit. When combined with `retries=0` (the default), submission
  happens synchronously in the calling thread before `map()` returns.
- Must be a positive integer when set. Passing `0` or a negative value
  raises `ValueError`.

:::{note}
Kinetic logs a warning when submitting more than 100 jobs with
`max_concurrent=None`, suggesting you set a limit to control resource
usage.
:::

## Cancellation and fail-fast

### Fail-fast behavior

The `fail_fast` and `cancel_running_on_fail` parameters control what
happens when a job fails.

| `fail_fast`       | `cancel_running_on_fail` | On first failure...                                                              |
| ----------------- | ------------------------ | -------------------------------------------------------------------------------- |
| `False` (default) | `False` (default)        | All remaining jobs continue. Failures are collected at the end.                  |
| `True`            | `False`                  | No new jobs are launched. Already-running jobs continue to completion.           |
| `True`            | `True`                   | No new jobs are launched. All running siblings are cancelled immediately.        |
| `False`           | `True`                   | **No effect.** `cancel_running_on_fail` only takes effect when `fail_fast=True`. |

```python
# Stop the batch as soon as any job fails, cancel all running siblings
batch = kinetic.map(
    train, configs,
    fail_fast=True,
    cancel_running_on_fail=True,
)
```

A "failure" here means either a submission error (the call to
`submit_fn` itself raised) or a runtime failure (the remote job reached
`FAILED` or `NOT_FOUND` status after exhausting retries).

### Manual cancellation

You can cancel all non-terminal jobs at any time, independent of the
`fail_fast` setting:

```python
batch.cancel()
```

Cancellation deletes each job's Kubernetes resource but preserves GCS
artifacts for debugging.

## Reattaching to a batch

If your local process exits or you want to check on a batch from a
different machine, save the `group_id` and reattach later.

```python
# Original session
batch = kinetic.map(train, configs)
print(f"Batch ID: {batch.group_id}")  # e.g., "grp-a1b2c3d4"

# Later, from any machine with access to the same GCP project
batch = kinetic.attach_batch("grp-a1b2c3d4")
results = batch.results()
```

`attach_batch()` downloads the group manifest from GCS and reconstructs
a `JobHandle` for each child. Index alignment is preserved: if the
original batch had 10 inputs and only 7 were submitted before a crash,
the returned `batch.jobs` list has 10 entries with `None` in the 3
unsubmitted slots.

:::{note}
Kinetic logs a warning when a reattached batch has fewer children than
expected, indicating partial submission.
:::

**Parameters:**

- **`group_id`** (`str`): The batch identifier (e.g., `"grp-a1b2c3d4"`).
- **`project`** (`str | None`, default `None`): GCP project. Uses the
  default when `None`.
- **`cluster`** (`str | None`, default `None`): GKE cluster name. Uses
  the default when `None`.

## Cleanup

There are two ways to clean up resources after a batch completes.

### Automatic cleanup via `results()`

By default, `results(cleanup=True)` deletes each child's Kubernetes
resources and GCS artifacts after downloading its result. The group
manifest is preserved, so `attach_batch()` still works.

```python
# Each child is cleaned up as its result is downloaded
results = batch.results()  # cleanup=True is the default
```

### Full teardown

To delete everything — all children's resources and the group manifest
itself — call `cleanup()` on the handle:

```python
batch.cleanup(k8s=True, gcs=True)
```

**Parameters:**

- **`k8s`** (`bool`, default `True`): Delete Kubernetes resources
  (Jobs/pods) for each child.
- **`gcs`** (`bool`, default `True`): Delete GCS artifacts for each
  child **and** the group manifest.

:::{important}
After calling `cleanup(gcs=True)`, the batch can no longer be reattached
via `attach_batch()` because the manifest has been deleted.
:::

## How it works

### Threading model

When `max_concurrent` is set (the default is 64) or `retries > 0`,
`kinetic.map()` launches a non-daemon background thread to manage
submissions. The thread polls active jobs for terminal states and
launches new ones as concurrency slots free up. The `BatchHandle` is
returned immediately.

When `max_concurrent=None` and `retries=0`, all jobs are submitted
synchronously in the calling thread before `map()` returns. No
background thread is created.

### Manifest

A JSON manifest is written to GCS before the first job is submitted. It
records the batch metadata (group ID, expected total, function name,
tags) and is updated after each successful submission with the child's
job ID and attempt count. This enables crash recovery: `attach_batch()`
reads the manifest to determine which jobs were submitted and
reconstructs the handle.

### Group ID

Each batch gets a unique identifier in the format `grp-{8-hex-chars}`
(e.g., `grp-a1b2c3d4`). This ID is set on each child `JobHandle` as
`group_id`, along with `group_kind="map"` and the child's `group_index`.

### Submission errors

If a call to `submit_fn` itself raises (e.g., a packaging or validation
error), the exception is captured internally and the corresponding slot
in `batch.jobs` remains `None`. These errors are surfaced when you call
`results()` — either as entries in the `BatchError.partial_results`
list or as exception objects when `return_exceptions=True`.

## Related pages

- [Detached Jobs](async_jobs.md) — the single-job `@kinetic.submit()`
  workflow each child of a batch is built on.
- [Cost Optimization](../guides/cost_optimization.md) — fan-out
  amplifies both throughput and spend; concurrency limits and spot
  instances matter here.
- [Checkpointing](../guides/checkpointing.md) — each child writes to
  its own `KINETIC_OUTPUT_DIR`; useful for long per-job work inside a
  batch.
- [Troubleshooting](../troubleshooting.md) — what to do when children
  stick in `PENDING` or repeatedly fail.
