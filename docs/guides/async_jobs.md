# Detached Jobs

Most Kinetic users start with `@kinetic.run()`, which blocks the local
process until the remote function returns. That's the right choice when the
job is short, when you want the result inline in your script, or when
you're iterating on code interactively.

When the job is **long**, when you want to **walk away from your laptop**,
or when you want to **fan out and check on multiple jobs in parallel**,
switch to `@kinetic.submit()`. It returns a `JobHandle` immediately and
leaves the actual work running on the cluster. You can then poll status,
tail logs, collect results, or reattach to the job from a different machine
— all backed by metadata Kinetic persisted to GCS at submit time.

This page covers the full submit → observe → collect → cleanup loop, both
from Python and from the `kinetic jobs` CLI.

## A first detached job

```python
import kinetic

@kinetic.submit(accelerator="tpu-v5e-1")
def train_model():
    # Long-running training code
    return {"final_loss": 0.123}

job = train_model()
print(f"Submitted: {job.job_id}")

# ... do something else, possibly close the script entirely ...

final = job.result(timeout=3600)  # blocks until done
print(final)
```

`@kinetic.submit()` accepts the same arguments as `@kinetic.run()` —
accelerator, project, zone, cluster, container_image, env vars, data
volumes, etc. The only difference is what the call returns.

## Python and CLI side by side

Every operation is available both as a `JobHandle` method and as a
`kinetic jobs` subcommand. Pick whichever fits your workflow.

Operation        | Python                            | CLI
---------------- | --------------------------------- | ----------------------------------------------
Submit           | `job = train_model()`             | (use the decorator from a script)
Reattach         | `job = kinetic.attach(job_id)`    | (pass `<id>` to any `kinetic jobs` subcommand)
List             | `kinetic.list_jobs()`             | `kinetic jobs list`
Check status     | `job.status()`                    | `kinetic jobs status <id>`
Tail logs        | `job.tail(n=100)`                 | `kinetic jobs logs <id> --tail 100`
Follow logs      | `job.logs(follow=True)`           | `kinetic jobs logs <id> --follow`
Wait for result  | `job.result(timeout=3600)`        | `kinetic jobs result <id> --timeout 3600`
Cancel           | `job.cancel()`                    | `kinetic jobs cancel <id>`
Clean up         | `job.cleanup(k8s=True, gcs=True)` | `kinetic jobs cleanup <id>`

## Job lifecycle

A submitted job moves through five states (defined as `JobStatus` in
`kinetic.job_status`):

```text
                  ┌──────────┐
   submit() ────▶ │ PENDING  │ ── pod is waiting on a node
                  └────┬─────┘
                       │ pod scheduled
                       ▼
                  ┌──────────┐
                  │ RUNNING  │ ── your function is executing
                  └────┬─────┘
              ┌────────┴────────┐
              ▼                 ▼
        ┌───────────┐     ┌──────────┐
        │ SUCCEEDED │     │  FAILED  │
        └───────────┘     └──────────┘

  NOT_FOUND ── the k8s resource no longer exists (cleaned up,
               or never registered)
```

What each state means and what to do:

- **PENDING** — Kubernetes has accepted the job but no pod is running yet.
  The cluster autoscaler may be provisioning a node; on a fresh accelerator
  pool this can take 2–5 minutes. *What to do:* wait. If it's stuck for
  much longer, check `kinetic doctor` and your accelerator quota.
- **RUNNING** — your function is executing inside the pod. Use
  `job.tail()` or `kinetic jobs logs --follow` to watch progress. *What to
  do:* nothing, unless you want to monitor.
- **SUCCEEDED** — your function returned normally and Kinetic uploaded the
  result. *What to do:* call `job.result()` to get the return value. By
  default this also cleans up the k8s resource and GCS artifacts.
- **FAILED** — the pod exited non-zero. The k8s resource is *not*
  auto-deleted so you can read logs. *What to do:* `job.tail()` or
  `kinetic jobs logs <id>` to see the error, then `job.cleanup()` when
  you're done debugging.
- **NOT_FOUND** — the Kubernetes Job has already been deleted (typically
  by a successful `result()` call, or by an explicit `cleanup`). If the
  result was uploaded to GCS, `result()` can still return it; otherwise
  this state means the job is truly gone. *What to do:* if you need the
  return value, call `result()` once — it will read from GCS even after
  the pod is gone. If `result()` raises, the job is unrecoverable.

The full submit-to-cleanup flow:

1. `submit()` packages your code, builds (or reuses) a container image,
   uploads artifacts to GCS, creates a k8s Job, and returns a `JobHandle`.
   Status is `PENDING`.
2. The cluster autoscaler provisions a node if needed; the pod is
   scheduled. Status moves to `RUNNING`.
3. Your function runs. The pod uploads its return value (or an exception
   payload) to GCS when it exits.
4. Status moves to `SUCCEEDED` or `FAILED`.
5. Calling `job.result()` downloads the payload, returns it (or raises
   the user exception), and — by default — deletes both the k8s resource
   and the GCS artifacts. Status is now `NOT_FOUND` and the handle is
   spent.

## Reattaching from another machine

The `JobHandle` is a small JSON-serializable dataclass that Kinetic
persists to GCS at submit time. Anywhere you have Kinetic installed and
GCP credentials for the same project, you can reconstruct it from the
job ID:

```python
import kinetic

job = kinetic.attach("v5e1-train-model-20260417-153012-abc1234")
print(f"Status: {job.status().value}")
print(job.tail(n=20))
```

If you don't remember the ID, list everything currently on the cluster:

```python
for j in kinetic.list_jobs():
    print(f"{j.job_id}  {j.func_name}  {j.status().value}")
```

The CLI equivalent is `kinetic jobs list`.

## Timeouts and cleanup

`result()` blocks indefinitely by default. Pass `timeout=` (in seconds) to
bound the wait:

```python
try:
    final = job.result(timeout=3600)
except TimeoutError:
    # Job is still running — handle is still valid; you can call .result()
    # again, .tail(), .cancel(), or just walk away.
    print(job.tail(n=50))
```

By default `result()` cleans up after success: the k8s Job/pod and the
GCS artifacts are deleted. Two ways to opt out:

```python
final = job.result(cleanup=False)  # keep everything
job.cleanup(k8s=True, gcs=False)   # later: delete pod, keep artifacts
```

Failed jobs are not auto-cleaned, so logs survive until you delete them.
Anything you wrote under `KINETIC_OUTPUT_DIR` is also kept regardless of
cleanup — see [Checkpointing](../guides/checkpointing.md).

## Recommendations for long-running jobs

The following practices reduce the cost of failures on jobs that run for
hours.

- **Checkpoint regularly.** Anything written to `KINETIC_OUTPUT_DIR`
  survives a failed pod, but only the checkpoints already written can be
  used on resume. Pick a cadence that bounds how much progress a restart
  would lose. See [Checkpointing](../guides/checkpointing.md) for resume
  patterns.
- **Persist the `job_id`.** Record it via stdout, a log file, or your
  workflow's tracking system. With the ID, you can reattach from any
  machine that has Kinetic installed and access to the same GCP project.
- **Do not rely on the local Python process.** Once `submit()` returns,
  the local script is no longer involved in the job's execution.
  Interrupting it (for example, with `Ctrl-C`) does not affect the
  remote job.
- **Avoid `--follow` for jobs that run for hours.** Continuous log
  streaming is sensitive to transient network failures. Use
  `kinetic jobs logs <id> --tail 200` from a fresh shell to check in
  periodically instead.
- **Retain artifacts on multi-host or expensive jobs.** Pass
  `cleanup=False` to the first successful `result()` call so the
  Kubernetes resources and GCS artifacts remain available for
  inspection. Call `cleanup` explicitly once they are no longer needed.

## Related pages

- [Checkpointing](../guides/checkpointing.md) — make long jobs resumable.
- [Cost Optimization](../guides/cost_optimization.md) — spot instances and
  scale-to-zero behavior for detached workloads.
- [Troubleshooting](../troubleshooting.md) — what to do when a job is
  stuck in `PENDING` or repeatedly failing.
