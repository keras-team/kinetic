# FAQ

## When should I use `run()` vs `submit()`?

Use `@kinetic.run()` when you want your local script to wait for the
result. Use `@kinetic.submit()` when the job is long enough that you'd
rather get a `JobHandle` back, walk away, and reattach later. `submit()` is
the right call for anything multi-hour, anything you might want to monitor
from a different machine, or anything you want to fan out and check on in
parallel. See [Managing Async Jobs](../advanced/async_jobs.md).

## Why is the first run slower?

The first run with a given set of dependencies builds a container image via
Cloud Build (~2–5 minutes). The image is tagged by a hash of your
dependencies, so any subsequent run with the same `requirements.txt` reuses
the cached image and starts in under a minute. If your dependencies change,
the build re-runs. When the build cost becomes a bottleneck (for example,
when you change `requirements.txt` several times a day), switch to
**prebuilt mode**, which installs deps at pod startup instead of baking
them into a fresh image. See [Execution Modes](execution_modes.md) and
[Dependencies](dependencies.md).

## Should I use prebuilt or bundled mode?

Default to **bundled**. It is the only mode that works without first
publishing a base image. Reach for **prebuilt** when you change
`requirements.txt` several times a day and the per-iteration build cost is
hurting you. Prebuilt mode itself works with any base image at the
configured repo, but the kinetic project does not currently publish public
base images, so you will need to run `kinetic build-base` once to push your
own before this becomes a usable option. See [Execution Modes](execution_modes.md).

## When should I use `Data(...)` vs direct `gs://...` URIs?

Always prefer `kinetic.Data(...)`. It accepts both local paths and
`gs://` URIs and resolves to a plain filesystem path on the remote, so
your function only sees paths regardless of where the bytes started.
That is the whole point: one consistent API whether you are shipping a
local directory, pointing at an existing GCS bucket, or asking for a
FUSE mount via `Data(..., fuse=True)`. Reach for raw `gs://` URIs in
your code only if you specifically want to bypass the `Data` abstraction.
See [Data](data.md) for the decision matrix.

## How do I save checkpoints and outputs?

Write everything you want to keep under `KINETIC_OUTPUT_DIR`. Kinetic sets
this env var inside the job pod to a per-job GCS prefix. Anything you write
under it is durable: it outlives the pod and is reachable from your local
machine. The job's Python return value is for small results; outputs and
checkpoints belong on the output dir. See [Checkpointing](checkpointing.md).

## How do I reattach to a job?

Use `kinetic.attach(job_id)`. It reconstructs a `JobHandle` from the
metadata Kinetic persisted to GCS at submit time, so you can call
`.status()`, `.result()`, `.tail()`, or `.cleanup()` from any machine that
has Kinetic and your GCP credentials. The `job_id` is what `submit()`
returned originally. If you have lost it, `kinetic.list_jobs()` enumerates
jobs on the cluster. See [Managing Async Jobs](../advanced/async_jobs.md).

## What gets cleaned up automatically?

When a job succeeds, Kinetic removes its Kubernetes Job and pod by default,
so they don't pile up in the cluster. Failed jobs are kept around so you
can read logs and debug. GCS artifacts (uploaded code, requirements,
metadata) are _not_ auto-deleted; call `JobHandle.cleanup(gcs=True)` if you
want them gone. Outputs you wrote under `KINETIC_OUTPUT_DIR` are also kept
unless you explicitly delete them.

## How do spot instances affect training?

Spot capacity costs significantly less than on-demand, but pods can be
preempted with very little warning. Single-host jobs with frequent
checkpoints recover well. Multi-host TPU slices do not, because losing
any one host fails the whole slice. Use `--spot` for fault-tolerant
single-host workloads, and write checkpoints often enough to absorb a
restart. See [Cost Optimization](cost_optimization.md).

## When do I need multiple clusters?

Most users don't. Spin up a second cluster when you want to isolate GPU
and TPU workloads, run jobs in different regions, or separate dev from
prod environments. Each cluster has its own GKE control plane management
fee, so don't add them speculatively. See [Multiple Clusters](../advanced/clusters.md).

## What does Pathways mean in practice?

[Pathways](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro) is a JAX
runtime that coordinates execution across many TPU hosts. Concretely,
when you set `backend="pathways"` on a multi-host accelerator (e.g.,
`tpu-v5litepod-2x4`), Kinetic launches your job against a
Pathways-enabled cluster and JAX's collective communication (`jax.pmap`,
sharding, etc.) Just Works across hosts. Without Pathways, you would have
to manage multi-host JAX coordination yourself. See [Distributed Training](distributed_training.md).

## Glossary

**Accelerator**: A TPU or GPU type identifier (e.g., `tpu-v6e-8`, `l4`,
`a100`) passed to `accelerator=` on the decorator. Picks both the hardware
and the topology.

**Topology**: How many chips are arranged into the slice. For TPUs,
encoded in the accelerator name (`tpu-v6e-8` is 8 chips; `tpu-v5litepod-2x4`
is a 2×4 slice across hosts).

**Pathways**: JAX runtime for multi-host TPU coordination. Selected via
`backend="pathways"` and required for cross-host collectives without
hand-rolled setup.

**Node pool**: A GKE-managed group of VMs of one accelerator type.
Created with `kinetic pool add`. Scales between `--min-nodes` and the max
you need for the job.

**Cluster**: A GKE cluster with its own control plane and Artifact
Registry. Default name `kinetic-cluster`. Managed with `kinetic up`,
`kinetic down`, and `kinetic status`.

**Bundled image**: A container image Kinetic builds for you via Cloud
Build, with your dependencies baked in. The default execution mode. Tagged
by a hash of your `requirements.txt`.

**Prebuilt image**: A published base image that already has the
accelerator runtime installed. Your project deps are installed at pod
startup. Selected with `container_image="prebuilt"`. Requires you to
publish base images with `kinetic build-base` first.

**FUSE**: Filesystem-in-userspace mount. With `kinetic.Data(..., fuse=True)`,
a GCS bucket is mounted lazily into the pod's filesystem so reads stream
on demand instead of downloading up front.

**Handle**: A `JobHandle` returned by `kinetic.submit()` (or
`kinetic.attach()`). Wraps `status()`, `result()`, `tail()`, and
`cleanup()` for one job.

**Output dir**: The GCS prefix at `KINETIC_OUTPUT_DIR` inside the job
pod. The canonical place to write checkpoints and any files you want to
keep after the pod exits.

## Related pages

- [Execution Modes](execution_modes.md): bundled vs prebuilt vs custom.
- [Troubleshooting](../troubleshooting.md): symptom-first debugging.
- [Getting Started](../getting_started.md): your first run, end-to-end.
