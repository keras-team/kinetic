# Troubleshooting

This page is organized by symptom rather than by error message. Locate
the section that best matches what you are observing and follow the
guidance there.

For a quick diagnostic of common environment problems, run:

```bash
kinetic doctor
```

It checks for missing tools, misconfigured credentials, and unhealthy
infrastructure, and prints a concrete fix command for each failed check.
The full list of categories it covers is described at the end of this
page.

## Startup and build issues

### "Project must be specified"

`KINETIC_PROJECT` (or `GOOGLE_CLOUD_PROJECT`) is not set. Set it once
in your shell profile:

```bash
export KINETIC_PROJECT="your-project-id"
```

Or pass `project=` to the decorator. See [Configuration](configuration.md).

### "404 Requested entity was not found"

A required GCP resource — usually an Artifact Registry repository or a
GKE cluster — doesn't exist yet. Run the setup once:

```bash
kinetic up
```

Or, if `up` already ran, enable the missing APIs and create the
registry manually (uncommon):

```bash
gcloud services enable compute.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com storage.googleapis.com \
    container.googleapis.com --project=$KINETIC_PROJECT

gcloud artifacts repositories create "kn-${KINETIC_CLUSTER:-kinetic-cluster}" \
    --repository-format=docker --location=us \
    --project=$KINETIC_PROJECT
```

### Container build is slow on first run

The first run with a given `requirements.txt` builds a new container
image via Cloud Build (~2–5 minutes). Subsequent runs reuse the cached
image and start in under a minute. If you're churning dependencies
multiple times a day and this is hurting you, see
[Execution Modes](guides/execution_modes.md) for prebuilt mode.

### Container build failures

Check Cloud Build logs:

```bash
gcloud builds list --project=$KINETIC_PROJECT --limit=5
gcloud builds log <build-id> --project=$KINETIC_PROJECT
```

Common causes: a package in `requirements.txt` that doesn't exist,
network issues during install, or a base image that's been updated
since you last built. See [Dependencies](guides/dependencies.md).

## Auth and config issues

### "Permission denied" on GCP operations

Your user (or service account) is missing IAM roles. The minimum set
for Kinetic is `roles/storage.admin`, `roles/artifactregistry.admin`,
`roles/container.admin`, and `roles/cloudbuild.builds.editor` on the
project:

```bash
gcloud projects add-iam-policy-binding $KINETIC_PROJECT \
    --member="user:your-email@example.com" \
    --role="roles/storage.admin"
```

Repeat for the other roles. `kinetic doctor` flags missing roles by
checking the actual operations that fail.

### Application Default Credentials missing or expired

```bash
gcloud auth login
gcloud auth application-default login
```

If you've previously set `GOOGLE_APPLICATION_CREDENTIALS` to a service
account key, that takes precedence over user ADC.

### Settings aren't taking effect

Run `kinetic config` — it prints every config value and where it came
from (decorator arg, CLI flag, env var, or default). The precedence
rules are documented in [Configuration](configuration.md).

## Scheduling and quota issues

### Job stuck in `PENDING` for more than 10 minutes

The cluster autoscaler is trying to provision a node but can't. Two
common reasons:

- **No quota for the requested accelerator** in your zone. Check
  Cloud Console → IAM & Admin → Quotas, filter by your accelerator
  type. If quota is exhausted, request more or try a different zone.
- **Spot capacity is unavailable.** If your node pool was created with
  `--spot`, GCP may have no spot capacity to allocate right now.
  Switch to on-demand or try later.

`kinetic doctor` includes a quota check that surfaces exhausted
accelerator quotas in your region. If it doesn't flag anything, inspect
the Cloud Console quota page directly for finer-grained breakdowns.

### Multi-host TPU job fails right after submit

Likely causes: topology mismatch (your code expected a different number
of devices than the slice has), a stale Pathways context from a prior
crashed job, or one host failing before the others can join the
collective. See [Distributed Training](guides/distributed_training.md)
for the full list of multi-host failure modes.

## Runtime failures

### `ImportError` on the remote pod

The package isn't in your `requirements.txt` or `pyproject.toml`. A
local `pip install` doesn't carry over — only what's in those files
gets installed. See [Dependencies](guides/dependencies.md) for the
common pitfalls list.

### Pickle / cloudpickle errors at submit time

The function or one of its closures references something that can't be
serialized — typically an open file handle, a database connection, or
a module-level singleton initialized for local use. Move that
initialization inside the decorated function.

### JAX version mismatch errors

You probably pinned `jax` or `jaxlib` in `requirements.txt`. Kinetic
filters those out by default; if you need a specific version, use
`# kn:keep` (see [Dependencies](guides/dependencies.md)), but expect
to debug runtime/library alignment yourself.

### Job FAILS but logs look fine

The pod exited non-zero without writing a result payload — usually
caused by an OOM kill or the kernel reaping the process. Check pod
events with `kubectl describe pod <pod-name>` (find the pod name from
`kinetic jobs status <id>`).

## Missing outputs and results

### `result()` raises "result payload not found"

The job either never produced one (it crashed before finishing), or
it was already cleaned up. Failed jobs don't write a result payload.
For long jobs, prefer writing artifacts under `KINETIC_OUTPUT_DIR`
instead of relying on the return value — see [Checkpointing](guides/checkpointing.md).

### Files I wrote inside the job are gone

Two possibilities:

- You wrote them under `/tmp` or another pod-local path. The pod is
  destroyed when the job ends; pod-local files don't survive. Always
  write to `KINETIC_OUTPUT_DIR`.
- You wrote them under `KINETIC_OUTPUT_DIR` but more than 30 days have
  passed. The default GCS bucket has a 30-day TTL. Copy critical
  artifacts to a bucket without lifecycle rules. See
  [Checkpointing](guides/checkpointing.md) for the TTL and retention
  details.

### Logs aren't streaming back

Network blip during a `--follow` stream is the most common cause. The
pod is unaffected — log retrieval is read-only. Use
`kinetic jobs logs <id>` (without `--follow`) or `--tail N` to fetch
fresh logs from any machine.

## What `kinetic doctor` actually checks

`kinetic doctor` runs eight groups of checks and prints concrete fix
commands when any fail. The groups (matching the source at
`kinetic/cli/commands/doctor.py`):

1. **Local Tools** — `gcloud`, `kubectl`, and
   `gke-gcloud-auth-plugin` are installed and on your PATH.
2. **Authentication** — Application Default Credentials are present,
   refreshable, and not expired.
3. **Configuration** — `KINETIC_PROJECT`, `KINETIC_ZONE`, and
   `KINETIC_CLUSTER` resolve to non-empty values.
4. **GCP Project** — the project exists and has billing enabled.
5. **GCP APIs** — Compute Engine, Cloud Build, Artifact Registry,
   Storage, and Container APIs are enabled.
6. **GCP Resources** — the Kinetic service accounts, Artifact Registry
   repository, GCS buckets, VPC network, and Cloud NAT all exist.
7. **Infrastructure** — Pulumi state is present and the GKE cluster is
   in the `RUNNING` state.
8. **Kubernetes** — your `kubeconfig` points at the cluster, the API
   server responds, node pools are healthy, GPU drivers are installed
   where needed, and accelerator quotas are not exhausted.

Each failing check prints a one-line fix suggestion. For multi-step
fixes, `kinetic doctor` prints a copy-paste command block.

## Related pages

- [Getting Started](getting_started.md) — first-run setup that
  shouldn't have to fail twice.
- [FAQ](guides/faq.md) — quick answers to common conceptual confusions.
- [Configuration](configuration.md) — env vars and precedence.
