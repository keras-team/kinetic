# Getting Started

Install Kinetic, point it at a cluster, and run your first remote
function. If your team has already provisioned a Kinetic cluster, skip
ahead to [Run your first job](#run-your-first-job).

## Prerequisites

- Python 3.11+.
- [uv](https://docs.astral.sh/uv/getting-started/installation/), used
  for the install command below.
- Google Cloud SDK (`gcloud`): [install guide](https://cloud.google.com/sdk/docs/install).
- `kubectl`: [install guide](https://kubernetes.io/docs/tasks/tools/).
  Kinetic auto-installs the `gke-gcloud-auth-plugin` for you on first
  use, but `kubectl` itself must already be on your `PATH`.
- A Google Cloud project with [billing enabled](https://docs.cloud.google.com/billing/docs/how-to/modify-project).

Authenticate with Google Cloud once:

```bash
gcloud auth login
gcloud auth application-default login
```

## Install

```bash
uv pip install "keras-kinetic[cli]"
```

The base `keras-kinetic` package installs the `@kinetic.run()`
decorator. The `[cli]` extra adds the dependencies the `kinetic` CLI
needs to provision and manage infrastructure (Pulumi and the GCP
plugins). We recommend installing the `[cli]` extra even if you're only
submitting jobs against an already-provisioned cluster — the CLI makes
troubleshooting and job management much easier. Drop it only if you're
sure you won't need CLI access.

> **Note:** The [Pulumi](https://www.pulumi.com/) CLI (used for
> infrastructure provisioning) is bundled and managed automatically.
> It will be installed to `~/.kinetic/pulumi` on first use if not
> already present.

## Set up your environment

```bash
kinetic init
```

`kinetic init` checks your local tools, auth, and project, then routes
you down one of two paths:

- **Join** — if any Kinetic clusters already exist in this GCP project
  (provisioned by you or a teammate), `init` lists them, lets you pick
  one, and configures `kubectl` for it. Cluster discovery reads the
  project's shared state bucket
  (`gs://{project}-kinetic-state`), so collaborators with access to
  the bucket all see the same set.
- **Create** — if no clusters exist yet, `init` calls `kinetic up` to
  enable APIs, provision a GKE cluster with an accelerator node pool,
  and wire up Docker / `kubectl` access.

Either way, `init` ends by saving a **profile** and making it active.
A profile is your saved infrastructure context like project, zone,
cluster, and namespace - persisted at `~/.kinetic/profiles.json`. The
active profile is what every `kinetic` command and every
`@kinetic.run()` invocation targets, so you don't need to export env vars or pass `--project` / `--zone` / `--cluster` on
the command line. Switch contexts with `kinetic profile use <profile-name>`, and
see what's saved with `kinetic profile ls`.

> **Cleanup reminder:** when you're done, run `kinetic down` to tear
> down all resources and stop incurring costs. See the
> [CLI Reference](cli) for the full set of commands.

**Sharing infrastructure with teammates?** Kinetic stores Pulumi
state in a per-project GCS bucket (`gs://{project}-kinetic-state`),
so any teammate with `roles/storage.objectAdmin` on the bucket sees
the same stack. The first `kinetic up` creates the bucket; the first
admin needs `roles/storage.admin` on the project. See
[Pulumi state](configuration.md#pulumi-state) for the full IAM story.

## Run your first job

```{literalinclude} ../examples/fashion_mnist.py
    :language: python
```

Run it:

```bash
python fashion_mnist.py
```

:::{note}
**Expected timing:**

- **First run:** ~5 minutes. The slow part is the first container
  build via Cloud Build, which freezes your dependencies into an
  image tagged by their hash.
- **Subsequent runs (same dependencies):** under a minute. The
  cached image is reused; only your code changes get re-uploaded.
- **Subsequent runs (changed dependencies):** ~5 minutes again,
  since a new hash forces a fresh build.
  :::

:::{tip}
**Recommended defaults:**

- Stay in **bundled mode** (the default — you don't need to pass
  `container_image=`). It's the only mode that works without
  publishing your own base image.
- Use **`@kinetic.run()`** while you're iterating; switch to
  **`@kinetic.submit()`** once your jobs run for more than a few
  minutes and you'd rather not block your local shell.
- Write any artifacts you want to keep under `KINETIC_OUTPUT_DIR`,
  not under `/tmp`.
  :::

## Next steps

After your first run works, the most useful follow-ups are:

- [Examples](examples.md): a catalog of runnable scripts that
  cover async jobs, data, checkpoints, parallel sweeps, and LLM
  fine-tuning. The fastest way to see real patterns end to end.
- [Execution Modes](guides/execution_modes.md): bundled vs prebuilt
  vs custom image, and when to switch.
- [Detached Jobs](guides/async_jobs.md): `@kinetic.submit()`,
  reattach, and the job lifecycle for long-running work.
- [Data](guides/data.md) and
  [Checkpointing](guides/checkpointing.md): `kinetic.Data(...)` for
  inputs and `KINETIC_OUTPUT_DIR` for durable outputs and resumable
  checkpoints.
