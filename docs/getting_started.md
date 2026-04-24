# Getting Started

Install Kinetic, point it at a cluster, and run your first remote
function. If your team has already provisioned a Kinetic cluster, skip
ahead to [Run your first job](#run-your-first-job).

## Prerequisites

- Python 3.11+.
- [uv](https://docs.astral.sh/uv/getting-started/installation/), used
  for the install command below.
- Google Cloud SDK (`gcloud`): [install guide](https://cloud.google.com/sdk/docs/install).
- A Google Cloud project with [billing enabled](https://docs.cloud.google.com/billing/docs/how-to/modify-project).

Authenticate with Google Cloud once:

```bash
gcloud auth login
gcloud auth application-default login
```

## Install

```bash
uv pip install keras-kinetic
```

This installs both the `@kinetic.run()` decorator and the `kinetic`
CLI for managing infrastructure.

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

- **Join** — if your shell has previously provisioned a cluster in
  this project, `init` lists it and configures `kubectl` for you.
- **Create** — if no local cluster is found, `init` calls
  `kinetic up` to enable APIs, provision a GKE cluster with an
  accelerator node pool, and wire up Docker / `kubectl` access.

Either way, `init` ends by saving a **profile** (a named bundle of
project, zone, cluster, and namespace) and setting it as active. Every
subsequent `kinetic` command picks that up automatically — no
`export KINETIC_*` needed.

> **Joining a cluster someone else provisioned?** If the team cluster
> isn't in your local Pulumi state yet, `init` won't find it. Create
> the profile directly:
>
> ```bash
> kinetic profile create team-prod \
>   --project my-proj --zone us-central1-a --cluster team-cluster
> kinetic profile use team-prod
> ```

To run `init` non-interactively, you can still pre-set the project:

```bash
kinetic init --project=my-project --yes
```

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
