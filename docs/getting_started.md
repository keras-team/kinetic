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

Set your GCP project ID so Kinetic knows where to run jobs:

```bash
export KINETIC_PROJECT="your-project-id"
```

Add this to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) so it
persists. See [Configuration](configuration.md) for the full list of
environment variables.

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

## Are you the first user, or joining a team?

Two paths from here:

- **Joining an existing Kinetic team.** Someone else has already run
  `kinetic up`. Point your shell at the team's cluster and skip ahead
  to [Run your first job](#run-your-first-job):

  ```bash
  export KINETIC_CLUSTER="cluster-name"
  export KINETIC_ZONE="us-central1-a"  # if it differs from the default
  ```

- **First user in your project.** You need to provision a cluster once
  before you can run anything. Continue with the next step.

## Provision infrastructure (first user only)

Skip this section if your team already runs a Kinetic cluster (the
"joining a team" path above). Otherwise, run the one-time setup. It
interactively prompts for your GCP project and accelerator type:

```bash
kinetic up
```

This:

- Enables required APIs (Cloud Build, Artifact Registry, Cloud
  Storage, GKE).
- Creates an Artifact Registry repository for container images.
- Provisions a GKE cluster with an accelerator node pool.
- Configures Docker authentication and `kubectl` access.

You can also run non-interactively:

```bash
kinetic up --project=my-project --accelerator=t4 --yes
```

> **Cleanup reminder:** when you're done, run `kinetic down` to tear
> down all resources and stop incurring costs. See the
> [CLI Reference](cli) for the full set of commands.

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

- [Examples](guides/examples.md): a catalog of runnable scripts that
  cover async jobs, data, checkpoints, parallel sweeps, and LLM
  fine-tuning. The fastest way to see real patterns end to end.
- [Execution Modes](guides/execution_modes.md): bundled vs prebuilt
  vs custom image, and when to switch.
- [Detached Jobs](advanced/async_jobs.md): `@kinetic.submit()`,
  reattach, and the job lifecycle for long-running work.
- [Data](guides/data.md) and
  [Checkpointing](guides/checkpointing.md): `kinetic.Data(...)` for
  inputs and `KINETIC_OUTPUT_DIR` for durable outputs and resumable
  checkpoints.
