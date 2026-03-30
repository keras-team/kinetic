# Getting Started

## Prerequisites

- Python 3.11+
- Google Cloud SDK (`gcloud`) — [install here](https://cloud.google.com/sdk/docs/install)
- A Google Cloud project with [billing enabled](https://docs.cloud.google.com/billing/docs/how-to/modify-project)

Authenticate with Google Cloud:

```bash
gcloud auth login
gcloud auth application-default login
```

Set your GCP project ID so the library knows where to run jobs:

```bash
export KINETIC_PROJECT="your-project-id"
```

Add this to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) to persist it. See :doc:`guides/configuration` for the full list of environment variables.

## Install

```bash
pip install keras-kinetic
```

This installs both the `@kinetic.run()` decorator and the `kinetic` CLI for managing infrastructure.

> **Note:** The [Pulumi](https://www.pulumi.com/) CLI (used for infrastructure
> provisioning) is bundled and managed automatically. It will be installed to
> `~/.kinetic/pulumi` on first use if not already present.

## Provision Infrastructure

If you are not the first Kinetic user in your org, you can skip this section.
Otherwise, run the one-time setup step to create a cluster for Kinetic:

```bash
kinetic up
```

This interactively prompts for your GCP project and accelerator type, then:

- Enables required APIs (Cloud Build, Artifact Registry, Cloud Storage, GKE)
- Creates an Artifact Registry repository for container images
- Provisions a GKE cluster with an accelerator node pool
- Configures Docker authentication and kubectl access

You can also run non-interactively:

```bash
kinetic up --project=my-project --accelerator=t4 --yes
```

> **Cleanup reminder:** When you're done, run `kinetic down` to tear down all resources and avoid ongoing charges. See [CLI Command here](cli.rst#kinetic-down).

## Run Your First Job

```{literalinclude} ../examples/fashion_mnist.py
    :language: python
```

> **First run timing:** The initial execution takes longer (~5 minutes) because
> it builds a container image with your dependencies. Subsequent runs with
> unchanged dependencies use the cached image and start in less than a minute.

Once this works, you can try:

- More complex examples like [fine-tuning LLMs](guides/llm_finetuning.md)
- Running on different [accelerators and topologies](accelerators.md)
