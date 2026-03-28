# Getting Started

## Prerequisites

- Python 3.11+
- Google Cloud SDK (`gcloud`) — [install guide](https://cloud.google.com/sdk/docs/install)
- A Google Cloud project with billing enabled

Authenticate with Google Cloud:

```bash
gcloud auth login
gcloud auth application-default login
```

> **Note:** The Pulumi CLI (used for infrastructure provisioning) is bundled and managed automatically. It will be installed to `~/.kinetic/pulumi` on first use if not already present.

## Install

```bash
git clone https://github.com/keras-team/kinetic.git
cd kinetic
pip install -e ".[cli]"
```

This installs both the `@kinetic.run()` decorator and the `kinetic` CLI for managing infrastructure.

> If your GKE cluster and Artifact Registry are already provisioned, you can install without the CLI: `pip install -e .`

## Provision Infrastructure

Run the one-time setup to create the required cloud resources:

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

> **Cleanup reminder:** When you're done, run `kinetic down` to tear down all resources and avoid ongoing charges. See [CLI Commands](cli.rst).

## Configure

Set your project ID so the library knows where to run jobs:

```bash
export KINETIC_PROJECT="your-project-id"
```

Add this to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) to persist it. See :doc:`usage` for the full list of environment variables.

## Run Your First Job

```python
import kinetic

@kinetic.run(accelerator="v6e-8")
def hello_tpu():
    import jax
    return f"Running on {jax.devices()}"

result = hello_tpu()
print(result)
```

> **First run timing:** The initial execution takes longer (~5 minutes) because it builds a container image with your dependencies. Subsequent runs with unchanged dependencies use the cached image and start in less than a minute.
