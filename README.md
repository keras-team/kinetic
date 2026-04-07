# Kinetic

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Run Keras and JAX workloads on cloud TPUs and GPUs with a simple decorator. No infrastructure management required.

```python
import kinetic

@kinetic.run(accelerator="v5e-1")
def train_model():
    import keras
    model = keras.Sequential([...])
    model.fit(x_train, y_train)
    return model.history.history["loss"][-1]

# Executes on TPU v5e-1, returns the result
final_loss = train_model()
```

## Table of Contents

- [How It Works](#how-it-works)
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage Guide](#usage-guide)
- [Reference](#reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## How It Works

When you call a decorated function, Kinetic handles the entire remote execution pipeline:

1. **Packages** your function, local code, and data dependencies
2. **Builds a container** with your dependencies via Cloud Build (cached after first build — subsequent runs skip this step)
3. **Runs the job** on a GKE cluster with the requested accelerator (TPU or GPU)
4. **Returns the result** to your local machine — logs are streamed in real time, and the function's return value is delivered back as if it ran locally

If the remote function raises an exception, it is re-raised locally with the original traceback, so debugging works the same as local development.

You need a GKE cluster with accelerator node pools to run jobs. The `kinetic` CLI handles this setup for you.

## Features

- **Simple decorator API** — Add `@kinetic.run()` to any function to execute it remotely
- **Detached execution** — Use `@kinetic.submit()` to launch work, reattach later, and collect results when convenient
- **Automatic infrastructure** — No manual VM provisioning or teardown required
- **Result serialization** — Functions return actual values, not just logs
- **Fast iteration** — Container images are cached by dependency hash; unchanged dependencies skip the build entirely (subsequent runs start in less than a minute)
- **Data API** — Declarative `Data` class with smart caching for local files and GCS data
- **Environment variable forwarding** — Propagate local env vars to the remote environment with wildcard patterns (`capture_env_vars=["KAGGLE_*"]`)
- **Built-in monitoring** — View job status and logs in Google Cloud Console
- **Automatic cleanup** — Resources are released when jobs complete
- **Transparent errors** — Remote exceptions are re-raised locally with the original traceback

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package installer
- Google Cloud SDK (`gcloud`) — [install guide](https://cloud.google.com/sdk/docs/install)
- A Google Cloud project with billing enabled

Authenticate with Google Cloud:

```bash
gcloud auth login
gcloud auth application-default login
```

> **Note:** The Pulumi CLI (used for infrastructure provisioning) is bundled and managed automatically. It will be installed to `~/.kinetic/pulumi` on first use if not already present.

### Install

```bash
uv pip install keras-kinetic
```

This installs both the `@kinetic.run()` decorator and the `kinetic` CLI for managing infrastructure.

### Provision Infrastructure

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

> **Cleanup reminder:** When you're done, run `kinetic down` to tear down all resources and avoid ongoing charges. See [CLI Commands](#cli-commands).

### Configure

Set your project ID so the library knows where to run jobs:

```bash
export KINETIC_PROJECT="your-project-id"
```

Add this to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) to persist it. See [Configuration](#configuration) for the full list of environment variables.

### Run Your First Job

```python
import kinetic

@kinetic.run(accelerator="v5e-1")
def hello_tpu():
    import jax
    return f"Running on {jax.devices()}"

result = hello_tpu()
print(result)
```

> **First run timing:** The initial execution takes longer (~5 minutes) because it builds a container image with your dependencies. Subsequent runs with unchanged dependencies use the cached image and start in less than a minute.

## Usage Guide

### Training a Keras Model

```python
import kinetic

@kinetic.run(accelerator="v5e-1")
def train_model():
    import keras
    import numpy as np

    model = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(10,)),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    x_train = np.random.randn(1000, 10)
    y_train = np.random.randn(1000, 1)

    history = model.fit(x_train, y_train, epochs=5, verbose=0)
    return history.history["loss"][-1]

final_loss = train_model()
print(f"Final loss: {final_loss}")
```

### Async Jobs

`@kinetic.run()` blocks until the remote function finishes and returns the result inline — convenient for interactive work, but limiting when jobs take hours or you want to launch several in parallel. `@kinetic.submit()` is the async counterpart: it launches the job, returns a `JobHandle` immediately, and lets you check status, stream logs, collect the result, or reattach from a completely different session whenever you're ready.

The two decorators accept the same parameters, so switching between them is a one-word change.

#### Submitting a Job

```python
import kinetic
import time

@kinetic.submit(accelerator="v5e-1")
def train_model():
    time.sleep(60)
    return {"loss": 0.1}

# Returns a JobHandle immediately — does not wait for the job to finish
job = train_model()
print(job.job_id)       # e.g. "job-a1b2c3d4"
print(job.func_name)    # "train_model"
print(job.accelerator)  # "v6e-8"
```

#### Monitoring Status and Logs

```python
# Poll the current status
status = job.status()  # PENDING → RUNNING → SUCCEEDED or FAILED
print(status.value)

# Grab the last N log lines from the active pod
print(job.tail(20))

# Or fetch the full log as a string
full_log = job.logs()

# Stream logs in real time (blocks until the job finishes)
job.logs(follow=True)
```

`JobStatus` values: `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`, `NOT_FOUND`.

#### Collecting the Result

```python
# Block until the job completes and return its value
metrics = job.result()   # {"loss": 0.1}

# With a timeout (raises TimeoutError if exceeded, but the job keeps running)
metrics = job.result(timeout=600)
```

By default, `result()` cleans up the Kubernetes resource and GCS artifacts after downloading the return value. Pass `cleanup=False` to keep them around for inspection.

If the remote function raises an exception, `result()` re-raises it locally with the original traceback attached.

#### Running Multiple Jobs Concurrently

Because `submit()` returns immediately, you can launch several jobs and collect results later:

```python
@kinetic.submit(accelerator="cpu")
def train_model_a():
    ...

@kinetic.submit(accelerator="cpu")
def train_model_b():
    ...

job_a = train_model_a()
job_b = train_model_b()

# Both are running in parallel — collect when ready
loss_a = job_a.result(cleanup=False)
loss_b = job_b.result(cleanup=False)
```

#### Reattaching to a Job

You can reconnect to a running (or finished) job from a different Python session, shell, or machine — all you need is the job ID and access to the same GCP project:

```python
job = kinetic.attach(
    job_id="job-a1b2c3d4",
    project="my-project",   # optional if KINETIC_PROJECT is set
    cluster="kinetic-cluster",  # optional if KINETIC_CLUSTER is set
)

print(job.status().value)
metrics = job.result()
```

#### Listing Jobs

Discover all live Kinetic jobs on the cluster:

```python
for job in kinetic.list_jobs(project="my-project", cluster="kinetic-cluster"):
    print(job.job_id, job.func_name, job.status().value)
```

Both `project` and `cluster` are optional when the corresponding environment variables are set.

#### Cancelling and Cleaning Up

```python
# Cancel a running job (deletes the Kubernetes resource)
job.cancel()

# Explicitly clean up Kubernetes resources and/or GCS artifacts
job.cleanup(k8s=True, gcs=True)
```

#### CLI

Async jobs can also be managed from the terminal — see [`kinetic jobs`](#kinetic-jobs) in the CLI reference below.

### Working with Data

Kinetic provides a declarative Data API to seamlessly make your local and cloud data available to remote functions.

The Data API is read-only — it delivers data to your pods at the start of a job. For saving model outputs or checkpointing, write directly to GCS from within your function.

Under the hood, the Data API provides two key optimizations:

- **Smart Caching:** Local data is content-hashed and uploaded to a cache bucket only once. Subsequent job runs with byte-identical data skip the upload entirely.
- **Automatic Zip Exclusion:** When you reference a data path inside your current working directory, Kinetic automatically excludes that directory from the project's zipped payload to avoid uploading the same data twice.

There are three approaches depending on your workflow:

#### Dynamic Data (The `Data` Class)

The simplest approach — pass `Data` objects as regular function arguments. The `Data` class wraps a local file/directory path or a Google Cloud Storage (GCS) URI.

On the remote pod, these objects are automatically resolved into plain string paths pointing to the downloaded files, so your function code never needs to know about GCS or cloud storage APIs.

```python
import pandas as pd
import kinetic
from kinetic import Data

@kinetic.run(accelerator="v5e-1")
def train(data_dir):
    # data_dir is resolved to a local path on the remote machine
    df = pd.read_csv(f"{data_dir}/train.csv")
    # ...

# Uploads the local directory to the remote pod automatically
train(Data("./my_dataset/"))

# Cache hit: subsequent runs with the same data skip the upload!
train(Data("./my_dataset/"))
```

> **GCS Directories:** When referencing a GCS directory with the `Data` class, include a trailing slash (e.g., `Data("gs://my-bucket/dataset/")`). Without the trailing slash, the system treats it as a single file object.

You can also pass multiple `Data` arguments, or nest them inside lists and dictionaries (e.g., `train(datasets=[Data("./d1"), Data("./d2")])`).

#### Static Data (The `volumes` Parameter)

For established training scripts where data requirements are fixed, use the `volumes` parameter in the decorator. This mounts data at hardcoded absolute filesystem paths, allowing you to use Kinetic with existing codebases without altering the function signature.

```python
import pandas as pd
import kinetic
from kinetic import Data

@kinetic.run(
    accelerator="v5e-1",
    volumes={
        "/data": Data("./my_dataset/"),
        "/weights": Data("gs://my-bucket/pretrained-weights/")
    }
)
def train():
    # Data is guaranteed to be available at these absolute paths
    df = pd.read_csv("/data/train.csv")
    model.load_weights("/weights/model.h5")
    # ...

# No data arguments needed!
train()
```

#### Direct GCS Streaming (For Large Datasets)

If your dataset is very large (e.g., > 10GB), it is inefficient to download the entire dataset to the pod's local disk. Instead, skip the `Data` wrapper and pass a GCS URI string directly. Use frameworks with native GCS streaming support (like `tf.data` or `grain`) to read the data on the fly.

```python
import grain.python as grain
import kinetic

@kinetic.run(accelerator="v5e-1")
def train(data_uri):
    # Native GCS reading, no download overhead
    data_source = grain.ArrayRecordDataSource(data_uri)
    # ...

# Pass as a plain string, no Data() wrapper needed
train("gs://my-bucket/arrayrecords/")
```

### Custom Dependencies

Create a `requirements.txt` in your project directory:

```text
tensorflow-datasets
pillow
scikit-learn
```

Alternatively, dependencies declared in `pyproject.toml` are also supported:

```toml
[project]
dependencies = [
    "tensorflow-datasets",
    "pillow",
    "scikit-learn",
]
```

Kinetic automatically detects and installs dependencies on the remote worker.
If both files exist in the same directory, `requirements.txt` takes precedence.

> **Note:** JAX packages (`jax`, `jaxlib`, `libtpu`, `libtpu-nightly`) are automatically filtered from your dependencies to prevent overriding the accelerator-specific JAX installation. To keep a JAX line, append `# kn:keep` to it.

### Container Image Modes

Kinetic supports three container image modes controlled by the `container_image` parameter.

#### Bundled Mode (Default)

The default mode builds a custom container image via Cloud Build with all dependencies baked in. Images are cached by dependency hash — unchanged dependencies reuse the cached image.

```python
# These are equivalent — both use bundled mode:
@kinetic.run(accelerator="v5e-1")
def train():
    ...

@kinetic.run(accelerator="v5e-1", container_image="bundled")
def train():
    ...
```

> **First run timing:** The initial build takes ~2-5 minutes. Subsequent runs with unchanged dependencies start within a few seconds.

To use your own prebuilt images instead of the official ones, build and push them with `kinetic build-base`, then point Kinetic at your repository:

```bash
kinetic build-base --repo us-docker.pkg.dev/my-project/kinetic-base
export KINETIC_BASE_IMAGE_REPO=us-docker.pkg.dev/my-project/kinetic-base
```

Or pass `base_image_repo` directly to the decorator:

```python
@kinetic.run(accelerator="l4", base_image_repo="us-docker.pkg.dev/my-project/kinetic-base")
def train():
    ...
```

#### Prebuilt Mode

Prebuilt mode uses a pre-published base image with the accelerator runtime already installed. Your project dependencies are installed at pod startup via `uv pip install` — no Cloud Build step required.

```python
@kinetic.run(accelerator="v5e-1", container_image="prebuilt")
def train():
    ...
```

#### Custom Image Mode

Provide a full container image URI to use your own image. Kinetic skips all build and dependency steps. The image must include `cloudpickle`, `google-cloud-storage`, and a compatible Python environment, and be accessible from the GKE nodes.

```python
@kinetic.run(
    accelerator="v5e-1",
    container_image="us-docker.pkg.dev/my-project/kinetic/my-image:v1.0"
)
def train():
    ...
```

For more details on each mode, see the [container images documentation](docs/advanced/containers.md).

### Forwarding Environment Variables

Use `capture_env_vars` to propagate local environment variables to the remote pod. This supports exact names and wildcard patterns:

```python
import kinetic

@kinetic.run(
    accelerator="v5litepod-1",
    capture_env_vars=["KAGGLE_*", "GOOGLE_CLOUD_*"]
)
def train_gemma():
    import keras_hub
    gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_1b")
    # KAGGLE_USERNAME and KAGGLE_KEY are available for model downloads
    # ...
```

This is useful for forwarding API keys, credentials, or configuration without hardcoding them.

### Multi-Host TPU (Pathways)

Multi-host TPU configurations (those requiring more than one node, such as `v2-16`, `v3-32`, or `v5p-16`) automatically use the [Pathways](https://cloud.google.com/tpu/docs/pathways-overview) backend. You can also set the backend explicitly:

```python
@kinetic.run(accelerator="v3-32", backend="pathways")
def distributed_train():
    ...
```

### Multiple Clusters

You can run multiple independent clusters within the same GCP project — for example, one for GPU workloads and another for TPUs. Each cluster gets its own isolated set of cloud resources (GKE cluster, Artifact Registry, storage buckets) backed by a separate infrastructure stack, so they never interfere with each other.

**Create clusters** by passing `--cluster` to `kinetic up`:

```bash
# Default cluster (named "kinetic-cluster")
kinetic up --project=my-project --accelerator=v5e-1

# A separate GPU cluster
kinetic up --project=my-project --cluster=gpu-cluster --accelerator=a100
```

**Target a cluster** in your code with the `cluster` parameter or the `KINETIC_CLUSTER` environment variable:

```python
# Run on the GPU cluster
@kinetic.run(accelerator="a100", cluster="gpu-cluster")
def train_on_gpu():
    ...

# Or set the env var to avoid repeating the cluster name
# export KINETIC_CLUSTER="gpu-cluster"
@kinetic.run(accelerator="a100")
def train_on_gpu():
    ...
```

All CLI commands accept `--cluster` as well, so you can manage each cluster independently:

```bash
kinetic status --cluster=gpu-cluster
kinetic pool add --cluster=gpu-cluster --accelerator=h100
kinetic down --cluster=gpu-cluster
```

For more examples, see the [`examples/`](examples/) directory.

## Reference

### Configuration

#### Environment Variables

| Variable                  | Required | Default           | Description                                                  |
| ------------------------- | -------- | ----------------- | ------------------------------------------------------------ |
| `KINETIC_PROJECT`         | Yes      | —                 | Google Cloud project ID                                      |
| `KINETIC_ZONE`            | No       | `us-central1-a`   | Default compute zone                                         |
| `KINETIC_CLUSTER`         | No       | `kinetic-cluster` | GKE cluster name                                             |
| `KINETIC_BASE_IMAGE_REPO` | No       | `kinetic`         | Docker repository for prebuilt base images                   |
| `KINETIC_NAMESPACE`       | No       | `default`         | Kubernetes namespace                                         |
| `KINETIC_LOG_LEVEL`       | No       | `INFO`            | Log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `FATAL`) |

Kinetic uses `absl-py` for logging. Set `KINETIC_LOG_LEVEL=DEBUG` for verbose output when debugging issues.

#### Decorator Parameters

```python
@kinetic.run(
    accelerator="v5e-1",       # TPU/GPU type (default: "v5e-1")
    container_image=None,      # None/"bundled", "prebuilt", or a custom image URI
    base_image_repo=None,      # Prebuilt image repo (default: KINETIC_BASE_IMAGE_REPO)
    zone=None,                 # Override default zone
    project=None,              # Override default project
    capture_env_vars=None,     # Env var names/patterns to forward (supports * wildcard)
    cluster=None,              # GKE cluster name
    backend=None,              # "gke", "pathways", or None (auto-detect)
    namespace="default",       # Kubernetes namespace
    volumes=None,              # Dict mapping absolute paths to Data objects
)
```

### Supported Accelerators

Each accelerator and topology requires [setting up its own node pool](#kinetic-pool) as a prerequisite.

#### TPUs

| Type           | Configurations                                                                                                                |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| TPU v3         | `v3-4`, `v3-16`, `v3-32`, `v3-64`, `v3-128`, `v3-256`, `v3-512`, `v3-1024`, `v3-2048`                                         |
| TPU v4         | `v4-4`, `v4-8`, `v4-16`, `v4-32`, `v4-64`, `v4-128`, `v4-256`, `v4-512`, `v4-1024`, `v4-2048`, `v4-4096`                      |
| TPU v5 Litepod | `v5litepod-1`, `v5litepod-4`, `v5litepod-8`, `v5litepod-16`, `v5litepod-32`, `v5litepod-64`, `v5litepod-128`, `v5litepod-256` |
| TPU v5p        | `v5p-8`, `v5p-16`, `v5p-32`                                                                                                   |
| TPU v6e        | `v6e-8`, `v6e-16`                                                                                                             |

#### GPUs

| Type             | Aliases                         | Multi-GPU Counts |
| ---------------- | ------------------------------- | ---------------- |
| NVIDIA T4        | `t4`, `nvidia-tesla-t4`         | 1, 2, 4          |
| NVIDIA L4        | `l4`, `nvidia-l4`               | 1, 2, 4, 8       |
| NVIDIA V100      | `v100`, `nvidia-tesla-v100`     | 1, 2, 4, 8       |
| NVIDIA A100      | `a100`, `nvidia-tesla-a100`     | 1, 2, 4, 8, 16   |
| NVIDIA A100 80GB | `a100-80gb`, `nvidia-a100-80gb` | 1, 2, 4, 8, 16   |
| NVIDIA H100      | `h100`, `nvidia-h100-80gb`      | 1, 2, 4, 8       |
| NVIDIA P4        | `p4`, `nvidia-tesla-p4`         | 1, 2, 4          |
| NVIDIA P100      | `p100`, `nvidia-tesla-p100`     | 1, 2, 4          |

For multi-GPU configurations on GKE, append the count: `a100x4`, `l4x2`, etc.

#### CPU

Use `accelerator="cpu"` to run on a CPU-only node (no accelerator attached).

### CLI Commands

The `kinetic` CLI manages your cloud infrastructure. Install it with `uv pip install keras-kinetic[cli]`.

#### `kinetic up`

Provision all required cloud resources (one-time setup):

```bash
kinetic up
kinetic up --project=my-project --accelerator=t4 --yes
```

#### `kinetic down`

Remove all Kinetic resources to avoid ongoing charges:

```bash
kinetic down
kinetic down --yes   # Skip confirmation prompt
```

This removes the GKE cluster and node pools, Artifact Registry repository and container images, and Cloud Storage buckets.

#### `kinetic status`

View current infrastructure state:

```bash
kinetic status
```

#### `kinetic config`

View current configuration:

```bash
kinetic config
```

#### `kinetic pool`

Manage accelerator node pools after initial setup:

```bash
# Add a node pool for a specific accelerator
kinetic pool add --accelerator=v5e-1

# List current node pools
kinetic pool list

# Remove a node pool by name
kinetic pool remove <pool-name>
```

#### `kinetic jobs`

Inspect and manage async jobs submitted with `@kinetic.submit()`. All subcommands accept `--project`, `--zone`, and `--cluster` overrides (or read from the corresponding environment variables).

**List** all live jobs on the cluster:

```bash
kinetic jobs list
```

**Status** of a specific job:

```bash
kinetic jobs status <job-id>
```

**Logs** — fetch the full log, the last N lines, or stream in real time:

```bash
kinetic jobs logs <job-id>              # full log
kinetic jobs logs <job-id> --tail 100   # last 100 lines
kinetic jobs logs <job-id> --follow     # stream until completion
```

`--follow` and `--tail` are mutually exclusive.

**Result** — block until the job completes and print its return value:

```bash
kinetic jobs result <job-id>
kinetic jobs result <job-id> --timeout 600    # give up after 10 min
kinetic jobs result <job-id> --no-cleanup     # keep k8s/GCS artifacts
```

**Cancel** a running job (deletes the Kubernetes resource):

```bash
kinetic jobs cancel <job-id>
```

**Cleanup** Kubernetes resources and/or GCS artifacts for a finished job:

```bash
kinetic jobs cleanup <job-id>
kinetic jobs cleanup <job-id> --no-k8s   # only delete GCS artifacts
kinetic jobs cleanup <job-id> --no-gcs   # only delete k8s resources
```

#### `kinetic build-base`

Build and push prebuilt base images to Docker Hub or Artifact Registry:

```bash
# Interactive mode
kinetic build-base

# Non-interactive
kinetic build-base --repo us-docker.pkg.dev/my-project/kinetic-base --yes

# Build specific categories only
kinetic build-base --repo myuser/kinetic --category gpu --category tpu
```

After a successful build, set `KINETIC_BASE_IMAGE_REPO` to your repository so Kinetic uses your images. See the [container images documentation](docs/advanced/containers.md) for full details.

#### `kinetic doctor`

Diagnose environment, credentials, and infrastructure issues:

```bash
kinetic doctor
kinetic doctor --project=my-project --zone=us-central2-b
```

Runs read-only checks across seven categories — local tools, authentication, configuration, GCP project access, GCP APIs, infrastructure, and Kubernetes — and reports actionable fix hints for any failures. Exits with code 1 if any check fails.

### Monitoring

#### Google Cloud Console

- **Cloud Build:** [console.cloud.google.com/cloud-build/builds](https://console.cloud.google.com/cloud-build/builds)
- **GKE Workloads:** [console.cloud.google.com/kubernetes/workload](https://console.cloud.google.com/kubernetes/workload)

#### Command Line

```bash
# List GKE jobs
kubectl get jobs -n default
```

## Troubleshooting

### Common Issues

#### "Project must be specified" error

```bash
export KINETIC_PROJECT="your-project-id"
```

#### "404 Requested entity was not found" error

Enable required APIs and create the Artifact Registry repository:

```bash
gcloud services enable compute.googleapis.com \
    cloudbuild.googleapis.com artifactregistry.googleapis.com \
    storage.googleapis.com container.googleapis.com \
    --project=$KINETIC_PROJECT

gcloud artifacts repositories create kinetic \
    --repository-format=docker \
    --location=us \
    --project=$KINETIC_PROJECT
```

#### Permission denied errors

Grant required IAM roles:

```bash
gcloud projects add-iam-policy-binding $KINETIC_PROJECT \
    --member="user:your-email@example.com" \
    --role="roles/storage.admin"
```

#### Container build failures

Check Cloud Build logs:

```bash
gcloud builds list --project=$KINETIC_PROJECT --limit=5
```

### Verify Setup

Run `kinetic doctor` for a comprehensive diagnostic of your environment, credentials, and infrastructure:

```bash
kinetic doctor
```

This checks local tools, authentication, GCP project access, required APIs, cluster health, Kubernetes connectivity, and more — with actionable fix suggestions for any issues found.

For quick infrastructure state, use `kinetic status`. For manual verification:

```bash
# Check authentication
gcloud auth list

# Check project
echo $KINETIC_PROJECT

# Check APIs
gcloud services list --enabled --project=$KINETIC_PROJECT \
    | grep -E "(cloudbuild|artifactregistry|storage|container)"
```

## Contributing

Contributions are welcome. Please read our [contributing guidelines](docs/contributing.md) before submitting pull requests.

All contributions must follow our [Code of Conduct](docs/code-of-conduct.md).

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

**Maintained by the Keras team at Google.**

- [Report Issues](https://github.com/keras-team/kinetic/issues)
- [Discussions](https://github.com/keras-team/kinetic/discussions)
