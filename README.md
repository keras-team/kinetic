# Keras Remote

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Run Keras and JAX workloads on cloud TPUs and GPUs with a simple decorator. No infrastructure management required.

```python
import keras_remote

@keras_remote.run(accelerator="v3-8")
def train_model():
    import keras
    model = keras.Sequential([...])
    model.fit(x_train, y_train)
    return model.history.history["loss"][-1]

# Executes on TPU v3-8, returns the result
final_loss = train_model()
```

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Supported Accelerators](#supported-accelerators)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Simple decorator API** — Add `@keras_remote.run()` to any function to execute it remotely
- **Automatic infrastructure** — No manual VM provisioning or teardown required
- **Result serialization** — Functions return actual values, not just logs
- **Container caching** — Subsequent runs start in 2-4 minutes after initial build
- **Built-in monitoring** — View job status and logs in Google Cloud Console
- **Automatic cleanup** — Resources are released when jobs complete

## Installation

### Library Only

Install the core package to use the `@keras_remote.run()` decorator in your code:

```bash
git clone https://github.com/keras-team/keras-remote.git
cd keras-remote
pip install -e .
```

This is sufficient if your infrastructure (GKE cluster, Artifact Registry, etc.) is already provisioned.

### Library + CLI

Install with the `cli` extra to also get the `keras-remote` command for managing infrastructure:

```bash
git clone https://github.com/keras-team/keras-remote.git
cd keras-remote
pip install -e ".[cli]"
```

This adds the `keras-remote up`, `keras-remote down`, `keras-remote status`, and `keras-remote config` commands for provisioning and tearing down cloud resources.

### Requirements

- Python 3.11+
- Google Cloud SDK (`gcloud`)
  - Run `gcloud auth login` and `gcloud auth application-default login`
- A Google Cloud project with billing enabled

Note: The Pulumi CLI is bundled and managed automatically. It will be installed to `~/.keras-remote/pulumi` on first use if not already present.

## Quick Start

### 1. Configure Google Cloud

Run the CLI setup command:

```bash
keras-remote up
```

This will interactively:

- Prompt for your GCP project ID
- Let you choose an accelerator type (CPU, GPU, or TPU)
- Enable required APIs (Cloud Build, Artifact Registry, Cloud Storage, GKE)
- Create the Artifact Registry repository
- Provision a GKE cluster with optional accelerator node pools
- Configure Docker authentication and kubectl access

You can also run non-interactively:

```bash
keras-remote up --project=my-project --accelerator=t4 --yes
```

To view current infrastructure state:

```bash
keras-remote status
```

To view configuration:

```bash
keras-remote config
```

### 2. Set Environment Variables

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
export KERAS_REMOTE_PROJECT="your-project-id"
export KERAS_REMOTE_ZONE="us-central1-a"  # Optional
```

### 3. Run Your First Job

```python
import keras_remote

@keras_remote.run(accelerator="v3-8")
def hello_tpu():
    import jax
    return f"Running on {jax.devices()}"

result = hello_tpu()
print(result)
```

## Usage Examples

### Basic Computation

```python
import keras_remote

@keras_remote.run(accelerator="v3-8")
def compute(x, y):
    return x + y

result = compute(5, 7)
print(f"Result: {result}")  # Output: Result: 12
```

### Keras Model Training

```python
import keras_remote

@keras_remote.run(accelerator="v3-8")
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

### Custom Dependencies

Create a `requirements.txt` in your project directory:

```text
tensorflow-datasets
pillow
scikit-learn
```

Keras Remote automatically detects and installs dependencies on the remote worker.

### Prebuilt Container Images

Skip container build time by using prebuilt images:

```python
@keras_remote.run(
    accelerator="v3-8",
    container_image="us-docker.pkg.dev/my-project/keras-remote/prebuilt:v1.0"
)
def train():
    ...
```

See [examples/Dockerfile.prebuilt](examples/Dockerfile.prebuilt) for a template.

## Handling Data

Keras Remote provides a declarative and performant Data API to seamlessly make your local and cloud data available to your remote functions.

The Data API is designed to be read-only. It reliably delivers data to your pods at the start of a job. For saving model outputs or checkpointing, you should write directly to GCS from within your function.

Under the hood, the Data API optimizes your workflows with two key features:

- **Smart Caching:** Local data is content-hashed and uploaded to a cache bucket only once. Subsequent job runs that use byte-identical data will hit the cache and skip the upload entirely, drastically speeding up execution.
- **Automatic Zip Exclusion:** When you reference a data path inside your current working directory, Keras Remote automatically excludes that directory from the project's zipped payload to avoid uploading the same data twice.

There are three main ways to handle data depending on your workflow:

### 1. Dynamic Data (The `Data` Class)

The simplest and most Pythonic approach is to pass `Data` objects as regular function arguments. The `Data` class wraps a local file/directory path or a Google Cloud Storage (GCS) URI.

On the remote pod, these objects are automatically resolved into plain string paths pointing to the downloaded files, meaning your function code never needs to know about GCS or cloud storage APIs.

```python
import pandas as pd
import keras_remote
from keras_remote import Data

@keras_remote.run(accelerator="v6e-8")
def train(data_dir):
    # data_dir is resolved to a dynamic local path on the remote machine
    df = pd.read_csv(f"{data_dir}/train.csv")
    # ...

# Uploads the local directory to the remote pod automatically
train(Data("./my_dataset/"))

# Cache hit: subsequent runs with the same data skip the upload!
train(Data("./my_dataset/"))
```

**Note on GCS Directories:** When referencing a GCS directory with the `Data` class, you must include a trailing slash (e.g., `Data("gs://my-bucket/dataset/")`). If you omit the trailing slash, the system will treat it as a single file object.

You can also pass multiple `Data` arguments, or nest them inside lists and dictionaries (e.g., `train(datasets=[Data("./d1"), Data("./d2")])`).

### 2. Static Data (The `volumes` Parameter)

For established training scripts where data requirements are static, you can use the `volumes` parameter in the `@keras_remote.run` decorator. This mounts data at fixed, hardcoded absolute filesystem paths, allowing you to drop `keras_remote` into existing codebases without altering the function signature.

```python
import pandas as pd
import keras_remote
from keras_remote import Data

@keras_remote.run(
    accelerator="v6e-8",
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

### 3. Direct GCS Streaming (For Large Datasets)

If your dataset is very large (e.g., > 10GB), it is inefficient to download the entire dataset to the remote pod's local disk. Instead, skip the `Data` wrapper entirely and pass a GCS URI string directly. You can then use frameworks with native GCS streaming support (like `tf.data` or `grain`) to read the data on the fly.

```python
import grain.python as grain
import keras_remote

@keras_remote.run(accelerator="v6e-8")
def train(data_uri):
    # Native GCS reading, no download overhead
    data_source = grain.ArrayRecordDataSource(data_uri)
    # ...

# Pass as a plain string, no Data() wrapper needed
train("gs://my-bucket/arrayrecords/")

```

## Configuration

### Environment Variables

| Variable               | Required | Default         | Description             |
| ---------------------- | -------- | --------------- | ----------------------- |
| `KERAS_REMOTE_PROJECT` | Yes      | —               | Google Cloud project ID |
| `KERAS_REMOTE_ZONE`    | No       | `us-central1-a` | Default compute zone    |
| `KERAS_REMOTE_CLUSTER` | No       | —               | GKE cluster name        |

### Decorator Parameters

```python
@keras_remote.run(
    accelerator="v3-8",        # Required: TPU/GPU type
    container_image=None,      # Custom container URI
    zone=None,                 # Override default zone
    project=None,              # Override default project
    cluster=None,              # GKE cluster name
    namespace="default"        # Kubernetes namespace
)
```

## Supported Accelerators

Note: each accelerator and topology requires
[setting up its own NodePool](#quick-start) as a prerequisite.

### TPUs

| Type           | Configurations                              |
| -------------- | ------------------------------------------- |
| TPU v2         | `v2-8`, `v2-32`                             |
| TPU v3         | `v3-8`, `v3-32`                             |
| TPU v5 Litepod | `v5litepod-1`, `v5litepod-4`, `v5litepod-8` |
| TPU v5p        | `v5p-8`, `v5p-16`                           |
| TPU v6e        | `v6e-8`, `v6e-16`                           |

### GPUs

| Type        | Aliases                     |
| ----------- | --------------------------- |
| NVIDIA T4   | `t4`, `nvidia-tesla-t4`     |
| NVIDIA L4   | `l4`, `nvidia-l4`           |
| NVIDIA V100 | `v100`, `nvidia-tesla-v100` |
| NVIDIA A100 | `a100`, `nvidia-tesla-a100` |
| NVIDIA H100 | `h100`, `nvidia-h100-80gb`  |

For multi-GPU configurations on GKE, append the count: `a100x4`, `l4x2`, etc.

## Monitoring

### Google Cloud Console

- **Cloud Build:** [console.cloud.google.com/cloud-build/builds](https://console.cloud.google.com/cloud-build/builds)
- **GKE Workloads:** [console.cloud.google.com/kubernetes/workload](https://console.cloud.google.com/kubernetes/workload)

### Command Line

```bash
# List GKE jobs
kubectl get jobs -n default
```

## Troubleshooting

### Common Issues

#### "Project must be specified" error

```bash
export KERAS_REMOTE_PROJECT="your-project-id"
```

#### "404 Requested entity was not found" error

Enable required APIs and create the Artifact Registry repository:

```bash
gcloud services enable cloudbuild.googleapis.com \
    artifactregistry.googleapis.com storage.googleapis.com \
    container.googleapis.com --project=$KERAS_REMOTE_PROJECT

gcloud artifacts repositories create keras-remote \
    --repository-format=docker \
    --location=us \
    --project=$KERAS_REMOTE_PROJECT
```

#### Permission denied errors

Grant required IAM roles:

```bash
gcloud projects add-iam-policy-binding $KERAS_REMOTE_PROJECT \
    --member="user:your-email@example.com" \
    --role="roles/storage.admin"
```

#### Container build failures

Check Cloud Build logs:

```bash
gcloud builds list --project=$KERAS_REMOTE_PROJECT --limit=5
```

### Debug Logging

```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Verify Setup

```bash
# Check authentication
gcloud auth list

# Check project
echo $KERAS_REMOTE_PROJECT

# Check APIs
gcloud services list --enabled --project=$KERAS_REMOTE_PROJECT \
    | grep -E "(cloudbuild|artifactregistry|storage|container)"

# Check Artifact Registry
gcloud artifacts repositories describe keras-remote \
    --location=us --project=$KERAS_REMOTE_PROJECT
```

## Resource Cleanup

Remove all Keras Remote resources to avoid charges:

```bash
keras-remote down
```

This removes:

- GKE cluster and accelerator node pools
- Artifact Registry repository and container images
- Cloud Storage buckets (jobs and builds)
  Use `--yes` to skip the confirmation prompt.

## Contributing

Contributions are welcome. Please read our [contributing guidelines](docs/contributing.md) before submitting pull requests.

All contributions must follow our [Code of Conduct](docs/code-of-conduct.md).

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

**Maintained by the Keras team at Google.**

- [Report Issues](https://github.com/keras-team/keras-remote/issues)
- [Discussions](https://github.com/keras-team/keras-remote/discussions)
