# Usage Guide

## Training a Keras Model

```python
import kinetic

@kinetic.run(accelerator="v6e-8")
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

## Working with Data

Kinetic provides a declarative Data API to seamlessly make your local and cloud data available to remote functions.

The Data API is read-only — it delivers data to your pods at the start of a job. For saving model outputs or checkpointing, write directly to GCS from within your function.

Under the hood, the Data API provides two key optimizations:

- **Smart Caching:** Local data is content-hashed and uploaded to a cache bucket only once. Subsequent job runs with byte-identical data skip the upload entirely.
- **Automatic Zip Exclusion:** When you reference a data path inside your current working directory, Kinetic automatically excludes that directory from the project's zipped payload to avoid uploading the same data twice.

There are three approaches depending on your workflow:

### Dynamic Data (The `Data` Class)

The simplest approach — pass `Data` objects as regular function arguments. The `Data` class wraps a local file/directory path or a Google Cloud Storage (GCS) URI.

On the remote pod, these objects are automatically resolved into plain string paths pointing to the downloaded files, so your function code never needs to know about GCS or cloud storage APIs.

```python
import pandas as pd
import kinetic
from kinetic import Data

@kinetic.run(accelerator="v6e-8")
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

### Static Data (The `volumes` Parameter)

For established training scripts where data requirements are fixed, use the `volumes` parameter in the decorator. This mounts data at hardcoded absolute filesystem paths, allowing you to use Kinetic with existing codebases without altering the function signature.

```python
import pandas as pd
import kinetic
from kinetic import Data

@kinetic.run(
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

### Direct GCS Streaming (For Large Datasets)

If your dataset is very large (e.g., > 10GB), it is inefficient to download the entire dataset to the pod's local disk. Instead, skip the `Data` wrapper and pass a GCS URI string directly. Use frameworks with native GCS streaming support (like `tf.data` or `grain`) to read the data on the fly.

```python
import grain.python as grain
import kinetic

@kinetic.run(accelerator="v6e-8")
def train(data_uri):
    # Native GCS reading, no download overhead
    data_source = grain.ArrayRecordDataSource(data_uri)
    # ...

# Pass as a plain string, no Data() wrapper needed
train("gs://my-bucket/arrayrecords/")
```

## Custom Dependencies

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

## Prebuilt Container Images

Skip container build time by using prebuilt images:

```python
@kinetic.run(
    accelerator="v6e-8",
    container_image="us-docker.pkg.dev/my-project/kinetic/prebuilt:v1.0"
)
def train():
    ...
```

Build your own prebuilt image using the project's Dockerfile template as a starting point.

## Forwarding Environment Variables

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

## Multi-Host TPU (Pathways)

Multi-host TPU configurations (those requiring more than one node, such as `v2-16`, `v3-32`, or `v5p-16`) automatically use the [Pathways](https://cloud.google.com/tpu/docs/pathways-overview) backend. You can also set the backend explicitly:

```python
@kinetic.run(accelerator="v3-32", backend="pathways")
def distributed_train():
    ...
```

## Multiple Clusters

You can run multiple independent clusters within the same GCP project — for example, one for GPU workloads and another for TPUs. Each cluster gets its own isolated set of cloud resources (GKE cluster, Artifact Registry, storage buckets) backed by a separate infrastructure stack, so they never interfere with each other.

**Create clusters** by passing `--cluster` to `kinetic up`:

```bash
# Default cluster (named "kinetic-cluster")
kinetic up --project=my-project --accelerator=v6e-8

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

## Configuration

### Environment Variables

| Variable                     | Required | Default                | Description                                                  |
| ---------------------------- | -------- | ---------------------- | ------------------------------------------------------------ |
| `KINETIC_PROJECT`       | Yes      | —                      | Google Cloud project ID                                      |
| `KINETIC_ZONE`          | No       | `us-central1-a`        | Default compute zone                                         |
| `KINETIC_CLUSTER`       | No       | `kinetic-cluster` | GKE cluster name                                             |
| `KINETIC_NAMESPACE`     | No       | `default`              | Kubernetes namespace                                         |
| `KINETIC_LOG_LEVEL`     | No       | `INFO`                 | Log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `FATAL`) |

### Supported Accelerators

Each accelerator and topology requires setting up its own node pool as a prerequisite.

#### TPUs

| Type           | Configurations                                                                                                                |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| TPU v3         | `v3-4`, `v3-16`, `v3-32`, `v3-64`, `v3-128`, `v3-256`, `v3-512`, `v3-1024`, `v3-2048`                                         |
| TPU v4         | `v4-4`, `v4-8`, `v4-16`, `v4-32`, `v4-64`, `v4-128`, `v4-256`, `v4-512`, `v4-1024`, `v4-4096`                                 |
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
