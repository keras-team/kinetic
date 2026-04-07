# Container Images

Kinetic supports three container image modes that control how your remote execution environment is built and deployed. Choose the mode that best fits your workflow by setting the `container_image` parameter in the `@kinetic.run()` or `@kinetic.submit()` decorator.

| Mode                  | `container_image=`    | Build step                       | Dependencies installed              |
| --------------------- | --------------------- | -------------------------------- | ----------------------------------- |
| **Bundled** (default) | `None` or `"bundled"` | Cloud Build (cached by dep hash) | Baked into the image                |
| **Prebuilt**          | `"prebuilt"`          | None                             | At pod startup via `uv pip install` |
| **Custom**            | `"<image-uri>"`       | None (you manage it)             | Whatever is in your image           |

## Bundled Mode (Default)

Bundled mode builds a custom container image via Cloud Build with all your dependencies baked in. The image is tagged by a hash of your dependencies, so unchanged dependencies reuse the cached image.

```python
import kinetic

# Bundled mode — these are equivalent:
@kinetic.run(accelerator="v6e-8")
def train():
    ...

@kinetic.run(accelerator="v6e-8", container_image="bundled")
def train():
    ...
```

### Tradeoffs

- **Reproducible**: The exact environment is frozen in the image.
- **First-run cost**: The initial build takes ~2-5 minutes. Subsequent runs with unchanged dependencies use the cached image and start within a few seconds.
- **Good for**: Production workloads, large dependency sets where you want to avoid per-run install overhead, or when you need a fully reproducible environment.

## Prebuilt Mode

Prebuilt mode uses a pre-published base image that already contains the accelerator runtime (JAX, CUDA/TPU libraries) and core dependencies. Your project's `requirements.txt` or `pyproject.toml` dependencies are installed at pod startup via `uv pip install`, so there is no Cloud Build step.

```python
@kinetic.run(accelerator="v6e-8", container_image="prebuilt")
def train():
    ...
```

### How it works

1. Kinetic resolves the base image from the image repository (see [Custom prebuilt images](#custom-prebuilt-images) below) using the accelerator category (`cpu`, `gpu`, or `tpu`) and the kinetic package version.
2. Your project dependencies are filtered (JAX packages are removed to avoid conflicts) and uploaded to GCS alongside your code.
3. At pod startup, the runner installs your dependencies with `uv pip install` before executing your function.

### Tradeoffs

- **Fast iteration**: No build step means jobs start quickly.
- **Startup cost**: `uv pip install` runs on every job. For large dependency sets this adds time to each run.
- **Good for**: Most workflows, especially during development and experimentation.

### Custom prebuilt images

By default, Kinetic pulls official base images from Docker Hub (`kinetic/base-{category}:{version}`). To use your own prebuilt images — for example, with additional system libraries or private packages — build and push them with the `kinetic build-base` command, then point Kinetic at your repository.

Build and push images:

```bash
kinetic build-base --repo us-docker.pkg.dev/my-project/kinetic-base
```

Then set the repository so Kinetic uses your images:

```bash
export KINETIC_BASE_IMAGE_REPO=us-docker.pkg.dev/my-project/kinetic-base
```

Or pass it directly to the decorator:

```python
@kinetic.run(accelerator="l4", base_image_repo="us-docker.pkg.dev/my-project/kinetic-base")
def train():
    ...
```

See [`kinetic build-base`](#kinetic-build-base) for the full command reference.

## Custom Image Mode

Provide a full container image URI to use your own image. Kinetic skips all build and dependency steps.

```python
@kinetic.run(
    accelerator="v6e-8",
    container_image="us-docker.pkg.dev/my-project/kinetic/my-image:v1.0"
)
def train():
    ...
```

### Requirements for custom images

Your custom image must:

1. Include `cloudpickle`, `google-cloud-storage`, and a compatible Python environment.
2. Include the necessary dependencies for your function.
3. Be accessible from the GKE nodes (e.g., Artifact Registry in the same GCP project, or a public registry).

### When to use custom images

- **Complex dependencies**: Non-Python system libraries (CUDA builds, C++ libs) that aren't in the default template.
- **Corporate compliance**: Base images vetted by your security or platform team.
- **Full control**: When you want to manage the entire image lifecycle yourself.

## `kinetic build-base`

Build and push prebuilt base images to a Docker Hub or Artifact Registry repository. One image is built per accelerator category (`cpu`, `gpu`, `tpu`) using Cloud Build.

```bash
# Interactive mode — guides you through registry selection and setup
kinetic build-base

# Non-interactive with Artifact Registry
kinetic build-base \
  --repo us-docker.pkg.dev/my-project/kinetic-base \
  --project my-project \
  --yes

# Build only GPU and TPU images
kinetic build-base --repo myuser/kinetic --category gpu --category tpu

# Use a custom Dockerfile
kinetic build-base --repo myuser/kinetic --dockerfile ./Dockerfile.custom

# Specific version tag (default: kinetic package version)
kinetic build-base --repo myuser/kinetic --tag v2.0.0
```

### Options

| Option                 | Description                                                                                                                    |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `--repo`               | Image repository (Docker Hub or Artifact Registry). Omit to select interactively.                                              |
| `--category`           | Accelerator categories to build: `cpu`, `gpu`, `tpu` (default: all). Repeatable.                                               |
| `--tag`                | Image version tag (default: kinetic package version).                                                                          |
| `--dockerfile`         | Path to a custom Dockerfile. Must install `uv`, `cloudpickle`, `google-cloud-storage`, and `COPY remote_runner.py` to `/app/`. |
| `--update-credentials` | Re-enter Docker Hub credentials even if they already exist in Secret Manager.                                                  |
| `--yes`, `-y`          | Skip confirmation prompt.                                                                                                      |
| `--project`            | GCP project ID (default: `KINETIC_PROJECT`).                                                                                   |
| `--cluster`            | GKE cluster name (default: `kinetic-cluster`).                                                                                 |

### Registry support

- **Docker Hub**: Credentials are stored in GCP Secret Manager and used by Cloud Build during the push. The command prompts for your Docker Hub username and access token on first use.
- **Artifact Registry**: No additional credentials needed — the build service account authenticates automatically. The command prints the required `gcloud` setup commands for creating the repository and granting permissions.
