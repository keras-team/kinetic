# AGENTS.md — Keras Remote

## Project Overview

Keras Remote lets users execute Keras/JAX workloads on cloud TPUs and GPUs via a single decorator (`@kinetic.run()`). It handles infrastructure provisioning, container building, job submission, and result retrieval on GCP.

## Architecture

```
kinetic/
├── core/           # @run decorator, accelerator registry & parser
├── backend/        # Job execution backends (GKE, Pathways)
├── data/           # Data class for declaring data dependencies
├── infra/          # Docker container building & caching
├── runner/         # Remote worker entrypoint (runs inside container)
├── utils/          # Serialization (packager) and Cloud Storage helpers
├── cli/            # CLI for infrastructure provisioning (Pulumi-based)
│   ├── commands/   # up, down, status, config, pool (add/remove/list)
│   ├── infra/      # Pulumi programs, stack management, state module, post-deploy steps
│   └── options.py  # Shared --project/--zone/--cluster Click options (common_options decorator)
├── credentials.py  # Credential verification & auto-setup (shared by core & CLI)
└── constants.py    # Zone/region utilities, get_default_cluster_name()
```

## Execution Pipeline

```python
@kinetic.run() called
  → JobContext.from_params()        # Resolve config from args/env vars
  → ensure_credentials()            # Verify/auto-configure gcloud, ADC, kubeconfig
  → _prepare_artifacts()            # Upload Data, serialize function, zip working dir
  → _build_container()              # Build or retrieve cached Docker image
  → _upload_artifacts()             # Upload payload.pkl, context.zip to GCS
  → backend.submit_job()            # Create K8s Job (GKE) or LeaderWorkerSet (Pathways)
  → backend.wait_for_job()          # Poll until completion
  → _download_result()              # Fetch result.pkl from GCS
  → _cleanup_and_return()           # Delete artifacts, return result or re-raise exception
```

## Key Modules

| Module                       | Responsibility                                                                                            |
| ---------------------------- | --------------------------------------------------------------------------------------------------------- |
| `core/core.py`               | `@run()` decorator, backend routing, env var capture                                                      |
| `core/accelerators.py`       | Accelerator registry (`GPUS`, `TPUS`), parser (`parse_accelerator`)                                       |
| `credentials.py`             | Credential verification & auto-setup (gcloud, ADC, kubeconfig)                                            |
| `backend/execution.py`       | `JobContext` dataclass (carries `cluster_name`), `BaseK8sBackend` base class, `execute_remote()` pipeline |
| `backend/gke_client.py`      | K8s Job creation, status polling, pod log retrieval                                                       |
| `backend/pathways_client.py` | LeaderWorkerSet creation for multi-host TPUs                                                              |
| `infra/container_builder.py` | Content-hashed Docker image building via Cloud Build                                                      |
| `data/data.py`               | `Data` class, content hashing, data ref serialization                                                     |
| `utils/packager.py`          | `save_payload()` (cloudpickle), `zip_working_dir()`, Data ref extraction                                  |
| `utils/storage.py`           | GCS upload/download/cleanup for job artifacts and Data cache                                              |
| `runner/remote_runner.py`    | Runs inside container: resolve Data refs/volumes, execute, upload result                                  |
| `cli/infra/state.py`         | Centralized Pulumi state: `load_state()`, `apply_update()`, `apply_destroy()`                             |
| `cli/options.py`             | Shared `common_options` Click decorator (`--project`/`--zone`/`--cluster`)                                |
| `cli/commands/pool.py`       | Node pool add/remove/list commands                                                                        |
| `cli/infra/post_deploy.py`   | kubectl, LWS CRD, GPU driver setup after stack.up()                                                       |
| `cli/constants.py`           | CLI defaults, paths, API list                                                                             |
| `cli/main.py`                | CLI entry point (`kinetic` command)                                                                  |

## Key Abstractions

- **`JobContext`** (`backend/execution.py`): Mutable dataclass carrying all job state through the pipeline — inputs, generated IDs, artifact paths, image URI, `cluster_name` (for cluster-scoped bucket/repo resolution).
- **`BaseK8sBackend`** (`backend/execution.py`): Base class with `submit_job`, `wait_for_job`, `cleanup_job`. Subclassed by `GKEBackend` and `PathwaysBackend`.
- **`GpuConfig` / `TpuConfig`** (`core/accelerators.py`): Frozen dataclasses for accelerator metadata. Single source of truth used by runtime, container builder, and CLI.
- **`Data`** (`data/data.py`): Wraps a local path or GCS URI. Passed as a function argument or via the `volumes` decorator parameter. Resolved to a plain filesystem path on the remote pod. Content-hashed for upload caching.
- **`InfraConfig` / `NodePoolConfig`** (`cli/config.py`): CLI provisioning configuration. `InfraConfig` holds project, zone, cluster name, and a list of `NodePoolConfig` entries. `NodePoolConfig` pairs a unique pool name (e.g., `gpu-l4-a3f2`) with a `GpuConfig` or `TpuConfig`.
- **`StackState`** (`cli/infra/state.py`): Dataclass bundling all state dimensions loaded from a Pulumi stack (project, zone, cluster_name, node_pools, stack handle). Returned by `load_state()` and consumed by commands.

## Data API

The `Data` class (`kinetic.Data`) declares data dependencies for remote functions. It accepts local file/directory paths or GCS URIs (`gs://...`).

### Two usage patterns

**Function arguments** — `Data` objects passed as args/kwargs are uploaded to GCS, serialized as data ref dicts in the payload, and resolved to local paths on the pod:

```python
@kinetic.run(accelerator="v3-8")
def train(data_dir, config_path):
    ...  # data_dir and config_path are plain strings

train(Data("./dataset/"), Data("./config.json"))
```

**Volumes** — `Data` objects in the `volumes=` decorator parameter are downloaded to fixed mount paths before execution:

```python
@kinetic.run(accelerator="v3-8", volumes={"/data": Data("./dataset/")})
def train():
    files = os.listdir("/data")  # available at mount path
```

Both patterns can be combined. `Data` objects can also be nested inside lists, dicts, and other containers — they are recursively discovered and resolved.

### Content-addressed caching

Local `Data` objects are content-hashed (SHA-256 over sorted file contents). Uploads go to `gs://{bucket}/{namespace}/data-cache/{hash}/`. A `.cache_marker` sentinel enables O(1) cache-hit checks. Identical data is uploaded only once.

### Pipeline integration

During `_prepare_artifacts()`:

1. Upload `Data` from `volumes` and function args via `storage.upload_data()` (content-addressed)
2. Replace `Data` objects in args/kwargs with serializable `__data_ref__` dicts
3. Local `Data` paths inside the caller directory are auto-excluded from `context.zip`

On the remote pod (`remote_runner.py`):

1. `resolve_volumes()` — download volume data to mount paths
2. `resolve_data_refs()` — recursively resolve `__data_ref__` dicts in args/kwargs to local paths
3. Single-file `Data` resolves to the file path; directory `Data` resolves to the directory path

## Conventions

### Code Style

- **Formatter/linter**: `ruff` (2-space indent, 80-char line length target, E501 ignored)
- **Rules**: B, E, F, N, PYI, T20, TID, SIM, W, I, NPY
- **Dataclasses**: Frozen for immutable configs, mutable for state objects

### Environment Variables & Resource Name Resolution

Every customizable resource name must follow the same resolution model across all usage paths:

- **`@run()` decorator**: explicit parameter → env var → error or default
- **CLI commands**: `--flag` (with `envvar=`) → env var → interactive prompt or default
- **`config show`**: displays current value and source for every configurable name

| Env Var                      | `@run()` param | CLI flag         | `config show` | Default                |
| ---------------------------- | -------------- | ---------------- | ------------- | ---------------------- |
| `KERAS_REMOTE_PROJECT`       | `project=`     | `--project`      | Yes           | _(required)_           |
| `KERAS_REMOTE_ZONE`          | `zone=`        | `--zone`         | Yes           | `us-central1-a`        |
| `KERAS_REMOTE_CLUSTER`       | `cluster=`     | `--cluster`      | Yes           | `kinetic-cluster` |
| `KERAS_REMOTE_NAMESPACE`     | `namespace=`   | _(runtime only)_ | Yes           | `default`              |

When adding a new configurable resource name, ensure it is wired into **all three paths** (decorator, CLI flags on every relevant command, and `config show`). The `GOOGLE_CLOUD_PROJECT` env var is also accepted as a fallback for project ID (after `KERAS_REMOTE_PROJECT`).

Additional CLI-only env vars:

| Env Var                  | Default                  | Description                  |
| ------------------------ | ------------------------ | ---------------------------- |
| `KERAS_REMOTE_STATE_DIR` | `~/.kinetic/pulumi` | Pulumi local state directory |

### CLI State Management

The CLI manages three layers of state: in-memory config (`InfraConfig`), Pulumi local state files (`~/.kinetic/pulumi/`), and GCP cloud resources. Each `(project, cluster_name)` pair gets its own Pulumi stack (stack name = `{project}-{cluster_name}`), so multiple clusters in the same GCP project are fully independent.

**Centralized state module (`cli/infra/state.py`)** — All Pulumi stack operations go through three functions:

| Function          | Purpose                                                                                 | Used by                         |
| ----------------- | --------------------------------------------------------------------------------------- | ------------------------------- |
| `load_state()`    | Load ALL state dimensions (prerequisites, defaults, refresh, node pools) → `StackState` | `up`, `pool`, `status`          |
| `apply_update()`  | Run `stack.up()` with a complete `InfraConfig`                                          | `up`, `pool add`, `pool remove` |
| `apply_destroy()` | Run `stack.destroy()`                                                                   | `down`                          |

**Safety invariants:**

- `stack.up()`, `stack.destroy()`, `stack.refresh()` appear **only** in `state.py`
- No command file imports `create_program` or `get_stack` directly
- No command file defines inline `--project`/`--zone`/`--cluster` options (use `common_options` from `cli/options.py`)
- When a new state dimension is added (e.g. namespaces), it is added to `StackState` and `load_state()` — every command gets it automatically

**Cluster-scoped resource naming:**

| Resource      | Name pattern                                      |
| ------------- | ------------------------------------------------- |
| Pulumi stack  | `{project}-{cluster_name}`                        |
| Jobs bucket   | `{project}-kr-{cluster_name}-jobs`                |
| Builds bucket | `{project}-kr-{cluster_name}-builds`              |
| AR repository | `kr-{cluster_name}`                               |
| GKE cluster   | `{cluster_name}`                                  |

*Note: GCP APIs are enabled project-wide, shared across clusters, and are not disabled when a cluster is destroyed (`disable_on_destroy=False`).*

Key behaviors:

- **`up` re-runs** preserve existing pools and ignore `--accelerator` (defer to `pool add/remove`)
- **All commands are idempotent** — safe to re-run after partial failure
- **Graceful degradation** — partial failures (refresh, post-deploy steps) log warnings but don't abort the operation
- **Pool state round-trips** through Pulumi stack exports (`accelerators` key) as a list of dicts, reconstructed via `_export_to_node_pool()`

### Testing

- **Framework**: `absl.testing` (not pytest)
- **Location**: Colocated `*_test.py` files alongside source modules
- **Patterns**: `@parameterized.named_parameters` for multi-case tests, mocked GCP/K8s APIs, `tempfile.TemporaryDirectory()` for file ops
- **Integration tests**: `tests/integration/`
- **E2E tests**: `tests/e2e/` (requires live GCP resources)
- **Run tests**: Use pytest (e.g., `/opt/miniconda3/envs/kinetic-3.12/bin/python -m pytest`). Tests use `absl.testing` internally but should be run via pytest for better output.

### Container Caching

Images are tagged with `SHA256(base_image + accelerator_type + requirements.txt + remote_runner.py)`. Identical inputs produce the same tag, skipping rebuild.

### Artifact Registry API

`get_docker_image` requires digest-based names (`image@sha256:...`), not tag-based (`image:tag`). Use `get_tag` with resource name `projects/{p}/locations/{l}/repositories/{r}/packages/{image}/tags/{tag}` to check tagged images.

## Build System

- **Build tool**: hatchling
- **Python**: >=3.11
- **Core deps**: absl-py, cloudpickle, numpy, keras, google-cloud-{artifact-registry,storage,build}, kubernetes
- **CLI deps** (optional `[cli]`): click, rich, pulumi, pulumi-gcp
- **Dev deps** (optional `[dev]`): pre-commit, ruff
- **Entry point**: `kinetic` → `kinetic.cli.main:cli`

## Backend Selection Logic

- **CPU / single-node GPU / single-node TPU**: GKE backend (K8s Job)
- **Multi-node TPU** (`TpuConfig.num_nodes > 1`): Pathways backend (LeaderWorkerSet)
- Explicit `backend=` parameter overrides auto-detection

## Result Serialization Format

```python
{
    "success": bool,
    "result": Any,       # if success=True
    "exception": Exception,  # if success=False
    "traceback": str,        # if success=False
}
```

Exceptions are re-raised locally with the original traceback.
