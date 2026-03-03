# AGENTS.md — Keras Remote

## Project Overview

Keras Remote lets users execute Keras/JAX workloads on cloud TPUs and GPUs via a single decorator (`@keras_remote.run()`). It handles infrastructure provisioning, container building, job submission, and result retrieval on GCP.

## Architecture

```
keras_remote/
├── core/           # @run decorator, accelerator registry & parser
├── backend/        # Job execution backends (GKE, Pathways)
├── infra/          # Docker container building & caching
├── runner/         # Remote worker entrypoint (runs inside container)
├── utils/          # Serialization (packager) and Cloud Storage helpers
├── cli/            # CLI for infrastructure provisioning (Pulumi-based)
│   ├── commands/   # up, down, status, config, pool (add/remove/list)
│   └── infra/      # Pulumi programs, stack management, post-deploy steps
├── credentials.py  # Credential verification & auto-setup (shared by core & CLI)
└── constants.py    # Zone/region utilities
```

## Execution Pipeline

```python
@keras_remote.run() called
  → JobContext.from_params()        # Resolve config from args/env vars
  → ensure_credentials()            # Verify/auto-configure gcloud, ADC, kubeconfig
  → _prepare_artifacts()            # Serialize function (cloudpickle), zip working dir
  → _build_container()              # Build or retrieve cached Docker image
  → _upload_artifacts()             # Upload payload.pkl, context.zip to GCS
  → backend.submit_job()            # Create K8s Job (GKE) or LeaderWorkerSet (Pathways)
  → backend.wait_for_job()          # Poll until completion
  → _download_result()              # Fetch result.pkl from GCS
  → _cleanup_and_return()           # Delete artifacts, return result or re-raise exception
```

## Key Modules

| Module                       | Responsibility                                                                   |
| ---------------------------- | -------------------------------------------------------------------------------- |
| `core/core.py`               | `@run()` decorator, backend routing, env var capture                             |
| `core/accelerators.py`       | Accelerator registry (`GPUS`, `TPUS`), parser (`parse_accelerator`)              |
| `credentials.py`             | Credential verification & auto-setup (gcloud, ADC, kubeconfig)                   |
| `backend/execution.py`       | `JobContext` dataclass, `BaseK8sBackend` base class, `execute_remote()` pipeline |
| `backend/gke_client.py`      | K8s Job creation, status polling, pod log retrieval                              |
| `backend/pathways_client.py` | LeaderWorkerSet creation for multi-host TPUs                                     |
| `infra/container_builder.py` | Content-hashed Docker image building via Cloud Build                             |
| `utils/packager.py`          | `save_payload()` (cloudpickle), `zip_working_dir()`                              |
| `utils/storage.py`           | GCS upload/download/cleanup for job artifacts                                    |
| `runner/remote_runner.py`    | Runs inside container: deserialize, execute, upload result                       |
| `cli/commands/pool.py`       | Node pool add/remove/list commands                                               |
| `cli/infra/post_deploy.py`   | kubectl, LWS CRD, GPU driver setup after stack.up()                              |
| `cli/constants.py`           | CLI defaults, paths, API list                                                    |
| `cli/main.py`                | CLI entry point (`keras-remote` command)                                         |

## Key Abstractions

- **`JobContext`** (`backend/execution.py`): Mutable dataclass carrying all job state through the pipeline — inputs, generated IDs, artifact paths, image URI.
- **`BaseK8sBackend`** (`backend/execution.py`): Base class with `submit_job`, `wait_for_job`, `cleanup_job`. Subclassed by `GKEBackend` and `PathwaysBackend`.
- **`GpuConfig` / `TpuConfig`** (`core/accelerators.py`): Frozen dataclasses for accelerator metadata. Single source of truth used by runtime, container builder, and CLI.
- **`InfraConfig` / `NodePoolConfig`** (`cli/config.py`): CLI provisioning configuration. `InfraConfig` holds project, zone, cluster name, and a list of `NodePoolConfig` entries. `NodePoolConfig` pairs a unique pool name (e.g., `gpu-l4-a3f2`) with a `GpuConfig` or `TpuConfig`.

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
| `KERAS_REMOTE_CLUSTER`       | `cluster=`     | `--cluster`      | Yes           | `keras-remote-cluster` |
| `KERAS_REMOTE_GKE_NAMESPACE` | `namespace=`   | _(runtime only)_ | Yes           | `default`              |

When adding a new configurable resource name, ensure it is wired into **all three paths** (decorator, CLI flags on every relevant command, and `config show`). The `GOOGLE_CLOUD_PROJECT` env var is also accepted as a fallback for project ID (after `KERAS_REMOTE_PROJECT`).

Additional CLI-only env vars:

| Env Var                  | Default                  | Description                  |
| ------------------------ | ------------------------ | ---------------------------- |
| `KERAS_REMOTE_STATE_DIR` | `~/.keras-remote/pulumi` | Pulumi local state directory |

### CLI State Management

The CLI manages three layers of state: in-memory config (`InfraConfig`), Pulumi local state files (`~/.keras-remote/pulumi/`), and GCP cloud resources. Each GCP project gets its own Pulumi stack (stack name = project ID).

Every mutating command (`up`, `pool add`, `pool remove`, etc.) follows this reconciliation pattern:

1. `stack.refresh()` — pull cloud reality into local state
2. `get_current_node_pools()` — read current pools from stack exports
3. Build new `InfraConfig` — merge existing pools with desired changes
4. `create_program(config)` — generate Pulumi program from desired state
5. `stack.up()` — diff desired vs current, apply only changes

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
- **Run tests**: Use pytest (e.g., `/opt/miniconda3/envs/keras-remote-3.12/bin/python -m pytest`). Tests use `absl.testing` internally but should be run via pytest for better output.

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
- **Entry point**: `keras-remote` → `keras_remote.cli.main:cli`

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
