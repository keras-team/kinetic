# Configuration

Kinetic uses environment variables, decorator arguments, and CLI flags
for configuration. This page is the source of truth for what each one
does, what the defaults are, and how the four come together when they
disagree.

## Environment variables

Variable                     | Used by                   | Default                          | Description
---------------------------- | ------------------------- | -------------------------------- | --------------------------------------------------------------------------------
`KINETIC_PROJECT`            | CLI + decorators          | _(required)_                     | GCP project ID. Falls back to `GOOGLE_CLOUD_PROJECT` if unset.
`KINETIC_ZONE`               | CLI + decorators          | `us-central1-a`                  | GCP zone for jobs and clusters.
`KINETIC_CLUSTER`            | CLI + decorators          | `kinetic-cluster`                | GKE cluster name.
`KINETIC_NAMESPACE`          | CLI + decorators          | `default`                        | Kubernetes namespace.
`KINETIC_BASE_IMAGE_REPO`    | Decorator (prebuilt mode) | `kinetic`                        | Repo for prebuilt base images. See [Execution Modes](guides/execution_modes.md).
`KINETIC_OUTPUT_DIR`         | CLI + remote pod          | `gs://{bucket}/outputs/{job_id}` | Per-job durable artifact prefix. See [Checkpointing](guides/checkpointing.md).
`KINETIC_RESERVATION`        | `kinetic pool add`        | _(unset)_                        | GCP capacity reservation to consume. Pool-level config, not a per-job setting.
`KINETIC_LOG_LEVEL`          | Library                   | `INFO`                           | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `FATAL`.
`KINETIC_STATE_DIR`          | Library                   | `~/.kinetic/pulumi`              | Pulumi state directory used by `kinetic up`/`down`.
`KINETIC_DEBUG_WAIT_TIMEOUT` | Library + remote pod      | `600`                            | Seconds the remote pod waits for a debugger client to attach when `debug=True`. Applies on both sides (local `debug_attach()` and the pod's debugpy server).

Set them in your shell profile (`~/.bashrc`, `~/.zshrc`) so they
persist across sessions:

```bash
export KINETIC_PROJECT="my-gcp-project-id"
export KINETIC_ZONE="us-central1-a"
```

## Precedence

When the same setting can come from multiple sources, the highest one
wins:

Setting         | Decorator arg      | CLI flag                         | Env var                                         | Built-in default
--------------- | ------------------ | -------------------------------- | ----------------------------------------------- | --------------------------------
Project         | `project=`         | `--project`                      | `KINETIC_PROJECT` (then `GOOGLE_CLOUD_PROJECT`) | _(required)_
Zone            | `zone=`            | `--zone`                         | `KINETIC_ZONE`                                  | `us-central1-a`
Cluster         | `cluster=`         | `--cluster`                      | `KINETIC_CLUSTER`                               | `kinetic-cluster`
Namespace       | `namespace=`       | `--namespace`                    | `KINETIC_NAMESPACE`                             | `default`
Output dir      | `output_dir=`      | `--output-dir`                   | `KINETIC_OUTPUT_DIR`                            | `gs://{bucket}/outputs/{job_id}`
Base image repo | `base_image_repo=` | `kinetic build-base --repo`      | `KINETIC_BASE_IMAGE_REPO`                       | `kinetic`
Reservation\*   | _(n/a)_            | `kinetic pool add --reservation` | `KINETIC_RESERVATION`                           | _(unset)_

\* Reservation is a node-pool-level setting, not a per-job one. You bind
a reservation to a pool when you create the pool with `kinetic pool add`,
and any job that lands on that pool consumes it. Because of that there is
no decorator argument; jobs select pools indirectly via `accelerator=`.

Read left to right: a decorator argument always beats a CLI flag, which
beats an env var, which beats the built-in default. Concretely:

```python
@kinetic.run(accelerator="tpu-v6e-8", project="explicit-project")
def train(): ...
```

uses `explicit-project` even if `KINETIC_PROJECT` is set to something
else.

## Logging

Kinetic uses `absl-py` for logging. Set `KINETIC_LOG_LEVEL` to control
verbosity:

- **DEBUG** — packaging details, dependency hashing, build pipeline,
  GKE submission.
- **INFO** — major lifecycle milestones (default).
- **WARNING / ERROR / FATAL** — only the named severity and above.

```bash
export KINETIC_LOG_LEVEL=DEBUG
```

## Where to look

If a setting isn't behaving the way you expect, `kinetic config` prints
the resolved value of the most common variables (project, zone,
cluster, namespace, output dir, and the local Pulumi state dir) and
where each came from. Run it before reaching for `kinetic doctor`.
Variables that aren't shown there (`KINETIC_BASE_IMAGE_REPO`,
`KINETIC_RESERVATION`, `KINETIC_LOG_LEVEL`, `KINETIC_DEBUG_WAIT_TIMEOUT`)
can be inspected with `env | grep KINETIC_`.

## Related pages

- [Getting Started](getting_started.md) — sets the canonical
  `KINETIC_PROJECT` once.
- [CLI Reference](cli.rst) — generated reference for every flag.
- [Troubleshooting](troubleshooting.md) — what to check when a setting
  doesn't take effect.
