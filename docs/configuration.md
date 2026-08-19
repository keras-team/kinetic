# Configuration

Kinetic uses environment variables, decorator arguments, CLI flags, and
optionally [named profiles](guides/profiles.md) for configuration. This
page is the source of truth for what each one does, what the defaults
are, and how they come together when they disagree.

:::{tip}
If you work with more than one cluster or project, consider saving
those combinations as [profiles](guides/profiles.md) — they remove the
need to re-export `KINETIC_*` env vars each time you switch.
:::

## Environment variables

| Variable                     | Used by                   | Default                          | Description                                                                                                                                                  |
| ---------------------------- | ------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `KINETIC_PROJECT`            | CLI + decorators          | _(required)_                     | GCP project ID. Falls back to `GOOGLE_CLOUD_PROJECT` if unset.                                                                                               |
| `KINETIC_ZONE`               | CLI + decorators          | `us-central1-a`                  | GCP zone for jobs and clusters.                                                                                                                              |
| `KINETIC_CLUSTER`            | CLI + decorators          | `kinetic-cluster`                | GKE cluster name.                                                                                                                                            |
| `KINETIC_NAMESPACE`          | CLI + decorators          | `default`                        | Kubernetes namespace.                                                                                                                                        |
| `KINETIC_BASE_IMAGE_REPO`    | Decorator (prebuilt mode) | `kinetic`                        | Repo for prebuilt base images. See [Execution Modes](guides/execution_modes.md).                                                                             |
| `KINETIC_OUTPUT_DIR`         | Library (submit) + remote pod | `gs://{bucket}/outputs/{job_id}` | Per-job durable artifact prefix. Kinetic reads this variable at submit time and sets it in the pod. Kinetic has no equivalent CLI flag. See [Checkpointing](guides/checkpointing.md).                        |
| `KINETIC_RESERVATION`        | `kinetic pool add`        | _(unset)_                        | GCP capacity reservation to consume. Pool-level config, not a per-job setting.                                                                               |
| `KINETIC_LOG_LEVEL`          | Library                   | `INFO`                           | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `FATAL`.                                                                                                                |
| `KINETIC_DEBUG_WAIT_TIMEOUT` | Library + remote pod      | `600`                            | Seconds that Kinetic waits for a debugger to attach when `debug=True`. Kinetic reads the variable when you submit the job, and applies the value to the local wait and to the pod. Set the variable before you submit. The value must be a positive whole number of seconds. If the value is not valid, Kinetic uses `600`. See [Interactive Debugging](guides/debugging.md). |
| `KINETIC_PACKAGE_ROOT`       | Library (submit)          | _(auto-detected)_                | Set this variable to the directory that Kinetic puts into `context.zip`. The directory must exist. The directory must also contain the directory that defines your function. If one of these conditions is not true, Kinetic raises a `ValueError` at submit time. See [What Ships to the Pod](guides/packaging.md). |
| `KINETIC_NO_DEFAULT_EXCLUDES`| Library (submit)          | _(unset)_                        | Set this variable to `1` to turn the default exclusions off. Kinetic then puts `.venv`, `node_modules`, and the cache directories into `context.zip`. Kinetic always excludes `.git` and `__pycache__`.                                                             |
| `KINETIC_CONTEXT_SIZE_WARN_MB`| Library (submit)         | `100`                            | Warning threshold in megabytes for `context.zip`. Above the threshold, Kinetic logs a warning and lists the five largest files. Set the value to `0` to turn the warning off.                                                                                    |
| `KINETIC_PAYLOAD_SIZE_WARN_MB`| Library (submit)         | `50`                             | Warning threshold in megabytes for `payload.pkl`. Above the threshold, Kinetic logs a warning about the arguments and the module-level globals that it captured by value. Set the value to `0` to turn the warning off.                                                                    |

Set them in your shell profile (`~/.bashrc`, `~/.zshrc`) so they
persist across sessions:

```bash
export KINETIC_PROJECT="my-gcp-project-id"
export KINETIC_ZONE="us-central1-a"
```

## Precedence

When the same setting can come from multiple sources, the highest one
wins:

| Setting         | Decorator arg      | CLI flag                         | Env var                                         | Active [profile](guides/profiles.md) | Built-in default                 |
| --------------- | ------------------ | -------------------------------- | ----------------------------------------------- | ------------------------------------ | -------------------------------- |
| Project         | `project=`         | `--project`                      | `KINETIC_PROJECT` (then `GOOGLE_CLOUD_PROJECT`) | `project`                            | _(required)_                     |
| Zone            | `zone=`            | `--zone`                         | `KINETIC_ZONE`                                  | `zone`                               | `us-central1-a`                  |
| Cluster         | `cluster=`         | `--cluster`                      | `KINETIC_CLUSTER`                               | `cluster`                            | `kinetic-cluster`                |
| Namespace       | `namespace=`       | `--namespace`                    | `KINETIC_NAMESPACE`                             | `namespace`                          | `default`                        |
| Output dir      | `output_dir=`      | _(n/a)_                          | `KINETIC_OUTPUT_DIR`                            | _(n/a)_                              | `gs://{bucket}/outputs/{job_id}` |
| Base image repo | `base_image_repo=` | `kinetic build-image --repo`     | `KINETIC_BASE_IMAGE_REPO`                       | _(n/a)_                              | `kinetic`                        |
| Reservation\*   | _(n/a)_            | `kinetic pool add --reservation` | `KINETIC_RESERVATION`                           | _(n/a)_                              | _(unset)_                        |

\* Reservation is a node-pool-level setting, not a per-job one. You bind
a reservation to a pool when you create the pool with `kinetic pool add`,
and any job that lands on that pool consumes it. Because of that there is
no decorator argument; jobs select pools indirectly via `accelerator=`.

Read left to right: a decorator argument always beats a CLI flag, which
beats an env var, which beats a profile field, which beats the built-in
default. Concretely:

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

## Pulumi state

Kinetic stores its Pulumi state in a Google Cloud Storage bucket
derived from the GCP project: `gs://{project}-kinetic-state`. The
bucket is created on first use (idempotent), with **versioning
enabled** and **uniform bucket-level access**, no public ACL.
Multiple clusters in one project share the bucket but get separate
stacks (named `{project}-{cluster}`), so a team running against the
same GCP project automatically converges on one authoritative state.

### IAM

Kinetic uses Application Default Credentials, the same auth path as
`gcloud`. The first admin to run `kinetic up` for a project needs
`roles/storage.admin` so the state bucket can be created. Every other
team member only needs `roles/storage.objectAdmin` on the bucket to
read and write state.

## Where to look

If a setting isn't behaving the way you expect, `kinetic config` prints
the resolved value of the most common variables (project, zone,
cluster, namespace, output dir, and the per-project Pulumi state
bucket) and where each came from (env var,
[profile](guides/profiles.md), or default). Run it before reaching
for `kinetic init`'s troubleshoot path. Variables that aren't shown there
(`KINETIC_BASE_IMAGE_REPO`, `KINETIC_RESERVATION`, `KINETIC_LOG_LEVEL`,
`KINETIC_DEBUG_WAIT_TIMEOUT`) can be inspected with `env | grep
KINETIC_`.

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1em` Getting Started
:link: getting_started
:link-type: doc

Sets the canonical `KINETIC_PROJECT` once.
:::

:::{grid-item-card} {octicon}`stack;1em` Profiles
:link: guides/profiles
:link-type: doc

Named bundles for project/zone/cluster/namespace; the ergonomic
alternative to re-exporting env vars when you target multiple clusters.
:::

:::{grid-item-card} {octicon}`terminal;1em` CLI Reference
:link: cli
:link-type: doc

Generated reference for every flag.
:::

:::{grid-item-card} {octicon}`bug;1em` Troubleshooting
:link: troubleshooting
:link-type: doc

What to check when a setting doesn't take effect.
:::
::::
