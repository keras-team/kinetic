# Configuration

This page is the reference for every setting that Kinetic reads: where
each setting can come from, which source wins, and what the default is.
For the everyday workflow, see [Profiles](guides/profiles.md).

## The four sources

Kinetic reads a setting from up to four sources. The first source that
has a value wins:

1. **A decorator argument or a CLI flag.** For example,
   `@kinetic.run(project="p")` or `kinetic status --project p`. Use these
   for a one-off override in one call.
2. **A `KINETIC_*` environment variable.** For example,
   `KINETIC_PROJECT=p`. Use these for one shell session, or in CI.
3. **The active profile.** A profile stores the project, the zone, the
   cluster, and the namespace. `kinetic init` creates the first profile.
   This is the source that you set one time.
4. **The built-in default.**

Profiles cover only the four target settings. The other settings in this
page come from a decorator argument, a flag, an environment variable, or
the default.

## Precedence table

| Setting | Decorator argument | CLI flag | Environment variable | Active profile | Built-in default |
| ------- | ------------------ | -------- | -------------------- | -------------- | ---------------- |
| Project | `project=` | `--project` | `KINETIC_PROJECT` | `project` | `GOOGLE_CLOUD_PROJECT` (Python API only), else required |
| Zone | `zone=` | `--zone` | `KINETIC_ZONE` | `zone` | `us-central1-a` |
| Cluster | `cluster=` | `--cluster` | `KINETIC_CLUSTER` | `cluster` | `kinetic-cluster` |
| Namespace | `namespace=` | `--namespace` (`kinetic jobs list`, `kinetic init`, `kinetic up`, `kinetic profile create`) | `KINETIC_NAMESPACE` | `namespace` | `default` |
| Output directory | `output_dir=` | _(none)_ | `KINETIC_OUTPUT_DIR` | _(none)_ | `gs://{jobs bucket}/outputs/{job_id}` |
| Base image repository | `base_image_repo=` | `kinetic build-image --repo` | `KINETIC_BASE_IMAGE_REPO` | _(none)_ | `kinetic` |
| Reservation | _(none)_ | `kinetic pool add --reservation` | `KINETIC_RESERVATION` | _(none)_ | _(unset)_ |
| Profile selection | _(none)_ | `kinetic --profile` | `KINETIC_PROFILE` | the stored `current` | _(none)_ |

Read each row from left to right. A decorator argument beats a flag,
which beats an environment variable, which beats the profile, which beats
the default. For example:

```python
@kinetic.run(accelerator="tpu-v5litepod-4", project="explicit-project")
def train(): ...
```

This job runs in `explicit-project`, even if `KINETIC_PROJECT` and the
active profile name a different project.

A reservation is a property of a node pool, not of a job. You bind a
reservation to a pool with `kinetic pool add`, and every job that lands
on that pool uses it. Jobs select pools through `accelerator=`.

## Environment variables

### Target selection

These four variables have profile equivalents. Set them only for a
one-off override.

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `KINETIC_PROJECT` | _(none)_ | Google Cloud project ID. Required when no profile is active. The Python API also accepts `GOOGLE_CLOUD_PROJECT`, after the profile. The CLI does not read `GOOGLE_CLOUD_PROJECT`. |
| `KINETIC_ZONE` | `us-central1-a` | Zone of the cluster. |
| `KINETIC_CLUSTER` | `kinetic-cluster` | Name of the cluster. |
| `KINETIC_NAMESPACE` | `default` | Kubernetes namespace for jobs. |

### Profile selection

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `KINETIC_PROFILE` | _(unset)_ | Name of the profile to use for this process. Overrides the stored active profile. Both the CLI and the Python API read it. |
| `KINETIC_PROFILES_FILE` | `~/.kinetic/profiles.json` | Path of the profile store. |

### Job behavior

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `KINETIC_OUTPUT_DIR` | `gs://{jobs bucket}/outputs/{job_id}` | On your machine before a submit: the output directory for the job. In the pod: the resolved value, always set. See [Outputs and Checkpoints](guides/checkpointing.md). |
| `KINETIC_BASE_IMAGE_REPO` | `kinetic` | Repository for prebuilt base images. Used only with `container_image="prebuilt"`. See [Container Images](guides/containers.md). |
| `KINETIC_DEBUG_WAIT_TIMEOUT` | `600` | Seconds that Kinetic waits for a debugger to attach when `debug=True`. Kinetic reads the variable when you submit the job, and applies the value to the local wait and to the pod. Set the variable before you submit. The value must be a positive whole number of seconds. If the value is not valid, Kinetic uses `600`. See [Interactive Debugging](guides/debugging.md). |
| `KINETIC_NO_TTY_DEBUG` | _(unset)_ | Set to `1` to permit a blocking call with `debug=True` when `stdin` is not a terminal. See [Interactive Debugging](guides/debugging.md). |

### Packaging

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `KINETIC_PACKAGE_ROOT` | _(detected)_ | The directory that Kinetic archives into `context.zip`. The directory must exist, and it must be the directory of your function or a parent of it. Otherwise Kinetic raises a `ValueError` at submit time. See [What Ships to the Pod](guides/packaging.md). |
| `KINETIC_NO_DEFAULT_EXCLUDES` | _(unset)_ | Set to `1` to turn off the default exclusions. Kinetic then archives `.venv`, `node_modules`, and the cache directories. `.git` and `__pycache__` stay excluded. |
| `KINETIC_CONTEXT_SIZE_WARN_MB` | `100` | Size of `context.zip`, in megabytes, above which Kinetic logs a warning and lists the five largest files. `0` turns the warning off. |
| `KINETIC_PAYLOAD_SIZE_WARN_MB` | `50` | Size of `payload.pkl`, in megabytes, above which Kinetic logs a warning about the arguments and globals that it captured by value. `0` turns the warning off. |

### CLI only

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `KINETIC_RESERVATION` | _(unset)_ | Capacity reservation for `kinetic pool add`. See [Capacity Reservations](guides/reservations.md). |
| `KINETIC_FORCE_DESTROY` | `true` | Whether `kinetic down` empties the buckets before it deletes them. `kinetic up --no-force-destroy` stores `false` in the stack. |
| `KINETIC_LOG_LEVEL` | `INFO` | Log level of the `kinetic` package: `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `FATAL`. |

To make an environment variable persist, put the `export` line in your
shell profile (`~/.bashrc` or `~/.zshrc`). For the target settings,
prefer a profile.

## Logging

Kinetic logs with `absl-py`. `KINETIC_LOG_LEVEL` sets the level:

- `DEBUG` — packaging details, dependency hashing, the build pipeline,
  and the Kubernetes submission.
- `INFO` — the main lifecycle events. This is the default.
- `WARNING`, `ERROR`, `FATAL` — that level and above only.

```bash
export KINETIC_LOG_LEVEL=DEBUG
```

## See the resolved values

`kinetic config` prints the active profile and, for the project, the
zone, the cluster, the namespace, and the output directory, the resolved
value and its source (`KINETIC_*`, `profile`, or `default`). The command
also prints the state bucket of the project. Run it first when a setting
does not take effect.

```bash
kinetic config
```

`kinetic config` cannot see a CLI flag or a decorator argument, because
those apply to one call. The other variables in this page do not appear
in the output. Inspect them with `env | grep KINETIC_`.

## Infrastructure state

The `kinetic` CLI stores its Pulumi state in a Cloud Storage bucket named
`gs://{project}-kinetic-state`. The first `kinetic up` in a project
creates the bucket, with versioning and uniform bucket-level access, and
without a public ACL. All clusters in the project share the bucket. Each
cluster has its own stack, named `{project}-{cluster}`. A team that works
in one project therefore sees one authoritative state.

### IAM

Kinetic uses Application Default Credentials, the same login path as
`gcloud`. The first person who runs `kinetic up` in a project needs
`roles/storage.admin`, because that run creates the state bucket. Every
other team member needs `roles/storage.objectAdmin` on the bucket to
read and write the state.

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`stack;1em` Profiles
:link: guides/profiles
:link-type: doc

The everyday way to set the project, zone, cluster, and namespace.
:::

:::{grid-item-card} {octicon}`server;1em` Clusters and Node Pools
:link: guides/clusters
:link-type: doc

What `kinetic up` creates, and how a team shares it.
:::

:::{grid-item-card} {octicon}`terminal;1em` CLI Reference
:link: cli
:link-type: doc

Generated reference for every command and flag.
:::

:::{grid-item-card} {octicon}`bug;1em` Troubleshooting
:link: troubleshooting
:link-type: doc

What to check when a setting does not take effect.
:::
::::
