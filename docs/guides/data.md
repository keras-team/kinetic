# Working with Data

`kinetic.Data(...)` is the API that makes input data available to your
remote function. It accepts a local file, a local directory, a `gs://`
Cloud Storage URI, or an `hf://` Hugging Face dataset URI. On the pod,
Kinetic replaces each `Data` object with a plain filesystem path. Your
function sees only paths, never URIs and never `Data` objects. This page
explains the three ways to read data, when to use each way, and the
limits. The appendix at the end is for contributors.

## A first example

```python
import os

import kinetic
from kinetic import Data


@kinetic.run(accelerator="cpu")
def process_data(data_path):
  print(f"Reading from: {data_path}")
  return sorted(os.listdir(data_path))


# A local directory. Kinetic uploads it one time and downloads it to the pod.
process_data(Data("./my_dataset/"))

# A Cloud Storage directory. The trailing slash marks a directory.
process_data(Data("gs://my-bucket/training-set/"))
```

The function code is the same for a local directory and for a Cloud
Storage directory. In both cases `data_path` is a directory on the pod.

A `Data` object that names one file resolves to a file path, not to a
directory. This applies to local files and to single GCS objects:

```python
@kinetic.run(accelerator="cpu")
def load_weights(weights_path):
  import h5py

  with h5py.File(weights_path) as f:
    return list(f.keys())


# Local file
load_weights(Data("./weights.h5"))

# Single GCS object: no trailing slash
load_weights(Data("gs://my-bucket/checkpoints/weights.h5"))
```

:::{important}
For a GCS URI, the trailing slash tells Kinetic which of the two you
mean. `Data("gs://my-bucket/dataset/")` is a directory. Without the
slash, the same URI names one object. Kinetic shows a warning at submit
time if a URI has no trailing slash and its last segment has no file
extension.
:::

## Two ways to pass `Data`

You can pass a `Data` object in two places:

- **As a function argument.** The function receives the path as the
  argument value. A `Data` object can also sit inside a list, a tuple, or
  a dict that you pass as an argument. Use this way when the function
  takes the path explicitly.
- **In the `volumes={...}` decorator argument.** Kinetic places the data
  at the mount path that you give. Use this way when your training script
  reads from a fixed absolute path.

```python
@kinetic.run(
  accelerator="tpu-v5litepod-4",
  volumes={"/data": Data("./dataset/")},
)
def train():
  import pandas as pd

  df = pd.read_csv("/data/train.csv")
  return len(df)
```

The mount path must be an absolute path that starts with `/`. The mount
path of a volume is a directory. If the `Data` object is a single file,
the pod places that file inside the mount directory.

## Choose an access pattern

Three patterns cover almost every job:

1. **Downloaded `Data`** (the default) — `Data("...")`. Kinetic copies
   the data to the local disk of the pod before your function starts.
   Reads are fast, but the pod waits for the download to finish.
2. **FUSE-mounted `Data`** — `Data("gs://...", fuse=True)`. Kinetic
   mounts the Cloud Storage prefix with the GCS FUSE CSI driver. The pod
   does not wait for a download. Each read fetches the bytes from Cloud
   Storage on demand.
3. **Direct `gs://` access** — your code reads Cloud Storage with
   `tf.io.gfile`, `gcsfs`, `tf.data`, `grain`, or a similar library.
   You pass the URI as a plain string, not as a `Data` object. Kinetic
   passes that string through unchanged. Use this pattern only when your
   framework already has a Cloud Storage reader that you want to keep.

Use this table to select a pattern:

| Dataset size       | Access                     | Use                                          |
| ------------------ | -------------------------- | -------------------------------------------- |
| Small (<10 GB)     | Read most or all files     | `Data(...)` (downloaded)                     |
| Small (<10 GB)     | Random access              | `Data(...)` (downloaded)                     |
| Medium (10–100 GB) | Stream one time            | `Data(..., fuse=True)`                       |
| Medium (10–100 GB) | Random access, many epochs | `Data(...)` (downloaded)                     |
| Large (>100 GB)    | Stream a sparse subset     | `Data(..., fuse=True)`                       |
| Large (>100 GB)    | Indexed shards             | `Data(..., fuse=True)` + `tf.data` / `grain` |
| Already in GCS     | Any size                   | `Data("gs://...")` (with or without `fuse`)  |

:::{tip}
**Recommended defaults:**

- For a small or medium dataset that you read every epoch, use plain
  `Data(...)`. The pod downloads the data one time at start. All later
  reads come from the local disk.
- For a dataset that does not fit on the disk of the pod, use
  `Data("gs://...", fuse=True)`. Also use `fuse=True` for a dataset where
  you read only a fraction of the files.
- Wrap Cloud Storage data in `Data(...)` even when the data is already in
  a bucket. Your function then uses the same path-based API for every
  source. Kinetic passes a `gs://` `Data` object through by URI. Kinetic
  does not hash it and does not upload it. The upload cache applies only
  to local data.
:::

:::{note}
The pod reads Cloud Storage as the node service account of the cluster,
`kn-{cluster}-nodes@{project}.iam.gserviceaccount.com`. That account can
read the jobs bucket. For a bucket that `kinetic up` did not create,
grant that account the `roles/storage.objectViewer` role on the bucket
before you submit the job. This rule applies to all three patterns.
:::

## FUSE mounting

`fuse=True` mounts the data through the
[GCS FUSE CSI driver](https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver)
instead of a download. Your function still receives a filesystem path.
Reads stream from Cloud Storage on demand.

```python
@kinetic.run(
  accelerator="tpu-v5litepod-4",
  volumes={"/data": Data("gs://my-bucket/imagenet/", fuse=True)},
)
def train():
  # The pod fetches only the files that the function opens.
  ...
```

FUSE works with `volumes={...}` and with function arguments. FUSE also
works with a local path: Kinetic uploads the local data one time and
then mounts the uploaded copy on the pod. A local single file resolves to
a file path on the pod, not to a directory:

```python
import json

import kinetic
from kinetic import Data


@kinetic.run(accelerator="cpu")
def read_config(config_path):
  with open(config_path) as f:
    return json.load(f)


read_config(Data("./config.json", fuse=True))

# A single GCS object works the same way
read_config(Data("gs://my-bucket/configs/model.json", fuse=True))
```

GCS FUSE can mount directories only. For a single object, Kinetic thus
mounts the parent directory of that object. Your function receives the
path of the object in that mount. The mount also shows the other objects
in the directory, but Kinetic reads no data from them.

You can mix FUSE-mounted data and downloaded data in one job:

```python
@kinetic.run(
  accelerator="tpu-v5litepod-4",
  volumes={
    "/data": Data("gs://my-bucket/large-dataset/", fuse=True),
    "/config": Data("./small-config/"),
  },
)
def train(extra_data): ...


train(Data("./labels.csv"))  # a downloaded function argument
```

Two more rules apply to FUSE:

- **A FUSE mount is read-only.** Your function cannot write under a FUSE
  mount path. Write outputs under `KINETIC_OUTPUT_DIR` instead. See
  [Outputs and Checkpoints](checkpointing.md).
- **Kinetic reserves the prefix `/_kinetic/fuse-data/`.** Kinetic mounts
  FUSE function arguments below that prefix. A `volumes` key below that
  prefix raises a `ValueError` at submit time.

:::{admonition} Prerequisites
:class: important

FUSE mounting needs the GCS FUSE CSI driver addon on the GKE cluster.
`kinetic up` enables the addon by default.
:::

:::{seealso}
For a runnable script that covers volume mounts, single files, many FUSE
volumes, and mixed FUSE and downloaded data in one job, see
[`examples/example_fuse.py`](https://github.com/keras-team/kinetic/blob/main/examples/example_fuse.py).
:::

## Hugging Face datasets

`Data` also accepts an `hf://` URI. The pod downloads the dataset with
the `datasets` library and saves it to a local directory. Your function
receives that directory path and loads it with `datasets.load_from_disk`.

```python
import kinetic
from kinetic import Data


@kinetic.run(accelerator="tpu-v5litepod-4")
def train(dataset_path):
  from datasets import load_from_disk

  ds = load_from_disk(dataset_path)
  return len(ds)


train(Data("hf://imdb?split=train"))
```

Three rules apply to `hf://` URIs:

- The dependency file of your project must list `datasets`. Kinetic does
  not install it for you. See [Dependencies](dependencies.md).
- The query string accepts `split`, `config_name`, and `revision`, for
  example `hf://user/repo?config_name=reviews&split=train`.
- `Data` rejects `fuse=True` with an `hf://` URI. `Data` raises a
  `ValueError`.

If the dataset repository runs its own loading code, pass
`Data("hf://user/repo", hf_trust_remote_code=True)`. The pod then runs
code from that repository, so use this option only for a repository that
you trust. For a gated or private dataset, forward your token to the pod
with `capture_env_vars=["HF_TOKEN"]`. The `datasets` library reads
`HF_TOKEN` from the environment of the pod. Kinetic itself does not read
the token. See [Forward Environment Variables](env_vars.md).

:::{seealso}
For a runnable script that loads a public Hugging Face dataset with
`config_name` and `split`, see
[`examples/hf_dataset_demo.py`](https://github.com/keras-team/kinetic/blob/main/examples/hf_dataset_demo.py).
:::

## Limits and pitfalls

- **A `gs://` directory needs a trailing slash.** Kinetic reads
  `Data("gs://my-bucket/dataset/")` as a directory and
  `Data("gs://my-bucket/dataset")` as a single object. Kinetic logs a
  warning when a `gs://` path has no trailing slash and the last segment
  has no file extension.
- **A `Data` instance is a snapshot.** Kinetic hashes the local files one
  time per `Data` instance and caches the hash. If you edit the files and
  submit the same instance again, the job uses the old upload. Create a
  new `Data` object to upload the changed files.
- **A `Data` object cannot be a set member or a dict key.** Before the
  upload, Kinetic replaces each `Data` object with a dict, and a dict is
  not hashable. Kinetic raises a `ValueError` at submit time for a `Data`
  object inside a set or a frozenset. Kinetic raises the same error for a
  `Data` object that is a dict key. Pass the `Data` object as its own
  argument, or inside a list, a tuple, or a dict value.
- **Kinetic finds `Data` objects only in containers.** Kinetic walks
  lists, tuples, dicts, and their subclasses. Kinetic does not find a
  `Data` object that you store as an attribute of your own class.
- **Large local data logs a warning.** If a local `Data` object is larger
  than 10 GB, Kinetic logs a warning before the upload. The warning
  recommends a `gs://` URI with framework-native I/O.

## How Kinetic caches local data

Kinetic uploads local data one time and reuses the upload for every later
job that references the same data. The cache key is a SHA-256 hash of
the relative path and the contents of every file. The hash also includes
a marker that identifies a single file or a directory. Two consequences
follow:

- A second run with the same directory skips the upload. Kinetic logs
  `Data cache hit` and passes the existing Cloud Storage location.
- A rename or a move of a file inside the directory changes the hash.
  Kinetic uploads the directory again.

Kinetic stores the upload in the jobs bucket at
`gs://{jobs bucket}/default/data-cache/{hash}/`. The `default` segment is
a literal string, not the Kubernetes namespace of your profile. The jobs
bucket deletes objects that are older than 30 days, so the cache is valid
for 30 days after the upload.

Kinetic also excludes a local `Data` path from the source archive of the
job when that path sits inside the package root. Kinetic uploads those
files one time, through the data cache. See
[What Ships to the Pod](packaging.md).

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`history;1em` Outputs and Checkpoints
:link: checkpointing
:link-type: doc

Write durable outputs and checkpoints under `KINETIC_OUTPUT_DIR`.
:::

:::{grid-item-card} {octicon}`package;1em` What Ships to the Pod
:link: packaging
:link-type: doc

The package root, the source archive, and how `Data` paths are excluded.
:::

:::{grid-item-card} {octicon}`key;1em` Forward Environment Variables
:link: env_vars
:link-type: doc

Copy tokens such as `HF_TOKEN` from your shell to the pod.
:::

:::{grid-item-card} {octicon}`beaker;1em` Examples
:link: ../examples
:link-type: doc

Runnable scripts for the `Data` API and for FUSE mounts.
:::
::::

---

## Appendix: implementation internals

This appendix is for contributors and for people who debug data issues.
Users do not need to read it.

### `Data` reference serialization

A `Data` object does not travel to the pod. During `_prepare_artifacts()`,
Kinetic uploads each local `Data` object and replaces every `Data` object
with a serializable `__data_ref__` dict:

```python
{
  "__data_ref__": True,
  "uri": "gs://bucket/default/data-cache/abc123",
  "is_dir": True,
  "mount_path": "/data",  # None unless the Data is a volume or uses FUSE
  "fuse": False,  # True when fuse=True was passed
  "hf_trust_remote_code": False,  # only used for hf:// URIs
}
```

`make_data_ref()` in `kinetic/data/data.py` builds this dict, and
`is_data_ref()` recognizes it. The key is `uri`, not `gcs_uri`, because
the same key also carries `hf://` URIs. FUSE volume specs use an unrelated
key with the name `gcs_uri`. The section below describes those specs.

Kinetic sets `mount_path` for two kinds of `Data`:

- A `Data` object in the `volumes` dictionary gets the mount path that
  you gave.
- A function argument with `fuse=True` gets a generated mount path below
  `/_kinetic/fuse-data/`.

A plain function argument gets `mount_path: None`.

On the pod, `resolve_data_refs()` in `remote_runner.py` walks the
deserialized args and kwargs and replaces these dicts with local
filesystem paths. The walk uses an identity memo. One `Data` object that
you pass two times thus resolves one time. The aliasing between your
arguments stays intact. Kinetic replaces the references only in lists,
tuples, dicts, and the subclasses of these types. Kinetic does not find a
`Data` object that you store as an attribute of your own class.

Kinetic rejects these two shapes at submit time, because Python cannot
hash the replacement dict:

- A `Data` object inside a set or a frozenset.
- A `Data` object used as a dictionary key.

For a downloaded reference, `resolve_data_refs()` returns a directory
path. If the reference is not a directory and the download produced
exactly one file, it returns the path of that file. For an `hf://`
reference, `_download_hf_data()` calls `datasets.load_dataset()` with the
`split`, `config_name`, and `revision` query parameters and then calls
`save_to_disk()` on the target directory.

### Upload and caching pipeline

`upload_data()` in `kinetic/utils/storage.py` uploads local data to
`gs://{jobs bucket}/default/data-cache/{hash}/`. The function has a
`namespace_prefix` parameter with the default value `"default"`. The
callers in `execution.py` do not pass that parameter, so the prefix is
always the literal `default`. The flow:

:::{container} kinetic-steps
1. **Compute the content hash.** `Data.content_hash()` hashes each file
   as SHA-256 of `relpath + "\0" + contents`, in sorted DFS order. It then
   combines the per-file digests under a `dir:` or `file:` prefix. The
   instance caches the result.

2. **Check for a sentinel blob** at `default/data-markers/{hash}`. If
   the blob exists, skip the upload.

3. **Upload the files** under the hash prefix. The directory structure
   is preserved.

4. **Write the sentinel blob** last. The blob signals that the upload is
   complete.
:::

For a single file, the blob is stored at `{hash}/{filename}`. For a
directory, the full tree is preserved under `{hash}/`. The returned Cloud
Storage URI always points to the hash prefix directory, not to a file.
For a `gs://` or `hf://` `Data` object, `upload_data()` returns the
original URI without an upload.

A GCS-hosted `Data` object does not use this pipeline. `upload_data()`
returns the URI that you gave. An `is_dir=False` ref thus has one of two
forms, and `_download_data()` in `remote_runner.py` must accept both:

| Source                         | Ref `uri`                          | The URI names |
| ------------------------------ | ---------------------------------- | ------------- |
| Uploaded local file            | `gs://bucket/ns/data-cache/{hash}` | a directory   |
| `Data("gs://bucket/dir/f.h5")` | `gs://bucket/dir/f.h5`             | the object    |

Only the second form names a blob. The download thus first tries to get
that object. If the bucket has no such object, the download lists the
URI as a prefix. Both branches put the file into the target directory
with its own name. `resolve_data_refs()` then gives your function that
file path.

If an `is_dir=False` ref matches no object and no prefix, the runner
raises `FileNotFoundError`. The message contains the URI. The ref does
not resolve to an empty directory.

### FUSE mount implementation

GCS FUSE mounts directories, not single files. Three layers handle
single files:

**Volume spec construction** (`execution.py`): for a `fuse=True` `Data`
object, Kinetic builds a FUSE volume spec. The spec has the keys
`gcs_uri`, `mount_path`, `is_dir`, and `read_only`. `read_only` is always
`True`. The specs live on `ctx.fuse_volume_specs` and pass through to the
backend.

**URI adjustment for uploaded single files:** `upload_data()` returns a
directory-level URI (`gs://bucket/default/data-cache/{hash}`), because
the hash prefix is a directory. For a FUSE single-file mount of a local
file, `_fuse_gcs_uri()` appends the original filename
(`gs://bucket/default/data-cache/{hash}/config.json`). The `only-dir`
mount option then scopes the mount to the hash directory and not to the
whole `data-cache/` tree. The data ref keeps the directory-level URI. A
`gs://` URI passes through `_fuse_gcs_uri()` unchanged.

**Kubernetes volume generation** (`k8s_utils.py`): each spec becomes an
inline ephemeral CSI volume with the `implicit-dirs` mount option. The
`only-dir` mount option scopes the mount to one Cloud Storage prefix. For
a single file (`is_dir=False`), Kinetic mounts the parent directory. The
pod receives a `gke-gcsfuse/volumes: "true"` annotation, and GKE then
injects the GCS FUSE sidecar.

**File selection in the mount:** `_resolve_fuse_single_file()` in
`remote_runner.py` changes the mounted directory into a file path. It
reads the last segment of the ref URI. Then it searches the mount for
that name. The two ref forms above use different branches, but both
branches are exact:

- A GCS-native ref names the object. The parent mount shows this object
  and the other objects in the directory, and the search finds the
  correct one.
- An uploaded file has a ref that names its hash directory, and the
  search thus finds nothing. That directory contains only one object,
  and that object is the file.

If the mount contains more than one entry and the named object is not
there, the runner raises `FileNotFoundError`. The runner does not select
a different object.
