# Working with Data

`kinetic.Data(...)` is the API for getting bytes into your remote function.
It accepts a local file or directory path, or a `gs://` URI, and resolves
to a plain filesystem path inside the pod. Your function code only sees
paths — never URIs, never `Data` objects.

That uniformity is the whole point: you write the same training code
whether the data started on your laptop, in a GCS bucket, or as a
FUSE-mounted dataset too large to fit on disk.

## A first example

```python
import kinetic
from kinetic import Data


@kinetic.run(accelerator="cpu")
def process_data(data_path):
  import os

  print(f"Reading from: {data_path}")
  return sorted(os.listdir(data_path))


# Local directory
process_data(Data("./my_dataset/"))

# GCS directory — trailing slash signals it's a directory
process_data(Data("gs://my-bucket/training-set/"))
```

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

`Data` works as a function argument, as a value inside a list/dict, and as
a value in the `volumes={...}` decorator argument:

```python
@kinetic.run(
  accelerator="tpu-v5e-4",
  volumes={"/data": Data("./dataset/")},
)
def train():
  import pandas as pd

  df = pd.read_csv("/data/train.csv")
  return len(df)
```

Use `volumes={...}` when your training script has hardcoded absolute
paths it expects to read from. Pass `Data(...)` as a function argument
when you'd rather receive the path explicitly.

## Choosing a data access pattern

Three patterns cover almost everything:

1. **Downloaded `Data`** (default) — `Data("...")`. Kinetic copies the
   bytes onto the pod's local disk before your function runs. Reads are
   fast (local disk), but the pod has to wait for the download to finish.
2. **FUSE-mounted `Data`** — `Data("gs://...", fuse=True)`. The bucket
   is mounted lazily; only files you actually `open()` are fetched from
   GCS. Pod startup is near-instant; per-file reads pay GCS latency.
3. **Raw `gs://` streaming** — your code uses `tf.io.gfile`,
   `gcsfs`, or a similar library to talk to GCS directly without
   `Data(...)`. This bypasses the `Data` abstraction entirely; reach for
   it only when you have a specific reason to.

Decision table:

| Dataset size       | Access pattern            | Use                                          |
| ------------------ | ------------------------- | -------------------------------------------- |
| Small (<10 GB)     | Read most/all files       | `Data(...)` (downloaded)                     |
| Small (<10 GB)     | Random access             | `Data(...)` (downloaded)                     |
| Medium (10–100 GB) | Streaming once-through    | `Data(..., fuse=True)`                       |
| Medium (10–100 GB) | Random access many epochs | `Data(...)` (downloaded)                     |
| Large (>100 GB)    | Streaming, sparse subset  | `Data(..., fuse=True)`                       |
| Large (>100 GB)    | Need indexed shards       | `Data(..., fuse=True)` + `tf.data` / `grain` |
| Already in GCS     | Any size                  | `Data("gs://...")` (with or without `fuse`)  |

:::{tip}
**Recommended defaults:**

- For small or medium datasets you read every epoch, use plain
  `Data(...)`. The download cost is paid once at pod startup; subsequent
  reads are local-disk fast.
- For datasets that are too large to fit on the pod's disk, or where you
  only touch a fraction of the files, use `Data("gs://...", fuse=True)`.
- Wrap GCS data in `Data(...)` even when it is already in GCS so your
  function uses the same path-based API regardless of source. Note that
  Kinetic's content-hash-based upload caching applies only to local
  data; GCS-hosted `Data` is passed through by URI without rehashing or
  re-uploading.
:::

## FUSE mounting

`fuse=True` mounts the data through the
[GCS FUSE CSI driver](https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver)
instead of downloading it. Your function still receives a filesystem
path; reads stream on demand from GCS.

```python
@kinetic.run(
  accelerator="tpu-v5e-4",
  volumes={"/data": Data("gs://my-bucket/imagenet/", fuse=True)},
)
def train():
  # Only files you open() are fetched from GCS
  ...
```

FUSE works with both `volumes={...}` and function arguments, with both
local paths and GCS URIs. Single files work transparently — the pod sees
a file path, not a directory:

```python
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

You can mix FUSE-mounted and downloaded data in the same job:

```python
@kinetic.run(
  accelerator="tpu-v5e-4",
  volumes={
    "/data": Data("gs://my-bucket/large-dataset/", fuse=True),
    "/config": Data("./small-config/"),
  },
)
def train(extra_data): ...


train(Data("./labels.csv"))  # downloaded function-argument data
```

:::{admonition} Prerequisites
:class: important

FUSE mounting needs the GCS FUSE CSI driver addon on
the GKE cluster. `kinetic up` enables it by default.
:::

:::{seealso}
For a runnable end-to-end walkthrough covering volume mounts, single
files, multiple FUSE volumes, and mixed FUSE/downloaded data in the same
job, see
[`examples/example_fuse.py`](https://github.com/keras-team/kinetic/blob/main/examples/example_fuse.py).
:::

## How it caches

Local data is content-addressed: identical bytes upload only once,
regardless of how many jobs reference them. SHA-256 of the contents
becomes the cache key, and re-runs with unchanged data skip the upload
entirely.

This also means files inside your project root that you wrap in
`Data(...)` are automatically excluded from the per-job `context.zip`
payload — no redundant upload of the same bytes.

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`history;1em` Checkpointing
:link: checkpointing
:link-type: doc

Durable outputs and `KINETIC_OUTPUT_DIR`.
:::

:::{grid-item-card} {octicon}`beaker;1em` Examples
:link: ../examples
:link-type: doc

Walks through the Data API end-to-end.
:::

:::{grid-item-card} {octicon}`graph;1em` Cost Optimization
:link: cost_optimization
:link-type: doc

FUSE vs download tradeoffs for repeated jobs.
:::
::::

---

## Appendix: implementation internals

The rest of this page is for contributors and people debugging
data-related issues. End users do not need to read it.

### `Data` reference serialization

`Data` objects can't be sent directly to the remote pod. During
`_prepare_artifacts()`, each `Data` is uploaded to GCS and replaced with
a serializable `__data_ref__` dict:

```python
{
  "__data_ref__": True,
  "uri": "gs://bucket/namespace/data-cache/abc123",
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

On the remote pod, `resolve_data_refs()` in `remote_runner.py` walks the
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

### Upload and caching pipeline

Local data is uploaded to `gs://{bucket}/{namespace}/data-cache/{hash}/`,
where `{hash}` is a SHA-256 computed over sorted file contents. The flow:

:::{container} kinetic-steps
1. **Compute content hash.**
   Deterministic: sorted DFS order, per-file SHA-256, then combined.

2. **Check for a sentinel blob** at `{namespace}/data-markers/{hash}` — if
   present, skip upload.

3. **Upload files** preserving directory structure under the hash prefix.

4. **Write the sentinel blob** last to signal upload-complete.
:::

For single files, the blob is stored at `{hash}/{filename}`. For
directories, the full tree is preserved under `{hash}/`. The returned
GCS URI always points to the hash prefix directory, not individual files.

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

GCS FUSE can only mount directories, not individual files. The system
handles this through several layers:

**Volume spec construction** (`execution.py`): for `fuse=True` Data, a
FUSE volume spec is built with `gcs_uri`, `mount_path`, `is_dir`, and
`read_only`. Specs live on `ctx.fuse_volume_specs` and pass through to
the backend.

**URI adjustment for uploaded single files:** `upload_data()` returns a
directory-level URI (`gs://bucket/ns/data-cache/{hash}`) since the hash
prefix is a directory. For FUSE single-file mounts, `_fuse_gcs_uri()`
appends the original filename (`gs://bucket/ns/data-cache/{hash}/config.json`)
so the `only-dir` mount option scopes to the hash directory rather than
the entire `data-cache/` tree. The data ref retains the directory-level
URI for download compatibility.

**K8s volume generation:** each spec becomes an inline ephemeral CSI
volume. The `only-dir` mount option scopes the mount to a specific GCS
prefix. For single files (`is_dir=False`), the parent directory is
mounted. The pod receives a `gke-gcsfuse/volumes: "true"` annotation to
trigger the GCS FUSE sidecar injection.

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
