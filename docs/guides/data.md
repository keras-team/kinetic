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
```

You can mix FUSE-mounted and downloaded data in the same job:

```python
@kinetic.run(
    accelerator="tpu-v5e-4",
    volumes={
        "/data": Data("gs://my-bucket/large-dataset/", fuse=True),
        "/config": Data("./small-config/"),
    },
)
def train(extra_data):
    ...

train(Data("./labels.csv"))  # downloaded function-argument data
```

**Prerequisites:** FUSE mounting needs the GCS FUSE CSI driver addon on
the GKE cluster. `kinetic up` enables it by default.

For a runnable end-to-end walkthrough covering volume mounts, single
files, multiple FUSE volumes, and mixed FUSE/downloaded data in the same
job, see
[`examples/example_fuse.py`](https://github.com/keras-team/kinetic/blob/main/examples/example_fuse.py).

## How it caches

Local data is content-addressed: identical bytes upload only once,
regardless of how many jobs reference them. SHA-256 of the contents
becomes the cache key, and re-runs with unchanged data skip the upload
entirely.

This also means files inside your project root that you wrap in
`Data(...)` are automatically excluded from the per-job `context.zip`
payload — no redundant upload of the same bytes.

## Related pages

- [Checkpointing](checkpointing.md): durable outputs and `KINETIC_OUTPUT_DIR`.
- [Examples](../examples.md): walks through the Data API end-to-end.
- [Cost Optimization](cost_optimization.md): FUSE vs download tradeoffs
  for repeated jobs.

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
    "gcs_uri": "gs://bucket/namespace/data-cache/abc123",
    "is_dir": True,
    "mount_path": "/data",      # None for function-argument Data
    "fuse": False,              # True when fuse=True was passed
}
```

On the remote pod, `resolve_data_refs()` in `remote_runner.py` walks the
deserialized args/kwargs recursively and replaces these dicts with local
filesystem paths.

### Upload and caching pipeline

Local data is uploaded to `gs://{bucket}/{namespace}/data-cache/{hash}/`,
where `{hash}` is a SHA-256 computed over sorted file contents. The flow:

1. Compute content hash (deterministic: sorted DFS order, per-file
   SHA-256, then combined).
2. Check for a sentinel blob at `{namespace}/data-markers/{hash}` — if
   present, skip upload.
3. Upload files preserving directory structure under the hash prefix.
4. Write the sentinel blob last to signal upload-complete.

For single files, the blob is stored at `{hash}/{filename}`. For
directories, the full tree is preserved under `{hash}/`. The returned
GCS URI always points to the hash prefix directory, not individual files.

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
