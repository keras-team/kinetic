# GCS FUSE Integration

This guide demonstrates how to use the GCS FUSE CSI driver with Kinetic to lazily mount data instead of downloading it into the container. This is useful for large datasets where your job only needs to read a subset of files at runtime.

## Overview

By default, Kinetic downloads data into the container before your function runs. Passing `fuse=True` to `Data(...)` tells Kinetic to mount the data via the [GCS FUSE CSI driver](https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver) instead. Files are fetched on demand — only the ones you actually open are read from GCS.

> **Prerequisites**: Your GKE cluster must have the GCS FUSE CSI driver addon enabled. `kinetic up` enables it by default.

## Example

Here is a complete example showing the various ways to use FUSE-mounted data with Kinetic. You can find this file at [`examples/example_fuse.py`](https://github.com/keras-team/kinetic/blob/main/examples/example_fuse.py) in the repository.

```{literalinclude} ../../examples/example_fuse.py
```

## When to Use FUSE

| Scenario | Recommendation |
|---|---|
| Large dataset, only read a subset of files | `fuse=True` |
| Streaming reads (e.g., `tf.data`, `grain`) | `fuse=True` |
| Small dataset, read everything | Default (download) |
| Need maximum read throughput on all files | Default (download) |

For more details on working with data in Kinetic, see the [Data guide](data.md).
