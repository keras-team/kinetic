# Kinetic Cheat Sheet

## CLI Commands

| Command | Description |
| --- | --- |
| `kinetic up` | Provision all required cloud resources (one-time setup). |
| `kinetic down` | Remove all Kinetic resources to avoid ongoing charges. |
| `kinetic status` | View current infrastructure state. |
| `kinetic config` | View current configuration (project, zone, cluster). |
| `kinetic pool add` | Add a node pool for a specific accelerator. |
| `kinetic pool list` | List current node pools. |
| `kinetic pool remove` | Remove a node pool by name. |
| `kinetic jobs list` | List all live jobs on the cluster. |
| `kinetic jobs status <id>` | Get status of a specific job. |
| `kinetic jobs logs <id>` | Fetch logs (use `--follow` to stream). |
| `kinetic jobs result <id>` | Block until completion and print return value. |
| `kinetic jobs cancel <id>` | Cancel a running job. |
| `kinetic jobs cleanup <id>` | Clean up k8s resources and/or GCS artifacts. |
| `kinetic doctor` | Diagnose environment, credentials, and infrastructure issues. |

## Decorators

### `@kinetic.run()`
Synchronous execution. Blocks until the remote function returns.

```python
@kinetic.run(accelerator="tpu-v5e-1")
def my_func(arg1):
  return "result"
```

### `@kinetic.submit()`
Asynchronous execution. Returns a `JobHandle` immediately.

```python
@kinetic.submit(accelerator="gpu-l4")
def my_func(arg1):
  return "result"

job = my_func("val")
print(job.job_id)
```

## Data API

### `kinetic.Data(path, fuse=False)`
- `path`: Local path or `gs://` URI.
- `fuse`: If `True`, use GCS FUSE mount instead of downloading.

```python
# As argument
train(Data("./dataset"))

# As volume mount
@kinetic.run(volumes={"/data": Data("./dataset")})
def train():
  pass
```

## Environment Variables

- `KINETIC_PROJECT`: Google Cloud project ID (Required).
- `KINETIC_ZONE`: Default compute zone (Default: `us-central1-a`).
- `KINETIC_CLUSTER`: GKE cluster name (Default: `kinetic-cluster`).
