# Multiple Clusters

Kinetic supports running multiple independent clusters within the same GCP project. Each cluster gets its own isolated set of cloud resources (GKE cluster, Artifact Registry, storage buckets), backed by a separate infrastructure stack.

## Why Use Multiple Clusters?

- **Isolation**: Separate your GPU and TPU workloads into different clusters.
- **Regions**: Run jobs in different GCP regions or zones.
- **Environment**: Maintain separate clusters for development, testing, and production.

## Creating a Cluster

Use the `--cluster` flag with `kinetic up` to create a named cluster.

```bash
# Create a GPU cluster in us-east1-b
kinetic up --cluster=gpu-cluster --zone=us-east1-b --accelerator=a100
```

If the `--cluster` flag is omitted, Kinetic uses the default name `kinetic-cluster`.

## Targeting a Cluster

You can target a specific cluster from your code using the `cluster` parameter or an environment variable.

### Using the Decorator

```python
@kinetic.run(accelerator="a100", cluster="gpu-cluster")
def train_on_gpu():
    ...
```

### Using Environment Variables

Set `KINETIC_CLUSTER` to avoid repeating the cluster name in every decorator.

```bash
export KINETIC_CLUSTER="gpu-cluster"
```

## Managing Clusters

All CLI commands accept the `--cluster` flag, allowing you to manage each cluster independently.

```bash
# Check status of a specific cluster
kinetic status --cluster=gpu-cluster

# Add a node pool to a specific cluster
kinetic pool add --cluster=gpu-cluster --accelerator=h100

# Tear down a specific cluster
kinetic down --cluster=gpu-cluster
```

## Resource Naming

Kinetic uses the cluster name to scope its GCP resources. For a cluster named `gpu-cluster`, the resources will follow this pattern:

- **GKE Cluster**: `gpu-cluster`
- **Artifact Registry**: `kn-gpu-cluster`
- **Storage Bucket**: `{project}-kn-gpu-cluster-jobs`

:::{warning}
**When not to use this:** most users only need one cluster.
Each additional cluster has its own GKE control plane (~$0.10/hr,
or ~$74/month) and its own Artifact Registry, so don't add a second
cluster speculatively. Add one when you have a real reason: GPU vs
TPU isolation, regional separation, or dev vs prod environments.
:::

## Related pages

- [Cost Optimization](../guides/cost_optimization.md) — control plane
  costs and how the GKE free tier covers exactly one cluster.
- [Capacity Reservations](reservations.md) — when reservations make
  multi-cluster setups worth the overhead.
- [Configuration](../configuration.md) — `KINETIC_CLUSTER` and the
  precedence rules.
