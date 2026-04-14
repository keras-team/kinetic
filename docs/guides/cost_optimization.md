# Cost Optimization Guide

Kinetic is designed to provide seamless cloud execution on Google Cloud Platform (GCP) while leveraging modern container orchestration (Google Kubernetes Engine, GKE) to keep operating costs extremely efficient. By default, Kinetic utilizes ephemeral functions and autoscaling node pools to ensure you only pay for compute while your workload is actively running.

This guide covers the primary configurations and workflows for optimizing your cloud bill when using Kinetic.

---

## 1. Understanding Scale-to-Zero Architecture

By default, Kinetic provisions accelerator node pools with **scale-to-zero** capability. 

### How it works:
- **Job Submission:** When you invoke a remote function (via the `@kinetic.run()` decorator), Kinetic creates a Kubernetes `Job` on your GKE cluster.
- **Provisioning:** The GKE Cluster Autoscaler detects pending pods and spins up the required GPU/TPU virtual machines to support the job.
- **Execution & Termination:** Once your workload terminates (either successfully or due to an exception), the container exits and the pod goes away.
- **Scaling Down:** After the pod terminates, if no new jobs are submitted within the GKE idle window (typically around 10 minutes by default in GCP), the Cluster Autoscaler automatically shuts down the underlying VM instances, returning your accelerator compute consumption to zero.

> [!IMPORTANT]
> Because the cluster control plane runs continuously, you will incur a baseline management fee of ~$0.10/hour per cluster. However, Google Cloud provides a [monthly free tier credit](https://cloud.google.com/kubernetes-engine/pricing) ($74.40/month) per billing account, which fully covers the management costs of exactly **one** active cluster. 

---

## 2. Controlling Startup Latency vs. Idle Costs (`--min-nodes`)

When adding a node pool, you must trade off between **cold-start latency** and **idle compute costs**.

### The Default Setup (Zero Idle Cost)
```bash
# Creates a pool that scales from 0 to max required nodes
kinetic pool add --accelerator l4
```
- **Benefit:** Absolutely zero idle accelerator costs when you aren't running jobs.
- **Trade-off:** Every time you submit a job after an idle period, you encounter a cold start (~2 to 5 minutes for VM provisioning, attach operations, and container image fetching).

### Warm Persistent Nodes (Optimizing Developer Latency)
If you are actively debugging or running highly iterative workloads, waiting for VMs to provision can interrupt your flow. You can keep a persistent warm node running using the `--min-nodes` parameter:

```bash
# Keeps at least 1 node alive even when there are no jobs running
kinetic pool add --accelerator l4 --min-nodes 1
```
- **Benefit:** Instant job scheduling and near-zero launch overhead because the container image is already cached locally on the node and the OS/driver layer is initialized.
- **Trade-off:** You pay the continuous on-demand hourly rate for that node until you run `kinetic pool remove` or scale the `--min-nodes` count back down.

---

## 3. Optimizing Container Build Costs (`container_image`)

When you submit a job, Kinetic builds or retrieves a container image to execute your workload.

By default, Kinetic uses **custom bundled images** (`container_image="bundled"`). This means it packages your local workspace and runs Cloud Build to freeze dependencies into an image. 

To save build time and Cloud Build execution charges, you can explicitly request prebuilt base configurations via `container_image="prebuilt"` in `@kinetic.run()`:

```python
@kinetic.run(
  accelerator="l4",
  container_image="prebuilt"
)
def train():
    ...
```

### Why `"prebuilt"` reduces GCP expenses:
- **Zero Cloud Build Fees:** Reuses existing Artifact Registry images immediately.
- **Fast Execution Start:** Minor script changes are uploaded compactly as a context zip payload (`context.zip`) directly to GCS without re-compiling environment layers.

---

## 4. Slashing Compute Costs with Spot Instances (`--spot`)

Spot instances run on unused Google Cloud capacity at significantly discounted rates (**up to 91% lower** than standard on-demand pricing according to the official [GCP Spot VM documentation](https://cloud.google.com/compute/docs/instances/spot)).

You can provision a fully automated Spot pool using the `--spot` flag:

```bash
# Add an A100 node pool backed by highly-discounted Spot capacity
kinetic pool add --accelerator a100 --spot
```

### Best Practices for Spot Nodes:
1. **Fault-Tolerant Workloads Only:** Spot instances can be preempted by GCP with only 30 seconds of notice when capacity constraints occur. Do not use Spot pools for stateful production serving or time-critical jobs that cannot afford restarts.
2. **Use Checkpointing:** Use Kinetic's integration with **Orbax** (see the [Checkpointing Guide](checkpointing.md)) to continuously flush state to Cloud Storage (`gs://`). If a Spot preemption kills your training run midway, you can resume from the last saved checkpoint instead of restarting from scratch.
3. **Multi-Host TPUs:** While you can provision Spot pools for multi-node TPUs (such as `v3`, `v4`, `v5p`, or `v6e`), if any single host in the TPU slice is preempted, the entire slice job will fail. Spot pricing is therefore highly effective for single-host jobs (like `v5litepod-4`, `v5litepod-8`, or `l4` workloads) where you minimize the probability of aggregate preemption.

---

## 5. Managing Capacity Reservations

If your Google Cloud project leverages centralized enterprise pricing via **On-Demand Capacity Reservations**, you can instruct Kinetic to consume that specific reserved capacity pool instead of competing for standard compute stock:

```bash
kinetic pool add --accelerator h100 --reservation my-h100-reservation
```

> [!NOTE]
> You cannot mix `--spot` pricing with `--reservation`. To utilize a reservation, your node pools must use standard on-demand billing tiers. For more details, consult the online [Reservations Reference Guide](https://kinetic.readthedocs.io/en/latest/advanced/reservations.html).

---

## 6. Summary Checklist for Cost Optimization

- [ ] **Utilize Prebuilt Images:** Apply `container_image="prebuilt"` to bypass unneeded cloud dependency compilation. 
- [ ] **Rely on defaults:** Let Kinetic's default `--min-nodes 0` configuration automatically scale your infrastructure down when you step away from your desk.
- [ ] **Utilize Spot capacity:** Use `--spot` for long-running pretraining jobs, ensuring you regularly save model weights to Cloud Storage.
- [ ] **Prune inactive pools:** Actively review your existing infrastructure by running `kinetic pool list` and drop idle pools using `kinetic pool remove <pool_name>`.
- [ ] **Tear down unused clusters:** If you are not using Kinetic for days or weeks at a time, remove the entire underlying cluster via `kinetic down` to avoid baseline control plane charges. **Warning:** This will also delete your Cloud Storage buckets and any saved job data.

