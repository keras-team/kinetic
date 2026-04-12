# Capacity Reservations

GCP on-demand capacity for newer accelerators (TPU v6e, H100) is not guaranteed and may fail with `FailedScaleUp` errors. A GCP capacity reservation guarantees hardware is available when your node pool scales up.

## GPU Reservations

GPU reservations can be created directly through the GCP CLI. For example, for a single H100:

```bash
gcloud compute reservations create my-h100-reservation \
  --machine-type=a3-highgpu-1g \
  --vm-count=1 \
  --zone=us-central1-a \
  --project=your-project-id
```

See the [GCP reservations documentation](https://cloud.google.com/compute/docs/instances/reservations-overview) for the full list of supported machine types and options.

## TPU Reservations

TPU machine families do not support self-service reservations via `gcloud compute reservations create`. TPU capacity reservations must be requested through [Google Cloud support](https://cloud.google.com/support). Once approved, the reservation name will be available in your project for the specified zone and TPU type.

## Using a Reservation with Kinetic

Once you have a reservation name (from either path above), pass it to `kinetic pool add`:

```bash
kinetic pool add \
  --accelerator tpu-v6e-8 \
  --reservation my-reservation-name \
  --project your-project-id
```

Kinetic sets `SPECIFIC_RESERVATION` affinity on the node pool so the autoscaler consumes nodes from your reservation instead of competing for on-demand capacity.

## Cleaning Up

Remove the reservation when you are done to avoid ongoing charges:

```bash
gcloud compute reservations delete my-h100-reservation \
  --zone=us-central1-a \
  --project=your-project-id
```

> **Note:** Reservations accrue charges based on the reserved machine type regardless of whether VMs are running.
