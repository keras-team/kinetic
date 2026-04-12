# Accelerator Support

Each accelerator and topology requires setting up its own node pool as a prerequisite.

## TPUs

| Type           | Configurations                                                                                                                |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| TPU v6e        | `v6e-8`, `v6e-16`                                                                                                             |
| TPU v5p        | `v5p-8`, `v5p-16`, `v5p-32`                                                                                                   |
| TPU v5 Litepod | `v5litepod-1`, `v5litepod-4`, `v5litepod-8`, `v5litepod-16`, `v5litepod-32`, `v5litepod-64`, `v5litepod-128`, `v5litepod-256` |
| TPU v4         | `v4-4`, `v4-8`, `v4-16`, `v4-32`, `v4-64`, `v4-128`, `v4-256`, `v4-512`, `v4-1024`, `v4-2048`, `v4-4096`                      |
| TPU v3         | `v3-4`, `v3-16`, `v3-32`, `v3-64`, `v3-128`, `v3-256`, `v3-512`, `v3-1024`, `v3-2048`                                         |

## GPUs

| Type             | Aliases                         | Multi-GPU Counts |
| ---------------- | ------------------------------- | ---------------- |
| NVIDIA H100      | `h100`, `nvidia-h100-80gb`      | 1, 2, 4, 8       |
| NVIDIA A100 80GB | `a100-80gb`, `nvidia-a100-80gb` | 1, 2, 4, 8, 16   |
| NVIDIA A100      | `a100`, `nvidia-tesla-a100`     | 1, 2, 4, 8, 16   |
| NVIDIA L4        | `l4`, `nvidia-l4`               | 1, 2, 4, 8       |
| NVIDIA V100      | `v100`, `nvidia-tesla-v100`     | 1, 2, 4, 8       |
| NVIDIA T4        | `t4`, `nvidia-tesla-t4`         | 1, 2, 4          |
| NVIDIA P100      | `p100`, `nvidia-tesla-p100`     | 1, 2, 4          |
| NVIDIA P4        | `p4`, `nvidia-tesla-p4`         | 1, 2, 4          |

For multi-GPU configurations on GKE, append the count: `a100x4`, `l4x2`, etc.

## CPU

Use `accelerator="cpu"` to run on a CPU-only node (no accelerator attached).

## Capacity Reservations

Newer accelerators (TPU v6e, H100) can have limited on-demand availability. If `kinetic pool add` fails to provision nodes, use a GCP capacity reservation to guarantee hardware:

```bash
kinetic pool add --accelerator tpu-v6e-8 --reservation my-v6e-reservation --project your-project-id
```

See the [Capacity Reservations](advanced/reservations.md) guide for details.
