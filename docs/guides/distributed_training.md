# Distributed Training

Scaling training to multiple TPU nodes (multi-host) is simplified with Kinetic and the [Pathways](https://cloud.google.com/tpu/docs/pathways-overview) backend. This allows you to treat a cluster of TPUs as a single high-performance machine.

## When to Use Distributed Training

A single TPU node (e.g., `tpu-v5litepod-8`, `tpu-v6e-16`) is often enough for many models. Move to multi-host configurations when:
- **Model Size**: The model weights exceed the total TPU memory of a single node.
- **Throughput**: You need to increase global batch size beyond what fits on one node.

## Multi-Host TPU Backend: Pathways

For accelerator configurations spanning more than one node (e.g., `tpu-v2-16`, `tpu-v3-32`, `tpu-v5p-16`, `tpu-v6e-2x4`), Kinetic automatically selects the Pathways backend.

```python
import kinetic

# tpu-v5litepod-2x4 uses two nodes with 4 TPU cores each (8 cores total)
@kinetic.run(accelerator="tpu-v5litepod-2x4")
def train_distributed():
    import jax
    print(f"Total devices across all hosts: {jax.device_count()}")
    # ...
```

## Data Parallelism with Keras

Keras makes it easy to distribute training across multiple TPU devices using `DeviceMesh` and `DataParallel`.

```python
@kinetic.run(accelerator="tpu-v5litepod-2x4", backend="pathways")
def train_data_parallel():
    import keras
    
    # 1. Setup DeviceMesh
    devices = keras.distribution.list_devices()
    device_mesh = keras.distribution.DeviceMesh(
        shape=(len(devices),),
        axis_names=["batch"],
        devices=devices,
    )

    # 2. Set global distribution to DataParallel
    keras.distribution.set_distribution(
        keras.distribution.DataParallel(device_mesh=device_mesh)
    )
    
    # 3. Training code as usual
    model = keras.Sequential([...])
    model.compile(...)
    model.fit(...)
```

## Collective Communication

The Pathways backend handles the complex initialization of the JAX distributed runtime across hosts automatically.

- **Process Isolation**: Each TPU host runs its own instance of your function.
- **Synchronization**: Use standard JAX/Keras collective operations (like `jax.lax.psum`, `keras.distribution.DataParallel`).
- **Unified Results**: Kinetic captures results and logs from all hosts, but only returns the value from the leader process (`jax.process_index() == 0`) to your local machine.

## Debugging Distributed Jobs

When running in a multi-host environment, Kinetic logs include the host identifier to help you trace issues.

```text
[host 0] Starting model.fit...
[host 1] Starting model.fit...
[host 0] Epoch 1/5 - loss: 0.456
[host 1] Epoch 1/5 - loss: 0.457
```

If a job fails on any host, Kinetic catches the exception and re-raises it locally, including the stack trace from the host where the error occurred.
