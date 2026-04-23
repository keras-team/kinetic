# Native JAX Training

**Who this is for:** users who write training loops directly in JAX
rather than going through Keras. Kinetic runs your JAX code on cloud
TPUs and GPUs the same way it runs Keras code — wrap the function in
`@kinetic.run()` and call it. JAX-specific details (multi-device
parallelism, dependency filtering, multi-host coordination) are covered
below.

## A first run

```python
import kinetic

@kinetic.run(accelerator="tpu-v5litepod-8")
def jax_computation():
    import jax
    import jax.numpy as jnp

    print(f"Devices: {jax.devices()}")

    x = jnp.ones((1000, 1000))
    result = jnp.dot(x, x)
    return float(result[0, 0])

print(jax_computation())  # 1000.0
```

A standard JAX training loop with `jax.grad` runs without modification:

```python
@kinetic.run(accelerator="tpu-v6e-8")
def train():
    import jax
    import jax.numpy as jnp

    def loss_fn(params, x, y):
        pred = x @ params["w"] + params["b"]
        return jnp.mean((pred - y) ** 2)

    grad_fn = jax.grad(loss_fn)

    key = jax.random.PRNGKey(0)
    params = {"w": jax.random.normal(key, (10, 1)), "b": jnp.zeros(1)}
    x = jax.random.normal(key, (512, 10))
    y = x @ jnp.ones((10, 1)) + 0.1 * jax.random.normal(key, (512, 1))

    lr = 0.01
    for step in range(200):
        grads = grad_fn(params, x, y)
        params = {k: params[k] - lr * grads[k] for k in params}
        if step % 50 == 0:
            print(f"step {step}: loss={loss_fn(params, x, y):.4f}")

    return float(loss_fn(params, x, y))
```

Imports for `jax`, `jaxlib`, and any other heavy library go **inside**
the decorated function so the remote worker uses its accelerator-tuned
install.

## How to think about it

JAX needs the right `jaxlib` and the right accelerator runtime
(`libtpu`, CUDA) to be installed in the container. Kinetic handles this
for you:

- **Bundled and prebuilt images** ship with JAX matched to the
  accelerator type. You don't need to pin `jax`, `jaxlib`, or `libtpu`
  in `requirements.txt`.
- **JAX packages in your `requirements.txt` are filtered out** before
  install so they don't shadow the accelerator-correct copy in the
  image. See [Dependencies](../guides/dependencies.md) for the filter behavior.

Inside the function, `jax.devices()` returns whatever the pod sees: an
8-chip TPU slice for `tpu-v6e-8`, an 8-device array for
`tpu-v5litepod-8`, a single GPU for `l4`, etc.

## Single-host parallelism

Use `jax.pmap` (or `jax.sharding`) to spread computation across all
devices on a single host:

```python
@kinetic.run(accelerator="tpu-v5litepod-8")
def parallel_computation():
    import jax
    import jax.numpy as jnp

    n_devices = jax.local_device_count()
    print(f"Running on {n_devices} devices")

    @jax.pmap
    def parallel_matmul(x):
        return jnp.dot(x, x.T)

    data = jnp.ones((n_devices, 256, 256))
    result = parallel_matmul(data)
    return float(result[0, 0, 0])
```

## Scaling beyond a single host

For multi-host slices (e.g., `tpu-v5litepod-2x4`) JAX needs a coordination
runtime to set up cross-host collectives. Kinetic provides this through
the Pathways backend:

```python
@kinetic.run(accelerator="tpu-v5litepod-2x4", backend="pathways")
def train_distributed():
    import jax
    # jax.process_count() > 1 here; pmap/sharding work cross-host.
    ...
```

Without `backend="pathways"`, multi-host JAX collectives won't have a
working coordinator. See [Distributed Training](../guides/distributed_training.md)
for the full multi-host setup.

## Data

To pass a dataset into a remote JAX function, construct a
`kinetic.Data(...)` object **at the call site** in your local script and
pass it as an argument. Kinetic uploads (or mounts) the source and
delivers a plain filesystem path to the remote function. The decorated
function only ever sees a `str` path:

```python
import kinetic
from kinetic import Data

@kinetic.run(accelerator="tpu-v6e-8")
def train(data_dir):
    # `data_dir` is a local filesystem path on the remote pod.
    import os
    files = os.listdir(data_dir)
    ...

# Local directory:
train(Data("./my_dataset/"))

# Existing GCS bucket:
train(Data("gs://my-bucket/dataset/"))

# Large GCS dataset, streamed on demand via FUSE:
train(Data("gs://my-bucket/large/", fuse=True))
```

`Data` accepts both local paths and `gs://` URIs. See [Data](../guides/data.md)
for the decision matrix between downloaded, FUSE-mounted, and direct
access patterns.

## Next steps

- [Distributed Training](../guides/distributed_training.md) — multi-host JAX with
  Pathways.
- [Checkpointing](../guides/checkpointing.md) — Orbax checkpoint patterns under
  `KINETIC_OUTPUT_DIR`.

## Related pages

- [Distributed Training](../guides/distributed_training.md) — Pathways and
  multi-host coordination.
- [Dependencies](../guides/dependencies.md) — JAX filtering and what gets
  installed.
- [Checkpointing](../guides/checkpointing.md) — Orbax + `KINETIC_OUTPUT_DIR`.
