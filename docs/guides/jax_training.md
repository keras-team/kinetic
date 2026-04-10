# Native JAX Training

Kinetic works with pure JAX code, not just Keras. If you prefer writing training loops directly with JAX, you can run them on cloud TPUs and GPUs the same way.

## Basic Usage

Wrap your JAX code in a decorated function. Import JAX inside the function so the remote worker picks up the hardware-optimized installation.

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

## Training Loop

A standard JAX training loop with `jax.grad` runs without modification.

```python
import kinetic

@kinetic.run(accelerator="tpu-v6e-8")
def train():
    import jax
    import jax.numpy as jnp

    # Simple linear regression
    def loss_fn(params, x, y):
        pred = x @ params["w"] + params["b"]
        return jnp.mean((pred - y) ** 2)

    grad_fn = jax.grad(loss_fn)

    key = jax.random.PRNGKey(0)
    params = {
        "w": jax.random.normal(key, (10, 1)),
        "b": jnp.zeros(1),
    }

    # Dummy data
    x = jax.random.normal(key, (512, 10))
    y = x @ jnp.ones((10, 1)) + 0.1 * jax.random.normal(key, (512, 1))

    lr = 0.01
    for step in range(200):
        grads = grad_fn(params, x, y)
        params = {k: params[k] - lr * grads[k] for k in params}
        if step % 50 == 0:
            print(f"step {step}: loss={loss_fn(params, x, y):.4f}")

    return float(loss_fn(params, x, y))

final_loss = train()
```

## Multi-Device Parallelism

Use `jax.pmap` or `jax.sharding` to spread computation across all available devices on a single host.

```python
import kinetic

@kinetic.run(accelerator="tpu-v5litepod-8")
def parallel_computation():
    import jax
    import jax.numpy as jnp

    n_devices = jax.local_device_count()
    print(f"Running on {n_devices} devices")

    @jax.pmap
    def parallel_matmul(x):
        return jnp.dot(x, x.T)

    # Shape: (n_devices, 256, 256) -- one slice per device
    data = jnp.ones((n_devices, 256, 256))
    result = parallel_matmul(data)
    return float(result[0, 0, 0])
```

For multi-host configurations, see the [Distributed Training](distributed_training.md) guide.

## Dependencies

JAX and its accelerator libraries (`jaxlib`, `libtpu`) are pre-installed on remote workers and automatically filtered from your `requirements.txt`. See [Managing Dependencies](dependencies.md) for details.
