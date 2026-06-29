import kinetic


@kinetic.run(accelerator="tpu-v5litepod-1x1")
def train_and_profile():
  """Capture an XProf trace of a few JAX training steps, saved to GCS."""
  import os

  import jax
  import jax.numpy as jnp

  trace_dir = os.path.join(
    os.environ.get("KINETIC_OUTPUT_DIR", "/tmp/kinetic-out"), "profile"
  )

  # A small MLP, params held as a pytree.
  key = jax.random.PRNGKey(0)
  k1, k2, k3 = jax.random.split(key, 3)
  params = {
    "w1": jax.random.normal(k1, (1024, 2048)) * 0.02,
    "w2": jax.random.normal(k2, (2048, 2048)) * 0.02,
    "w3": jax.random.normal(k3, (2048, 10)) * 0.02,
  }
  x = jax.random.normal(key, (4096, 1024))
  y = jax.random.normal(key, (4096, 10))

  def loss_fn(p, x, y):
    h = jnp.tanh(x @ p["w1"])
    h = jnp.tanh(h @ p["w2"])
    pred = h @ p["w3"]
    return jnp.mean((pred - y) ** 2)

  @jax.jit
  def update(p, x, y, lr=1e-3):
    grads = jax.grad(loss_fn)(p, x, y)
    return {k: p[k] - lr * grads[k] for k in p}

  # Warm up once so XLA compilation doesn't pollute the trace.
  params = update(params, x, y)
  jax.block_until_ready(params)

  # The context manager flushes the trace even if a step raises.
  with jax.profiler.trace(trace_dir):
    for _ in range(10):
      params = update(params, x, y)
    # force the async work to land before the trace closes
    jax.block_until_ready(params)

  print(f"final loss: {float(loss_fn(params, x, y)):.4f}")
  print(f"trace written to: {trace_dir}")
  return trace_dir


if __name__ == "__main__":
  train_and_profile()
