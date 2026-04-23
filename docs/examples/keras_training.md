# Training Keras Models

**Who this is for:** anyone with a working Keras training script who wants
it to run on a cloud TPU or GPU without standing up infrastructure.
Kinetic ships your existing `model.compile()` / `model.fit()` code to a
remote accelerator with a single decorator change. You don't need to
restructure your training loop.

## A first run

```python
import kinetic

@kinetic.run(accelerator="tpu-v6e-8")
def train_model():
    import keras
    import numpy as np

    model = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(10,)),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    x_train = np.random.randn(1000, 10)
    y_train = np.random.randn(1000, 1)

    history = model.fit(x_train, y_train, epochs=5, verbose=0)
    return history.history["loss"][-1]

final_loss = train_model()
print(f"Final loss: {final_loss}")
```

A few things to note:

- Imports for `keras`, `jax`, etc. live **inside** the function so the
  remote worker uses its hardware-tuned install.
- The return value is serialized back to your local process. Keep it
  small — a final metric, a path under `KINETIC_OUTPUT_DIR`, a dict of
  numbers. Don't return the model object itself.
- `accelerator="tpu-v6e-8"` picks an 8-chip TPU v6e slice. Use `cpu` while
  iterating; switch when you're ready for hardware. See
  [Accelerators](../accelerators.md).

For the canonical end-to-end example with a real dataset, see
[`fashion_mnist.py`](../examples.md) (first entry under Quickstart).

## How to think about it

Your decorated function runs in a fresh process inside a container on a
remote node. That has two practical consequences:

- **No local state crosses the boundary.** Anything the function needs
  must either be passed as an argument, captured by closure, or shipped
  via [`kinetic.Data`](../guides/data.md). Locally-loaded variables that you reference
  by global name will not be there on the remote.
- **The Keras backend is whatever the remote has installed.** By default
  Kinetic's prebuilt and bundled images use JAX. Set `KERAS_BACKEND` if
  you need otherwise:

  ```python
  @kinetic.run(accelerator="tpu-v6e-8", capture_env_vars=["KERAS_BACKEND"])
  def train(): ...
  ```

## Scaling beyond a single host

For multi-host TPU slices like `tpu-v5litepod-2x4`, switch to the Pathways
backend so Keras's distribution strategies have a working multi-host
runtime to talk to:

```python
@kinetic.run(accelerator="tpu-v5litepod-2x4", backend="pathways")
def train_distributed():
    ...
```

See [Distributed Training](../guides/distributed_training.md) for the full
multi-host setup, and [LLM Fine-tuning](llm_finetuning.md) for a
concrete Gemma example.

## Data

Pulling NumPy arrays from inside the function works for tiny datasets,
but breaks down quickly. For real data, construct a
`kinetic.Data(...)` object **at the call site** in your local script
and pass it as an argument. Kinetic uploads (or mounts) the source and
delivers a plain filesystem path to the remote function. The decorated
function only ever sees a `str` path:

```python
import kinetic
from kinetic import Data

@kinetic.run(accelerator="tpu-v6e-8")
def train(data_dir):
    # `data_dir` is a local filesystem path on the remote pod.
    import keras
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

- [`fashion_mnist.py`](../examples.md) — full working example with a real
  dataset (first entry under Quickstart).
- [Checkpointing](../guides/checkpointing.md) — persist model weights and resume
  across runs.

## Related pages

- [Data](../guides/data.md) — shipping local files and reading from GCS.
- [Checkpointing](../guides/checkpointing.md) — `KINETIC_OUTPUT_DIR` and resumable
  training.
- [LLM Fine-tuning](llm_finetuning.md) — KerasHub + Gemma walkthrough.
