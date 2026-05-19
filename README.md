# Kinetic

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI - Version](https://img.shields.io/pypi/v/keras-kinetic)](https://pypi.org/project/keras-kinetic/)

Run Keras and JAX workloads on cloud TPUs and GPUs with a simple decorator. No infrastructure management required.

```python
import kinetic

@kinetic.run(accelerator="tpu-v5e-1")
def train_model():
    import keras
    model = keras.Sequential([...])
    model.fit(x_train, y_train)
    return model.history.history["loss"][-1]

# Executes on a TPU v5e-1 slice, returns the result locally
final_loss = train_model()
```

## Why Kinetic

- **Simple remote execution.** A `@kinetic.run()` decorator runs the
  function on the accelerator you ask for and returns the result.
  Nothing else changes about your code.
- **Detached jobs.** Switch to `@kinetic.submit()` for long runs.
  You get a `JobHandle` back — poll status, tail logs, collect the
  result later, or reattach from another machine entirely.
- **Data and checkpoint support.** Wrap inputs in `kinetic.Data(...)`
  to ship local files (or stream from GCS) into the job. Write durable
  outputs and resumable checkpoints under `KINETIC_OUTPUT_DIR`.

## Documentation
Comprehensive documentation is available at: https://kinetic.readthedocs.io

## Install

```bash
uv pip install "keras-kinetic[cli]"
```

The base `keras-kinetic` package installs the `@kinetic.run()`
decorator. The `[cli]` extra adds the dependencies the `kinetic` CLI
needs to provision and manage infrastructure. Drop the `[cli]` extra
only if you just need to submit jobs against an already-provisioned
cluster.

## One-time setup

```bash
kinetic init
```

This detects your local environment, then either joins an existing
Kinetic cluster in the project (your own or a teammate's — discovery
goes through the shared state bucket) or walks you through creating a
new one. It ends by saving a profile that becomes your active context
— subsequent commands pick up project, zone, and cluster automatically.

Behind the scenes, the Create path runs `kinetic up` to enable APIs,
provision a GKE cluster with an accelerator node pool, and configure
local Docker / `kubectl` access. Run `kinetic down` when you're done.

## Recommended first run

```bash
python examples/fashion_mnist.py
```

No environment variables needed — `kinetic init` set an active
profile. The first run takes ~5 minutes (it builds a container image
with your dependencies via Cloud Build). Subsequent runs with
unchanged dependencies start in under a minute.

For the full first-run walkthrough, see the
[Getting Started](https://kinetic.readthedocs.io/en/latest/getting_started.html)
guide.

## Where to go next

| Question                                         | Where to look                                                                                                                                             |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| How do I get my first job running?               | [Getting Started](https://kinetic.readthedocs.io/en/latest/getting_started.html)                                                                          |
| When should I use `submit()` instead of `run()`? | [Detached Jobs](https://kinetic.readthedocs.io/en/latest/advanced/async_jobs.html)                                                                        |
| How do I ship data and persist outputs?          | [Data](https://kinetic.readthedocs.io/en/latest/guides/data.html) and [Checkpointing](https://kinetic.readthedocs.io/en/latest/guides/checkpointing.html) |
| Bundled vs prebuilt vs custom image — which one? | [Execution Modes](https://kinetic.readthedocs.io/en/latest/guides/execution_modes.html)                                                                   |
| Something's broken; where do I start?            | [Troubleshooting](https://kinetic.readthedocs.io/en/latest/troubleshooting.html)                                                                          |

## Configuration

The recommended way to configure Kinetic is via a profile — the named
context that `kinetic init` creates and `kinetic profile ls | use`
manages. For ad-hoc overrides, every profile field also has a
`KINETIC_*` env-var equivalent (`KINETIC_PROJECT`, `KINETIC_ZONE`,
`KINETIC_CLUSTER`, `KINETIC_NAMESPACE`) and a matching CLI flag.

Precedence is: **CLI flag > `KINETIC_*` env var > active profile >
built-in default.**

The full surface — every variable, every CLI flag, and the profile
model — lives in the
[Configuration reference](https://kinetic.readthedocs.io/en/latest/configuration.html).

## Contributing

See the [Contributing guide](https://kinetic.readthedocs.io/en/latest/contributing.html).

## License

[Apache 2.0](LICENSE)
