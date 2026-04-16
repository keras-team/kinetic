# Kinetic

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

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

## Install

```bash
uv pip install keras-kinetic
```

This installs both the decorator and the `kinetic` CLI.

## One-time setup

If nobody on your team has provisioned a Kinetic cluster yet, run:

```bash
kinetic up
```

This enables the required GCP APIs, creates an Artifact Registry
repository, provisions a GKE cluster with an accelerator node pool,
and configures local Docker / `kubectl` access. Run `kinetic down`
when you're finished to tear everything back down.

## Recommended first run

```bash
export KINETIC_PROJECT="your-gcp-project-id"
python examples/fashion_mnist.py
```

The first run takes ~5 minutes (it builds a container image with your
dependencies via Cloud Build). Subsequent runs with unchanged
dependencies start in under a minute.

For the full first-run walkthrough, see the
[Getting Started](https://keras-team.github.io/kinetic/getting_started.html)
guide.

## Where to go next

| Question                                         | Where to look                                                                                                                                     |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| How do I get my first job running?               | [Getting Started](https://keras-team.github.io/kinetic/getting_started.html)                                                                      |
| When should I use `submit()` instead of `run()`? | [Detached Jobs](https://keras-team.github.io/kinetic/advanced/async_jobs.html)                                                                    |
| How do I ship data and persist outputs?          | [Data](https://keras-team.github.io/kinetic/guides/data.html) and [Checkpointing](https://keras-team.github.io/kinetic/guides/checkpointing.html) |
| Bundled vs prebuilt vs custom image — which one? | [Execution Modes](https://keras-team.github.io/kinetic/guides/execution_modes.html)                                                               |
| Something's broken; where do I start?            | [Troubleshooting](https://keras-team.github.io/kinetic/troubleshooting.html)                                                                      |

## Configuration

Kinetic reads `KINETIC_PROJECT` (required), `KINETIC_ZONE`,
`KINETIC_CLUSTER`, and a handful of other environment variables. The
short version:

```bash
export KINETIC_PROJECT="your-project-id"      # required
export KINETIC_ZONE="us-central1-a"           # optional
export KINETIC_CLUSTER="kinetic-cluster"      # optional
```

The full surface — every variable, every CLI flag, and how the
precedence rules combine them — lives in the
[Configuration reference](https://keras-team.github.io/kinetic/configuration.html).

## Contributing

See the [Contributing guide](https://keras-team.github.io/kinetic/contributing.html).

## License

[Apache 2.0](LICENSE)
