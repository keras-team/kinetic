# LLM Post-training with SLIME

[SLIME](https://github.com/THUDM/slime) is an LLM post-training
framework for RL scaling. It connects Megatron for training with
SGLang for rollout generation, and its quickstart covers GRPO-style
training on NVIDIA GPUs.

This guide shows how to run the SLIME quickstart from Kinetic. The key
difference from a local SLIME run is the execution environment: SLIME
ships patched Megatron/SGLang dependencies in its Docker image, so use
that image as a Kinetic prebuilt GPU image and let Kinetic schedule,
stream logs, and resolve checkpoint data with `kinetic.Data(...)`.

:::{note}
SLIME is a CUDA/GPU workflow, not a TPU workflow. Use Kinetic GPU
accelerators such as `gpu-h100x8` or `gpu-a100x8` for this guide.
The upstream SLIME quickstart is validated primarily on H-series
NVIDIA GPUs.
:::

## Prerequisites

Before starting, you need:

- A Kinetic cluster provisioned with `kinetic up`.
- An 8-GPU node pool for the quickstart. H100 is the recommended
  target:

  ```bash
  kinetic pool add --accelerator gpu-h100x8 --project your-project-id
  ```

- An Artifact Registry or Docker Hub repository where Kinetic can push
  the SLIME base image.
- Optional Hugging Face and Weights & Biases credentials in your local
  environment:

  ```bash
  export HF_TOKEN="hf_..."
  export WANDB_API_KEY="..."
  ```

The examples below use the GLM4-9B quickstart from the
[SLIME quickstart](https://thudm.github.io/slime/get_started/quick_start.html).
For production runs, review the upstream SLIME documentation for the
model-specific parallelism and reward settings.

## Build a Kinetic-compatible SLIME Image

Create a `Dockerfile.slime` next to your training launcher:

```dockerfile
FROM slimerl/slime:latest

# Kinetic prebuilt mode installs project requirements at pod startup.
COPY --from=ghcr.io/astral-sh/uv:0.11.1 /uv /uvx /usr/local/bin/

# Runtime dependencies for /app/remote_runner.py.
RUN uv pip install --system \
    absl-py \
    cloudpickle \
    google-cloud-storage

WORKDIR /app
COPY remote_runner.py /app/remote_runner.py

ENV PYTHONUNBUFFERED=1
CMD ["python3"]
```

Build and push it as the GPU prebuilt image for this Kinetic project:

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export KINETIC_SLIME_REPO="us-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT}/kinetic-slime"

gcloud artifacts repositories create kinetic-slime \
  --repository-format=docker \
  --location=us \
  --project="${GOOGLE_CLOUD_PROJECT}"

kinetic build-base \
  --repo "${KINETIC_SLIME_REPO}" \
  --category gpu \
  --dockerfile ./Dockerfile.slime \
  --project "${GOOGLE_CLOUD_PROJECT}" \
  --yes
```

`kinetic build-base` bundles Kinetic's `remote_runner.py` into the build
context, which is why the Dockerfile can copy it into `/app`. After the
build, use this repository with `container_image="prebuilt"` and
`base_image_repo=...`.

## Submit the SLIME Quickstart

Use `@kinetic.submit()` for SLIME runs. RL post-training can run for
hours, so a detached job is easier to inspect and clean up than a
blocking `@kinetic.run()` call.

Create `examples/slime_post_training.py`:

```{literalinclude} ../../examples/slime_post_training.py
:language: python
```

The default `num_rollout=2` is a smoke run. Once the pod reaches the
training loop and writes checkpoints successfully, raise it to the
upstream value or your experiment target.

The checkpoint prefix is passed through `kinetic.Data(...)` at the call
site. Kinetic resolves it to a regular path inside the pod before
`run_slime_glm4_quickstart()` starts:

```python
job = run_slime_glm4_quickstart(
    Data("gs://your-bucket/slime-runs/GLM-Z1-9B-0414_slime/")
)
```

:::{tip}
Run this launcher from a small directory that only contains the launcher
and `Dockerfile.slime`, or keep your `requirements.txt` minimal. The
SLIME image already owns the training stack, and extra project
requirements slow down prebuilt-mode startup.
:::

## Monitor the Run

The launcher prints the Kinetic job ID. From another terminal:

```bash
kinetic jobs status JOB_ID --project your-project-id
kinetic jobs logs --follow JOB_ID --project your-project-id
```

The SLIME script starts a local Ray head process inside the Kinetic pod,
then submits the Megatron/SGLang training job to that local Ray cluster.
The Kinetic job is still the outer unit of scheduling, logging, and
cleanup.

When the function returns, the checkpoint directory is available at the
path resolved from:

```text
gs://your-bucket/slime-runs/GLM-Z1-9B-0414_slime/
```

Use a stable GCS prefix for the `Data(...)` argument when you want later
jobs to start from the same checkpoint tree.

## Scale the Quickstart

The upstream GLM4-9B script splits one 8-GPU node into:

- 4 actor-training GPUs: `--actor-num-nodes 1` and
  `--actor-num-gpus-per-node 4`.
- 4 rollout GPUs: `--rollout-num-gpus 4`.
- SGLang engines with 2 GPUs each:
  `--rollout-num-gpus-per-engine 2`.

For a full run, increase `num_rollout`, tune the batch relationship,
and adjust the checkpoint interval:

```text
rollout-batch-size * n-samples-per-prompt = global-batch-size * num-steps-per-rollout
```

SLIME validates this relationship when `--num-steps-per-rollout` is set.
Keep `--use-dynamic-batch-size` enabled unless you have a reason
to pin micro-batches manually.

## Common Adjustments

- **Using a different supported model:** Change the sourced model config in
the SLIME script, the Hugging Face download target, and the conversion
paths together. SLIME model configs live under `/root/slime/scripts/models`.

- **Using Weights & Biases:** Uncomment `WANDB_ARGS` in the patched script
or maintain your own copy of the run script, then submit with
`capture_env_vars=["WANDB_*"]`.

- **Resume from a previous run:** Pass the previous GCS checkpoint prefix
with `kinetic.Data("gs://.../GLM-Z1-9B-0414_slime/")`, as shown above.
Kinetic resolves it to a filesystem path before starting the training
script.

- **Keep outputs durable during long runs:** Use a stable checkpoint
prefix and follow the checkpointing patterns for your training stack.
For multi-hour jobs, choose a checkpoint cadence that bounds how much
work a restart would lose.

## Related pages

- [PyTorch Training](pytorch_training.md) - basic Kinetic GPU usage.
- [Container Images](../advanced/containers.md) - custom prebuilt
  images and the `kinetic build-base` contract.
- [Detached Jobs](../advanced/async_jobs.md) - monitor long-running
  `@kinetic.submit()` workloads.
- [Checkpointing](checkpointing.md) - durable output patterns.
