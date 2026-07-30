# RL Post-training with verl

[verl](https://verl.readthedocs.io/en/latest/) is an RL post-training
framework for LLMs. It supports PPO, GRPO, DAPO, and related algorithms
on top of common training and rollout backends such as FSDP, Megatron,
vLLM, and SGLang.

This guide focuses only on the Kinetic parts of a verl run: building a
compatible GPU image, submitting a detached job, passing credentials,
staging input data, and keeping checkpoints durable. For algorithm
choices, reward design, and model-specific tuning, use the upstream verl
documentation.

:::{note}
verl RL jobs are CUDA/GPU workloads, not TPU workloads. Use Kinetic GPU
accelerators such as `gpu-h100`, `gpu-h100x8`, or `gpu-a100x8`.
:::

## Prerequisites

Before starting, you need:

- A Kinetic cluster provisioned with `kinetic up`.
- A GPU node pool for the size of run you want:

  ```bash
  kinetic pool add --accelerator gpu-h100 --project your-project-id
  ```

- An Artifact Registry or Docker Hub repository where Kinetic can push a
  prebuilt GPU image.
- Optional Hugging Face and Weights & Biases credentials in your local
  environment:

  ```bash
  export HF_TOKEN="hf_..."
  export WANDB_API_KEY="..."
  ```

The example below adapts verl's
[GSM8K PPO quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html).
It is a smoke-run template, not a recommendation about which algorithm
or reward to use in production.

## Build a Kinetic-compatible verl Image

verl's dependency stack is large and tightly coupled to CUDA, PyTorch,
and the rollout backend. Use Kinetic prebuilt mode and start from one of
the official verl Docker images instead of asking Kinetic to resolve
those packages from a plain `requirements.txt`.

Create `Dockerfile.verl` next to your launcher:

```dockerfile
FROM verlai/verl:vllm011.latest

# Kinetic prebuilt mode installs project requirements at pod startup.
COPY --from=ghcr.io/astral-sh/uv:0.11.1 /uv /uvx /usr/local/bin/

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# The published verl images carry the heavy CUDA/runtime dependencies.
# Install verl itself editable so examples, preprocessors, and trainer
# entrypoints are available inside the remote job.
ARG VERL_REF=main
RUN git clone https://github.com/verl-project/verl.git /opt/verl && \
    cd /opt/verl && \
    git checkout "${VERL_REF}" && \
    pip3 install --no-deps -e . && \
    pip3 install google-cloud-storage cloudpickle absl-py

WORKDIR /app
COPY remote_runner.py /app/remote_runner.py

ENV PYTHONUNBUFFERED=1
ENV VLLM_USE_V1=1
CMD ["python3"]
```

Build and publish it as the GPU prebuilt image for this Kinetic project:

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export KINETIC_VERL_REPO="us-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT}/kn-your-cluster-name"

gcloud artifacts repositories create kn-your-cluster-name \
  --repository-format=docker \
  --location=us \
  --project="${GOOGLE_CLOUD_PROJECT}"

kinetic build-base \
  --repo "${KINETIC_VERL_REPO}" \
  --category gpu \
  --dockerfile ./Dockerfile.verl \
  --project "${GOOGLE_CLOUD_PROJECT}" \
  --yes
```

For reproducible experiments, pin `VERL_REF` to a release tag or commit
and pin the `verlai/verl` image tag you validated.

## Submit a verl Smoke Run

Use `@kinetic.submit()` for RL runs. verl launches Ray workers inside the
pod, and the outer Kinetic job remains the unit you monitor, clean up,
and reattach to.

Create `examples/verl_rl.py`:

```{literalinclude} ../../examples/verl_rl.py
:language: python
```

The defaults intentionally run on a small sample. Once the image,
credentials, model download, Ray startup, and checkpoint writes all work,
raise `train_max_samples`, `val_max_samples`, `trainer.total_epochs`,
and the batch sizes for the real experiment.

To use a dataset you already prepared, pass it with `kinetic.Data(...)`
at the call site. Kinetic resolves it to a normal path inside the pod,
and the trainer still receives local parquet paths:

```python
job = run_verl_gsm8k_ppo(
    checkpoint_dir=kinetic.Data("gs://your-bucket/verl-checkpoints/", fuse=True),
    prepared_data_dir=kinetic.Data(
        "gs://your-project-id-kn-your-cluster-name-data/gsm8k/",
        fuse=True,
    ),
)
```

## Resume from a Kinetic Checkpoint

verl's FSDP checkpoints are a directory tree under
`trainer.default_local_dir`, with `latest_checkpointed_iteration.txt`
tracking the latest saved step. The example passes the checkpoint root
through `kinetic.Data(...)`, so Kinetic resolves the checkpoint prefix to
a regular filesystem path before the job starts.

Pass the stable checkpoint prefix at the call site:

```python
job = run_verl_gsm8k_ppo(
    checkpoint_dir=kinetic.Data("gs://your-bucket/verl-checkpoints/", fuse=True),
)
```

With `trainer.resume_mode=auto`, verl resumes from the latest checkpoint
found under `kinetic-verl/gsm8k-ppo` inside that resolved directory.

For long runs, choose a checkpoint cadence that bounds how much work a
restart would lose.

## Kinetic-specific Scaling Knobs

Keep these settings aligned when you scale beyond the smoke run:

- `accelerator` and `trainer.n_gpus_per_node`: use `gpu-h100x8` with
  `trainer.n_gpus_per_node=8`, `gpu-a100x4` with
  `trainer.n_gpus_per_node=4`, and so on.
- `trainer.nnodes`: keep this at `1` for single-node Kinetic GPU jobs.
  Use upstream verl multi-node guidance before trying multi-node RL.
- `actor_rollout_ref.rollout.tensor_model_parallel_size`: increase this
  when the rollout model itself needs multiple GPUs.
- `actor_rollout_ref.rollout.gpu_memory_utilization`: lower this if vLLM
  competes with FSDP for memory on the same GPU set.
- `data.train_files` and `data.val_files`: use local paths inside the
  pod. For large prepared datasets, pass them with
  `kinetic.Data("gs://...", fuse=True)` rather than downloading them in
  the function.
- `trainer.default_local_dir`: keep it under the path resolved from
  `kinetic.Data(...)` so resume logic sees the same checkpoint tree.
- `capture_env_vars`: pass only the credentials the job needs, usually
  `HF_TOKEN` and optionally `WANDB_*`.

## Monitor the Run

The launcher prints a Kinetic job ID:

```bash
kinetic jobs status JOB_ID --project your-project-id
kinetic jobs logs --follow JOB_ID --project your-project-id
```

verl emits trainer metrics to the console when `trainer.logger=console`.
If you switch to W&B, keep `capture_env_vars=["WANDB_*"]` and set the
verl logger override accordingly.

When the function returns, checkpoints are available under the resolved
checkpoint path:

```text
gs://your-bucket/verl-checkpoints/kinetic-verl/gsm8k-ppo
```

## Related pages

- [PyTorch Training](pytorch_training.md) - basic Kinetic GPU usage.
- [Container Images](../guides/containers.md) - custom prebuilt images
  and the `kinetic build-base` contract.
- [Detached Jobs](../guides/async_jobs.md) - monitor long-running
  `@kinetic.submit()` workloads.
- [Checkpointing](../guides/checkpointing.md) - durable output patterns.
