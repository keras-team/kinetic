# Examples

```{toctree}
:hidden:

examples/fashion_mnist
examples/simple_demo
examples/example_async_jobs
examples/example_data_api
examples/example_checkpoint
examples/example_keras_checkpoint
examples/example_collections
examples/example_gke
examples/pathways_example
examples/gemma_sft_pathways_distributed
examples/gemma3_sft_demo
examples/tunix_sft
examples/tunix_grpo
```

A catalog of runnable example scripts using Kinetic. Every example below is rendered directly on this site and is also available as a raw Python script in the GitHub repository.

Tier badges:

- **Quickstart:** your first run. Minimal setup, sensible defaults.
- **Core:** the everyday product surface: async jobs, data, checkpoints,
  parallel sweeps.
- **Advanced:** multi-host Pathways jobs, LLM fine-tuning, anything that
  needs special quota or external credentials.

To run any example: clone the repo, install Kinetic, set `KINETIC_PROJECT`,
and `python examples/<file>.py`.

```bash
git clone https://github.com/keras-team/kinetic.git
cd kinetic
uv pip install -e .
export KINETIC_PROJECT="your-project-id"
python examples/fashion_mnist.py
```

## Quickstart

::::{grid} 1 2 2 3
:gutter: 3
:class-container: sd-text-left

:::{grid-item-card} Fashion-MNIST on a TPU
:link: examples/fashion_mnist.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

The first thing to run after `kinetic up`. A small Keras classifier on
Fashion-MNIST that confirms your cluster can schedule a TPU pod and
stream a real result back to your shell.

+++

{bdg-secondary}`Keras` &nbsp;
{bdg-secondary}`TPU`
:::

:::{grid-item-card} Keras + JAX smoke test
:link: examples/simple_demo.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

The cheapest sanity check there is. Keras-on-JAX on a CPU node — no
accelerator quota needed, useful for verifying your install before you
ask for hardware.

+++

{bdg-secondary}`Keras` &nbsp;
{bdg-secondary}`JAX` &nbsp;
{bdg-secondary}`CPU`
:::
::::

## Core

::::{grid} 1 2 2 3
:gutter: 3
:class-container: sd-text-left

:::{grid-item-card} Submit, monitor, and reattach
:link: examples/example_async_jobs.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

Walks through every part of the detached-job API end-to-end: `submit()`,
`status()`/`tail()`/`result()`, reattach from another shell with
`kinetic.attach()`, and enumerate jobs with `list_jobs()`.

+++

{bdg-secondary}`Async` &nbsp;
{bdg-secondary}`Reattach`
:::

:::{grid-item-card} Ship local files into the job
:link: examples/example_data_api.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

Wrap a local directory in `kinetic.Data(...)` and let it land as a
plain filesystem path on the remote — your training code doesn't have
to know whether the bytes started on your laptop or in GCS.

+++

{bdg-secondary}`Data` &nbsp;
{bdg-secondary}`GCS`
:::

:::{grid-item-card} Resumable JAX training with Orbax
:link: examples/example_checkpoint.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

JAX training that picks up where it left off. Writes Orbax checkpoints
to `KINETIC_OUTPUT_DIR` and proves the resume path by relaunching the
same function and seeing it skip already-completed steps.

+++

{bdg-secondary}`JAX` &nbsp;
{bdg-secondary}`Checkpointing` &nbsp;
{bdg-secondary}`Orbax`
:::

:::{grid-item-card} Resumable Keras training
:link: examples/example_keras_checkpoint.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

Auto-resumable Keras training. Round-trips `model.get_weights()` through
Orbax so a restarted job picks up at the right step without any custom
save/load code.

+++

{bdg-secondary}`Keras` &nbsp;
{bdg-secondary}`Checkpointing` &nbsp;
{bdg-secondary}`Orbax`
:::

:::{grid-item-card} Parallel hyperparameter sweep
:link: examples/example_collections.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

Fan out a grid of jobs with `kinetic.map()`, batch submissions to keep
the cluster happy, and gather results — including how to handle the
job that inevitably fails halfway through.

+++

{bdg-secondary}`Sweep` &nbsp;
{bdg-secondary}`Parallel`
:::

:::{grid-item-card} Mix accelerators in one driver
:link: examples/example_gke.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

One driver script that successively schedules work on CPU, TPU, and
GPU pools — handy for verifying which hardware your cluster will
actually serve.

+++

{bdg-secondary}`Multi-accelerator` &nbsp;
{bdg-secondary}`Cluster`
:::
::::

## Advanced

::::{grid} 1 2 2 3
:gutter: 3
:class-container: sd-text-left

:::{grid-item-card} Multi-host JAX on Pathways
:link: examples/pathways_example.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

The reference for scaling beyond a single TPU host. A short JAX program
that verifies cross-host collectives are actually wired up before you
trust them with a real workload.

+++

{bdg-secondary}`JAX` &nbsp;
{bdg-secondary}`Pathways` &nbsp;
{bdg-secondary}`Distributed`
:::

:::{grid-item-card} Distributed Gemma 2B fine-tune
:link: examples/gemma_sft_pathways_distributed.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

End-to-end SFT of Gemma 2B with LoRA across multiple TPU hosts. The
realistic LLM workload to model your own fine-tuning runs after — pulls
weights from Kaggle and runs on Pathways.

+++

{bdg-secondary}`LLM` &nbsp;
{bdg-secondary}`Pathways` &nbsp;
{bdg-secondary}`Distributed`
:::

:::{grid-item-card} Single-TPU Gemma 3 fine-tune
:link: examples/gemma3_sft_demo.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

Compact Gemma 3 1B SFT on a single TPU. A good baseline for getting an
LLM workload running before scaling out to Pathways, and a worked
example of forwarding Kaggle credentials into the remote pod.

+++

{bdg-secondary}`LLM` &nbsp;
{bdg-secondary}`TPU`
:::

:::{grid-item-card} Tunix SFT Example
:link: examples/tunix_sft.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

SFT of Gemma 3 with LoRA/QLoRA on TPU v5litepod. Demonstrates how to
run the Tunix SFT script on a remote cluster with environment variable
capture for credentials.

+++

{bdg-secondary}`LLM` &nbsp;
{bdg-secondary}`TPU` &nbsp;
{bdg-secondary}`LoRA`
:::

:::{grid-item-card} Tunix GRPO Example
:link: examples/tunix_grpo.md
:class-card: sd-shadow-sm
:class-body: sd-fs-6
:class-title: sd-fs-5

GRPO post-training for Gemma with Tunix on a Kinetic TPU job. Shows the
Data API for GSM8K-style inputs, detached execution, reward functions,
and durable Orbax checkpoints under `KINETIC_OUTPUT_DIR`.

+++

{bdg-secondary}`LLM` &nbsp;
{bdg-secondary}`TPU` &nbsp;
{bdg-secondary}`GRPO`
:::
::::

## Related pages

- [Getting Started](getting_started.md): your first run, end-to-end.
- [Keras Training](examples/keras_training.md): patterns for Keras users.
- [LLM Fine-tuning](examples/llm_finetuning.md): extended walkthrough using the
  Gemma examples.
