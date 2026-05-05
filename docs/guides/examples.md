# Examples

A catalog of runnable example scripts using Kinetic. Click any card to open the source code on GitHub.

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
:link: https://github.com/keras-team/kinetic/blob/main/examples/fashion_mnist.py
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
:link: https://github.com/keras-team/kinetic/blob/main/examples/simple_demo.py
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
:link: https://github.com/keras-team/kinetic/blob/main/examples/example_async_jobs.py
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
:link: https://github.com/keras-team/kinetic/blob/main/examples/example_data_api.py
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
:link: https://github.com/keras-team/kinetic/blob/main/examples/example_checkpoint.py
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
:link: https://github.com/keras-team/kinetic/blob/main/examples/example_keras_checkpoint.py
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
:link: https://github.com/keras-team/kinetic/blob/main/examples/example_collections.py
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
:link: https://github.com/keras-team/kinetic/blob/main/examples/example_gke.py
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
:link: https://github.com/keras-team/kinetic/blob/main/examples/pathways_example.py
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
:link: https://github.com/keras-team/kinetic/blob/main/examples/gemma_sft_pathways_distributed.py
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
:link: https://github.com/keras-team/kinetic/blob/main/examples/gemma3_sft_demo.py
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
:link: https://github.com/keras-team/kinetic/blob/main/examples/tunix_sft.py
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
::::

## Related pages

- [Getting Started](../getting_started.md): your first run, end-to-end.
- [Keras Training](keras_training.md): patterns for Keras users.
- [LLM Fine-tuning](llm_finetuning.md): extended walkthrough using the
  Gemma examples.
