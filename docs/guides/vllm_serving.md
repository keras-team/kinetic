# Serving KerasHub models with vLLM using Kinetic

## Overview

Export a KerasHub model to the Hugging Face Transformers format and serve it 
with [vLLM](https://docs.vllm.ai) — on a Cloud**TPU** or **GPU**, in a single 
Kinetic job, with any Keras backend. KerasHub
causal LMs export natively with `export_to_transformers()`, producing a standard
Hugging Face checkpoint (config, safetensors weights, tokenizer) that is
independent of the backend used to create it. Any preset with a Transformers
exporter works (Gemma, Gemma 3, Qwen, GPT-2, …).

The export runs in a short-lived child process: when it exits, the OS releases
its memory and the device is clean for vLLM. This keeps the export and serving
stacks isolated — so any Keras backend works — while staying a single Kinetic
job (the export is just a subprocess inside the pod).

## Prerequisites

1. **A node pool** for your accelerator (default scale-to-zero). Pick any
   slice/GPU that fits your model:

   ```bash
   kinetic pool add --accelerator tpu-v5litepod-1 --project your-project-id   # TPU
   kinetic pool add --accelerator gpu-l4          --project your-project-id   # GPU
   ```

   Larger models need a bigger slice (e.g. `tpu-v5litepod-4`, `gpu-a100`).
   Make sure your project has matching **quota**, or the pod will sit `Pending`.

2. **A `requirements.txt`** next to the scripts. The base set depends on the
   device; a non-default export backend adds one entry:

   | Device | base requirements           | backend extras                                                   |
   |--------|-----------------------------|------------------------------------------------------------------|
   | TPU    | `keras keras-hub vllm-tpu`  | `jax` / `torch`: none · `tensorflow`: add `tensorflow`           |
   | GPU    | `keras keras-hub vllm`      | `torch`: none · `jax`: add `jax[cuda12]` · `tensorflow`: add `tensorflow[and-cuda]` |

   The default backend per device (`jax` on TPU, `torch` on GPU) needs no extras.
   On TPU, `torch`/`tensorflow` exports run on the host CPU, leaving the chip for
   vLLM. Kinetic builds the remote container to match your **local Python
   version**, so use one with `vllm`/`vllm-tpu` wheels available (3.10–3.12).

3. **Kaggle credentials** for gated models like Gemma: accept the
   [license](https://www.kaggle.com/models/keras/gemma3) and set
   `KAGGLE_USERNAME` / `KAGGLE_KEY` locally. Kinetic forwards them via
   `capture_env_vars`.

## The example

Two files: `vllm_serving.py` (the orchestrator — set `DEVICE`, `BACKEND`,
`MODEL_PRESET`, and `ACCELERATOR` at the top) and `export_worker.py` (the
standalone export, run as a subprocess).

```{literalinclude} ../../examples/export_worker.py
:language: python
:caption: examples/export_worker.py
```

```{literalinclude} ../../examples/vllm_serving.py
:language: python
:caption: examples/vllm_serving.py
```

On TPU the example sets `JAX_PLATFORMS=tpu,cpu` and runs the engine in-process
(`VLLM_ENABLE_V1_MULTIPROCESSING=0`); on GPU it exposes the NVIDIA driver before
the export. `find_spec("export_worker")` locates the worker on the pod (Kinetic
unpacks the job's files onto `sys.path`) without importing it.

## Running

```bash
python vllm_serving.py
```

The first run builds the container image (15–25 minutes; later runs reuse it as
a cache hit) and provisions a node from the scale-to-zero pool. Monitor from a
second terminal with `kinetic jobs list` and
`kinetic jobs logs --follow JOB_ID --project your-project-id`.

## Single job vs. two jobs

This example is one Kinetic job — one pod — that runs the export as a child
process (two processes, one pod). To **export once and serve the same checkpoint
many times**, split it into two jobs instead: an export job that uploads the
checkpoint to `KINETIC_OUTPUT_DIR` (a GCS path), and a serve job that downloads
and serves it. Each job gets its own container, which also lets `vllm` and
`vllm-tpu` live in separate images.
