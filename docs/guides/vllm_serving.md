# Serving KerasHub Models with vLLM

This guide explains how to export a KerasHub model to the Hugging Face
Transformers format and serve it with vLLM using the Kinetic framework —
on a Cloud GPU in a single job, or on TPU as a two-job workflow.

## Overview

KerasHub causal LMs can be exported natively with
`export_to_transformers()`, producing a standard Hugging Face checkpoint
(config, safetensors weights, tokenizer). Kinetic lets you run the export
and vLLM serving in a single GPU job: the model is downloaded and exported
on the remote worker, the GPU is handed over to vLLM, and batched
completions are returned to your local machine. The exported checkpoint is
also archived to `KINETIC_OUTPUT_DIR`, so it can be served again anywhere
vLLM runs.

The example uses `gemma3_4b` (~8 GB in bfloat16), which fits a single
NVIDIA L4. `gpt2_large_en` is an ungated alternative if you don't have
Kaggle access to Gemma (use `dtype="float32"` with it).

## Prerequisites

1.  **Kinetic Cluster**: You need a provisioned Kinetic cluster with GPU
    nodes (e.g., `l4`), created with the latest `keras-kinetic`:

    ```bash
    kinetic pool add --accelerator l4 --project your-project-id
    ```

    GPU quota is the most common first-time blocker: in *IAM & Admin →
    Quotas*, both **GPUs (all regions)** and **NVIDIA L4 GPUs** (regional)
    must be ≥ 1.
2.  **Kaggle Credentials**: If you are using gated models like Gemma 3,
    you need a Kaggle account with the
    [model license accepted](https://www.kaggle.com/models/keras/gemma3),
    and `KAGGLE_USERNAME` / `KAGGLE_KEY` set in your local environment.

## Configuration

To run vLLM successfully on GPU via Kinetic, you need to handle
dependencies and environment variables properly.

### 1. Dependencies

Create a `requirements.txt` file in the directory of your script
containing:

```text
keras
keras-hub
tensorflow-text
vllm
```

Kinetic will detect this file and build a container with vLLM installed.
`tensorflow-text` is required —
KerasHub tokenizers preprocess with `tf.data` on every backend (CPU-side
only). Use a **Python 3.12** local venv; the remote container matches your
local interpreter, and `tensorflow-text` doesn't publish wheels for the
newest Python yet.

### 2. Environment Variables

The example sets the following environment variables on the remote worker
to ensure correct execution:

-   `KERAS_BACKEND="torch"`: vLLM is PyTorch-based, and the torch backend
    is the only one that releases VRAM cleanly after the export, so vLLM's
    KV cache gets the full GPU.
-   `VLLM_USE_FLASHINFER_SAMPLER="0"`: vLLM's FlashInfer sampler
    JIT-compiles CUDA kernels with `nvcc`, which pip-only containers don't
    have; this selects the native torch sampler instead.
-   `LD_LIBRARY_PATH=/usr/local/nvidia/lib64:...`: GKE mounts the host
    NVIDIA driver at `/usr/local/nvidia`; this makes `libcuda` /
    `libnvidia-ml` visible to vLLM's spawned engine processes (the example
    also preloads them into the main process via `ctypes`).

These are set inside the remote function by the example itself. Kaggle
credentials are forwarded from your local environment via
`capture_env_vars` in the `@kinetic.run` decorator.

## Example

```{literalinclude} ../../examples/vllm_serving.py
:language: python
```

## Running the Example

```bash
python3 vllm_serving.py
```

The first run builds the container image (15–25 minutes; subsequent runs
reuse it as a cache hit) and provisions a GPU node from the scale-to-zero
pool (~10 minutes including the image pull). Monitor from a second
terminal with `kinetic jobs list` and
`kinetic jobs logs --follow JOB_ID --project your-project-id`.

## Serving on TPU

The export step is accelerator-agnostic, but vLLM serving itself always
needs a GPU or TPU. TPU serving uses a different vLLM build (`vllm-tpu`)
and different environment variables than the GPU example above, and
Kinetic builds one container per script directory — `vllm` and `vllm-tpu`
cannot share an image. So on TPU, export and serving run as **two scripts
in two directories, each with its own `requirements.txt`**. The checkpoint
moves between them through GCS; nothing is downloaded to your local
machine, and no re-initialization of Kinetic is needed.

1. **Export script** (directory A, `requirements.txt`: `keras`,
   `keras-hub`, `tensorflow-text`) — the export half of this example, on
   any accelerator (`"cpu"` works). It archives the checkpoint to
   `KINETIC_OUTPUT_DIR` (a GCS path), which it returns; pass that path to
   the serving script.
2. **Serving script** (directory B, `requirements.txt`: `vllm-tpu`) —
   configured per
   [Running vLLM on TPU with Kinetic](../guides/vllm_tpu.md): set
   `VLLM_TARGET_DEVICE="tpu"`, `VLLM_USE_V1="0"`, and
   `JAX_PLATFORMS="tpu,cpu"` locally and forward them with
   `capture_env_vars`. The job downloads the archive from GCS to the
   pod's local disk, unpacks it, and points `LLM(model=...)` at that
   directory instead of a Hub model ID:

```python
@kinetic.run(
    accelerator="tpu-v5litepod-8",
    capture_env_vars=["VLLM_*", "JAX_*"],
)
def serve_on_tpu(checkpoint_gs_path, prompts):
    # Download + unpack the exported checkpoint from GCS to /tmp/hf_export
    # (google.cloud.storage download, tarfile.extractall), then:
    from vllm import LLM, SamplingParams

    llm = LLM(model="/tmp/hf_export", max_model_len=1024)
    return llm.generate(prompts, SamplingParams(max_tokens=128))
```
