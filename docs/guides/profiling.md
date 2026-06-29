# Performance Profiling (XProf)

A job that *runs* on a TPU or GPU isn't necessarily using it well.
Profiling shows where the accelerator time actually goes. This guide
covers **XProf** — the profiler for XLA workloads — and how to capture a
trace from a Kinetic job and view it on your own machine.

Because a Kinetic pod scales to zero the moment your job finishes, the
workflow is: **capture inside the function, write the trace to
`KINETIC_OUTPUT_DIR` (durable GCS), then view it locally.**

## What XProf is

XProf is the open-source accelerator profiler from
[OpenXLA](https://openxla.org/xprof) (formerly the TensorBoard "profile"
plugin). It runs standalone or as a TensorBoard tab, reads hardware
counters rather than wall-clock timers, and adds little overhead during
the capture window.

It is an **XLA** profiler, and that decides which tool you capture with:

| Backend / workload | Capture with | View in |
| --- | --- | --- |
| Keras-on-JAX (Kinetic default), native JAX | `jax.profiler` | XProf |
| Keras-on-TensorFlow | `TensorBoard(profile_batch=…)` | XProf |
| PyTorch/XLA (`torch_xla`) | `torch_xla.debug.profiler` | XProf |
| Native PyTorch (eager CUDA) | `torch.profiler` | Perfetto |

Native eager PyTorch doesn't compile through XLA, so it uses
`torch.profiler` and views in Perfetto — not XProf.

## Capture a profile

Capture a handful of steps **after a warm-up**, inside the decorated
function, and write to `$KINETIC_OUTPUT_DIR/profile` so the trace
survives the pod (see [Checkpointing](checkpointing.md)). The job needs
**no extra packages** — `jax.profiler` ships with JAX, and `xprof` is a
*local viewer*, not a job dependency. If a profiling job does need an
extra package, add it to a `requirements.txt` in that script's own
directory (see [Dependencies](dependencies.md)).

This is a deliberately minimal demo — a tiny JAX training loop — but the
same capture pattern drops into any job (real training, KerasHub
fine-tuning, vLLM serving, multi-host runs):

```{literalinclude} ../../examples/jax_profiling_demo.py
:language: python
:caption: examples/jax_profiling_demo.py
```

For **Keras-on-JAX**, it's the same idea: wrap a short `model.fit(...)`
in `with jax.profiler.trace(trace_dir):` after a warm-up epoch.

:::{note}
JAX dispatches asynchronously, so always `block_until_ready()` inside the
trace region — otherwise the profiler can close before the device work
lands. Keep the window to a few steps; traces grow fast.
:::

Other backends: **native PyTorch** uses `torch.profiler` with
`tensorboard_trace_handler(trace_dir)` (view in Perfetto);
**PyTorch/XLA** uses `torch_xla.debug.profiler`, which produces
XProf-readable traces.

## View the trace

Install the viewer and point it at the trace path your job printed:

```bash
pip install xprof gcsfs        # gcsfs lets XProf read gs:// directly
xprof --logdir gs://<project>-kn-<cluster>-jobs/outputs/<job_id>/profile --port 6006
# --logdir is the directory that contains plugins/ (the path your job printed)
# then open http://localhost:6006
```

Or copy it down first — `gcloud storage cp -r <gs-path> ./trace` and
`--logdir ./trace`. In the UI, use the tool dropdown: **Trace Viewer**
for the step timeline, **Overview Page** for the summary. (The
**Capture Profile** button does live, on-demand capture against a
running profiler server, so it isn't used here — the trace was already
captured inside the job.)

## What the tools show

- **Overview Page** — top-level summary; whether you're host- or
  device-bound. Start here.
- **Trace Viewer** — per-event timeline across host / TPU / GPU; where
  you spot gaps and stalls.
- **Roofline** — memory-bound vs. compute-bound, which decides your
  optimization strategy.
- **Framework / HLO Op Stats** — cost by framework op and by compiled
  HLO op.
- **Memory Viewer / Profile** — usage over time and at peak; first stop
  after an OOM.
- **Megascale Stats** — cross-slice (DCN) communication on multi-host
  [Pathways](distributed_training.md) runs.

## Related pages

- [Cost Optimization](cost_optimization.md) — a profile shows *where* the
  accelerator-hours go; that guide shows how to cut the bill.
- [Checkpointing and Outputs](checkpointing.md) — how `KINETIC_OUTPUT_DIR`
  keeps your trace after the pod exits.
- [Dependencies](dependencies.md) — per-job `requirements.txt`.
- [Distributed Training](distributed_training.md) — multi-host runs where
  Megascale Stats applies.
