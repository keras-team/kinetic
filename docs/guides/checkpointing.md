# Checkpointing and Outputs

Long jobs need somewhere durable to write to. Pods come and go — when
your training script exits, anything that lived only inside the pod's
filesystem is gone. Kinetic gives you `KINETIC_OUTPUT_DIR`: a per-job
GCS prefix that survives the pod, so your checkpoints, logs, and final
artifacts are still there when you come back.

This page covers what to write where, how Orbax (or any other library)
plugs into it, and how cleanup and TTLs work.

## A first checkpointed job

Inside the pod, `KINETIC_OUTPUT_DIR` is already set. Read it and write
under it. Fall back to a local path when the variable is not present so
that the same function works when you exercise it locally:

```python
import os

import kinetic

@kinetic.run(accelerator="cpu")
def train():
    # Remote: KINETIC_OUTPUT_DIR resolves to gs://.../outputs/<job_id>.
    # Local: fall back to a filesystem path under /tmp so the same code
    # works when you run the function directly for testing.
    output_dir = os.environ.get("KINETIC_OUTPUT_DIR", "/tmp/local_checkpoints")
    # ... train and write checkpoints/artifacts under output_dir ...
    return f"saved to {output_dir}"
```

For full Orbax-managed auto-resume with JAX or Keras, the canonical
runnable examples live in the repo:

- [`examples/example_checkpoint.py`](https://github.com/keras-team/kinetic/blob/main/examples/example_checkpoint.py)
  — JAX + Orbax with auto-resume.
- [`examples/example_keras_checkpoint.py`](https://github.com/keras-team/kinetic/blob/main/examples/example_keras_checkpoint.py)
  — same pattern using `model.get_weights()` / `set_weights()`.

## Outputs and checkpoints

A Kinetic job produces three distinct kinds of artifact, each with its
own storage location and lifecycle:

Artifact              | What it is                             | Where it lives
--------------------- | -------------------------------------- | -------------------------------------------------------------------------------
Job return value      | The Python value your function returns | Persisted to `gs://{bucket}/{job_id}/result.pkl`, then downloaded to your local process
Durable outputs       | Files you wrote during the run         | `KINETIC_OUTPUT_DIR` (GCS)
Resumable checkpoints | Periodic state snapshots for restart   | `KINETIC_OUTPUT_DIR/<your-subdir>` (GCS)

The return value is the right channel for **small** results: a final
loss, a metric dict, a path string. Large files belong on the output
dir; checkpoints belong on a stable subpath under the output dir so
restarts can find them.

`KINETIC_OUTPUT_DIR` is set automatically when the job starts. By
default it resolves to the jobs bucket for your cluster:

```text
gs://{project}-kn-{cluster}-jobs/outputs/{job_id}
```

`{project}` is your GCP project (from `KINETIC_PROJECT`) and `{cluster}`
is the Kinetic cluster name (from `KINETIC_CLUSTER`, defaulting to
`kinetic-cluster`). The bucket is created by `kinetic up` and reused
across all jobs submitted to that cluster.

You can override it per job by passing `output_dir=` to the decorator,
setting `KINETIC_OUTPUT_DIR` in your local environment before
submission, or (when inspecting an existing job from the CLI) passing
`--output-dir` to the relevant `kinetic jobs` subcommand. See the
precedence table in [Configuration](../configuration.md) for how these
resolution paths combine.

## Recommended directory layout

A simple convention that scales from one job to many:

```text
$KINETIC_OUTPUT_DIR/
├── checkpoints/        # Orbax / model.save_weights — periodic snapshots
├── logs/               # extra logs your code writes (stdout already streams)
├── metrics/            # tensorboard / json metric dumps
└── final/              # post-training artifacts: exported model, eval results
```

Use whichever subdirectories make sense for your workflow. The point is
that the layout is yours to control — Kinetic only cares that you write
under the prefix it gave you.

## TTL and retention

By default the GCS bucket Kinetic creates has a **30-day TTL** on its
contents. Anything written to `KINETIC_OUTPUT_DIR` is auto-deleted
after 30 days. That's the right default for ephemeral training, but if
you want a checkpoint to outlive a month:

- Copy it to a bucket with no lifecycle policy (`gsutil cp` or the GCS
  client library).
- Or set `output_dir=` to a bucket you manage yourself, with whatever
  lifecycle rules you want.

`JobHandle.cleanup(gcs=True)` removes the per-job artifacts under the
GCS prefix used for code and result payloads — it does **not** touch
files you wrote under `KINETIC_OUTPUT_DIR`. Outputs survive cleanup.

## Copy-paste checklist

A short checklist for any long-running job that you don't want to redo
from scratch:

- [ ] Read `KINETIC_OUTPUT_DIR` inside the function and write everything
      durable under it.
- [ ] Write checkpoints to a stable subdirectory (e.g.
      `$KINETIC_OUTPUT_DIR/checkpoints/`) so the resume path is
      predictable.
- [ ] Choose a checkpoint cadence that bounds how much work a restart
      would lose (every N steps, or every M minutes).
- [ ] Verify resume works locally before the long run — submit the same
      function twice with the same `output_dir` and confirm the second
      call picks up where the first left off.
- [ ] If the run is critical, copy the final artifacts to a bucket
      without the 30-day TTL after success.

## JAX example

```{literalinclude} ../../examples/example_checkpoint.py
```

After the snippet:

- The function reads `KINETIC_OUTPUT_DIR` and points Orbax's
  `CheckpointManager` at it.
- Calling the function a second time picks up from the latest step
  rather than restarting from scratch.

## Keras example

```{literalinclude} ../../examples/example_keras_checkpoint.py
```

After the snippet:

- `model.get_weights()` produces a PyTree of NumPy arrays that Orbax
  knows how to save.
- `model.set_weights()` restores them on resume.

## Related pages

- [Data](data.md) — input side of the I/O story.
- [Managing Async Jobs](async_jobs.md) — long jobs are also
  the place where you most want detached submission.
- [Cost Optimization](cost_optimization.md) — spot instances make
  checkpointing essential.
