# Interactive Debugging

Pass `debug=True` to `@kinetic.run()` or `@kinetic.submit()` to attach
a VS Code debugger to the remote pod. Set breakpoints, step through
your function, inspect variables, and evaluate expressions against the
accelerator your code is running on.

Kinetic prints a ready-to-paste `launch.json` entry — standard
`debugpy` attach config — so every VS Code-derived editor picks it up
as-is: **VS Code**, **Cursor**, **Windsurf**, **Antigravity**,
**VSCodium**, and anything else that ships the Python / debugpy
extension.

## A first debug session

Add `debug=True` to any `@kinetic.run()` or `@kinetic.submit()`
invocation:

```python
import kinetic

@kinetic.run(accelerator="tpu-v5e-2x2", debug=True)
def train():
    import jax
    breakpoint()  # debugger will pause here
    x = jax.numpy.arange(16)
    return x.sum()

train()
```

When you call `train()`, Kinetic:

1. Schedules the pod with debugging enabled and an extended **2-hour**
   TTL (vs 10 minutes for normal jobs) so the session has time to
   breathe.
2. Pauses execution just before your function runs and waits for a
   debugger to attach.
3. Prints a VS Code `launch.json` snippet to your terminal — paste it
   into `.vscode/launch.json`.
4. Press **F5** (Run → Start Debugging) in your editor. The debugger
   attaches and pauses inside Kinetic's runner. Press **F11** to step
   into your function, or **F10** to run straight through to your own
   `breakpoint()`.

When your function returns, the debugger connection is torn down
automatically and the pod cleans up.

:::{tip}
You don't need to call `breakpoint()` explicitly. Set breakpoints in
your editor's UI like you would locally — Kinetic pauses before your
function runs, and UI breakpoints work from there.
:::

## Attaching to a submitted job

For longer sessions, use `@kinetic.submit(debug=True)` and attach later
from the CLI:

```python
@kinetic.submit(accelerator="tpu-v5e-2x2", debug=True)
def train():
    import jax
    breakpoint()
    ...

job = train()
print(job.job_id)
```

Then from a terminal (same machine or a different one with access to
the same GCP project):

```bash
kinetic jobs debug <job_id>
```

`kinetic jobs debug` blocks until the job finishes or you hit Ctrl+C,
then tears down the connection. The command fails fast if the job
wasn't submitted with `debug=True`.

You can also drive it from Python:

```python
import kinetic
from kinetic.debug import cleanup_port_forward

job = kinetic.attach("<job_id>")
pf = job.debug_attach(local_port=5678)
try:
    job.result()  # or job.status() in a loop
finally:
    cleanup_port_forward(pf)
```

## Port conflicts

The default port is `5678` — debugpy's default, which VS Code's Python
extension auto-fills in `launch.json`. If something else is already
bound to `5678` locally, Kinetic raises a `RuntimeError` pointing you
at a different port:

```bash
kinetic jobs debug <job_id> --port 5679
```

Or from Python:

```python
pf = job.debug_attach(local_port=5679)
```

Remember to update the `port` field in your `launch.json` snippet to
match.

## Path mappings and source files

Kinetic fills in `pathMappings` in the printed `launch.json` so
breakpoints set in your local files hit the matching remote files —
no "unverified breakpoint" warnings, no file mismatch.

If you attach from a directory that isn't your project root, pass
`working_dir=` to `debug_attach()` (or replace `${workspaceFolder}` in
the printed snippet) so the mapping points at the sources you actually
have open.

## Timeouts and the attach window

The pod waits up to 10 minutes for a debugger client to attach. If no
one connects in that window, it proceeds with your function running
normally — the job does not hang indefinitely. To extend or shorten
that window, set `KINETIC_DEBUG_WAIT_TIMEOUT` (seconds) in your local
environment before submitting:

```bash
export KINETIC_DEBUG_WAIT_TIMEOUT=1800  # 30 minutes
```

## Multi-host debugging

On multi-host TPU slices (Pathways backend), you attach once to the
leader pod; Kinetic sequences the non-leader workers so the
distributed runtime doesn't start until you're ready. `jax.process_index()`
semantics stay predictable, and you don't need to attach to each host
separately.

:::{warning}
**Avoid `spot=True` with `debug=True`.** Preemption mid-session
terminates the pod, dropping your debug connection. Kinetic warns at
decoration time if both are set. Use on-demand capacity for interactive
work.
:::

## Automated environments

`@kinetic.run(debug=True)` requires an interactive terminal — if
`stdin` isn't a TTY (CI, `nohup`, piped input), the local client
raises `RuntimeError` before submission so your job doesn't silently
hang waiting for someone to attach.

For async submission there's no TTY requirement —
`@kinetic.submit(debug=True)` works fine in any environment, and
`kinetic jobs debug` from an interactive shell attaches whenever
you're ready.

## Related pages

- [Detached Jobs](async_jobs.md) — pairs with
  `kinetic jobs debug <job_id>` for long debug sessions.
- [Configuration](../configuration.md) — `KINETIC_DEBUG_WAIT_TIMEOUT`
  and the other user-facing environment variables.
- [Troubleshooting](../troubleshooting.md) — what to check when a pod
  doesn't reach `RUNNING` or the debugger fails to attach.
