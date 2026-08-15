# Interactive Debugging

Pass `debug=True` to `@kinetic.run()` to attach
a VS Code debugger to the remote pod. Set breakpoints, step through
your function, inspect variables, and evaluate expressions against the
accelerator your code is running on.

Kinetic prints a ready-to-paste `launch.json` entry — standard
`debugpy` attach config — so every VS Code-derived editor picks it up
as-is: **VS Code**, **Cursor**, **Windsurf**, **Antigravity**,
**VSCodium**, and anything else that ships the Python / debugpy
extension.

## A first debug session

Add `debug=True` to `@kinetic.run()`:

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

:::{container} kinetic-steps
1. **Schedules the pod** with debugging enabled and an extended **2-hour**
   TTL (vs 10 minutes for normal jobs) so the session has time to
   breathe.
2. **Pauses execution** just before your function runs and waits for a
   debugger to attach.
3. **Prints a VS Code `launch.json` snippet** to your terminal — paste it
   into `.vscode/launch.json`.
4. **Press F5** (Run → Start Debugging) in your editor. The debugger
   attaches and pauses inside Kinetic's runner. Press **F11** to step
   into your function, or **F10** to run straight through to your own
   `breakpoint()`.
:::

When your function returns, the debugger connection is torn down
automatically and the pod cleans up.

:::{tip}
You don't need to call `breakpoint()` explicitly. Set breakpoints in
your editor's UI like you would locally — Kinetic pauses before your
function runs, and UI breakpoints work from there.
:::

## Attaching to a submitted job

For longer sessions, use `@kinetic.run(debug=True)` and attach later
from the CLI:

```python
@kinetic.run(accelerator="tpu-v5e-2x2", debug=True)
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

The pod uses the same source paths as your machine. The runner
extracts the workspace into a temporary directory, then makes a
symlink to it at the directory you submitted from. Your files thus
have one path on both sides.

Because the paths agree, the printed `launch.json` uses an identity
`pathMappings` entry: `localRoot` and `remoteRoot` are the same
directory. Breakpoints in your local files hit the matching remote
files, with no "unverified breakpoint" warnings.

`kinetic jobs debug <job_id>` does not know which directory you
submitted from, so it prints no `pathMappings` entry. Unmapped paths
are what debugpy assumes, which is correct here.

Add or edit the mapping only if you open the sources from a different
directory than the one you submitted from. Set `localRoot` to the
directory you have open, and `remoteRoot` to the directory you
submitted from.

## Timeouts and the attach window

The pod waits up to 10 minutes for a debugger client to attach. If no
one connects in that window, it proceeds with your function running
normally — the job does not hang indefinitely. To extend or shorten
that window, set `KINETIC_DEBUG_WAIT_TIMEOUT` (seconds) in your local
environment before submitting:

```bash
export KINETIC_DEBUG_WAIT_TIMEOUT=1800  # 30 minutes
```

Kinetic reads the variable when you submit, and puts the value into
the pod. The client and the pod thus wait for the same time. Set the
variable before you submit: a change after that has no effect on a job
that is already on the cluster.

The value must be a positive whole number of seconds. Kinetic ignores
a different value, logs a warning, and uses 10 minutes.

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
`@kinetic.run(debug=True)` works fine in any environment, and
`kinetic jobs debug` from an interactive shell attaches whenever
you're ready.

## Related pages

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`history;1em` Detached Jobs
:link: async_jobs
:link-type: doc

Pairs with `kinetic jobs debug <job_id>` for long debug sessions.
:::

:::{grid-item-card} {octicon}`gear;1em` Configuration
:link: ../configuration
:link-type: doc

`KINETIC_DEBUG_WAIT_TIMEOUT` and the other user-facing environment
variables.
:::

:::{grid-item-card} {octicon}`bug;1em` Troubleshooting
:link: ../troubleshooting
:link-type: doc

What to check when a pod doesn't reach `RUNNING` or the debugger fails
to attach.
:::
::::
