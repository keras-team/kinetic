# Managing Dependencies

There are three independent things going on when Kinetic runs your job:

1. **Dependency discovery** — Kinetic figures out which packages your
   project needs by reading `requirements.txt` or `pyproject.toml` from
   your working directory.
2. **Container mode choice** — those dependencies either get baked into
   a custom image (bundled mode), installed at pod startup (prebuilt
   mode), or ignored entirely (custom image mode). See
   [Execution Modes](execution_modes.md).
3. **JAX filtering** — accelerator runtime packages (`jax`, `jaxlib`,
   `libtpu`) are filtered out before install so they don't shadow the
   hardware-correct versions in the container.

This page focuses on (1) and (3). (2) lives on its own page:
[Execution Modes](execution_modes.md).

## A first run

Drop a `requirements.txt` next to your script and Kinetic picks it up
automatically:

```text
# requirements.txt
keras
numpy
pandas
```

```python
@kinetic.run(accelerator="tpu-v6e-8")
def train():
    import pandas as pd  # installed automatically on the remote
    ...
```

`pyproject.toml` works equally well — Kinetic reads
`[project.dependencies]`. If both files exist, `requirements.txt` wins.

:::{tip}
**Recommended defaults:**

- Pin only the libraries you actually depend on. The fewer packages, the
  faster your image builds (or your prebuilt-mode pod start).
- Don't pin `jax`, `jaxlib`, `libtpu`, or any other accelerator runtime
  — Kinetic filters them out and uses the version in the container.
- Use a `pyproject.toml` if you already have one for local development
  rather than maintaining a separate `requirements.txt`.
:::

## How discovery works

When you call a decorated function, Kinetic looks in your working
directory for a dependency file. The lookup is straightforward:

1. If `requirements.txt` exists, use it.
2. Otherwise, if `pyproject.toml` exists, extract `[project.dependencies]`.
3. Otherwise, no dependency file is registered and the container ships
   with only the base image's packages.

In bundled mode, the discovered file is hashed and used as part of the
image cache key — change the file, and the next run rebuilds. In
prebuilt mode, the same file is uploaded and installed at pod startup.
In custom image mode, the file is ignored entirely.

## JAX and accelerator runtimes

Kinetic's bundled and prebuilt images already have `jax`, `jaxlib`, and
the right accelerator backend (`libtpu` on TPU, CUDA libs on GPU)
installed and pinned to versions that match the container. To prevent
your `requirements.txt` from clobbering that, Kinetic strips these
entries before install:

- `jax`
- `jaxlib`
- `libtpu`
- `libtpu-nightly`

If you have a specific reason to override the in-container JAX —
testing a new release, reproducing a bug — append `# kn:keep` to the
line:

```text
jax==0.4.25 # kn:keep
jaxlib==0.4.25 # kn:keep
```

This works in `requirements.txt`. Use it sparingly; getting JAX +
`jaxlib` + accelerator runtime versions to line up by hand is a known
source of obscure crashes.

## Private packages

Bundled-mode builds install your dependencies inside Cloud Build. Cloud
Build does not inherit your local `pip.conf`, environment variables, or
shell credentials, so anything the installer needs in order to find or
authenticate to a private index has to be present in the project source
that gets uploaded to the build.

You have two practical options:

- **Bundled mode with the index URL inside `requirements.txt`.** Add
  `--index-url` or `--extra-index-url` as a line in `requirements.txt`.
  The installer reads these directives and uses them when resolving
  every package in the file:

  ```text
  --extra-index-url https://my-org-private-index.example.com/simple
  my-private-package==1.2.3
  some-public-dep==2.0.0
  ```

  This works without extra setup if the index is publicly reachable
  (no auth required), or if it sits behind network ACLs that the Cloud
  Build pool already satisfies (for example, a GCP-internal Artifact
  Registry repo that the build service account has read access to).
- **Custom image mode.** If your private packages need credentials at
  install time, system libraries, or unusual build flags, prebuild a
  container image with them installed and pass it as
  `container_image="<your-image-uri>"`. This gives you full control
  over the build environment, including `pip.conf`, secret mounts, and
  `gcloud` authentication. See [Container Images](../advanced/containers.md).

Avoid embedding secrets in `requirements.txt`
(`https://user:token@host/...`); the file is uploaded to GCS and used
as part of the build context, so any credentials it contains will end
up in build logs and cached artifacts.

## Common dependency pitfalls

- **Pinning `jax` without `# kn:keep`** — the pin is silently dropped
  and you get the in-container version anyway. If you actually want a
  pin, use `# kn:keep`. If you don't, drop the line.
- **Listing TensorFlow alongside JAX** — both ship their own copy of
  the accelerator runtime. They can co-exist, but on TPU you typically
  want only one. If `tf.data` is the only thing you need from
  TensorFlow, `tensorflow-cpu` is enough and won't fight with `libtpu`.
- **Forgetting to add a new package locally** — Kinetic only sees what's
  in `requirements.txt` or `pyproject.toml`. A `pip install` in your
  shell that isn't reflected in those files won't carry over.
- **Massive dependency sets** — every `requirements.txt` change forces
  a bundled rebuild. If your deps churn daily, consider prebuilt mode
  (after publishing a base image with `kinetic build-base`).
- **Editable installs (`pip install -e`)** — these don't show up in
  `requirements.txt` and won't carry over. Either ship the source via
  your working directory (already auto-packaged) or publish the package
  and pin a real version.

## Related pages

- [Execution Modes](execution_modes.md) — where the discovered deps go.
- [Container Images](../advanced/containers.md) — custom image and
  base-image workflows.
- [Troubleshooting](../troubleshooting.md) — what to check when an
  import fails on the remote.
