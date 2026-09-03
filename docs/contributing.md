# Contributing

We welcome your patches and contributions to this project. This page
explains the legal requirements, the development setup, and the review
process.

## Before you begin

### Sign our Contributor License Agreement

Every contribution to this project needs a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You, or your employer, keep the copyright to your contribution. The CLA
gives us the permission to use and to redistribute your contribution as
part of the project.

:::{note}
If you or your current employer already signed the Google CLA, for this
project or for a different one, you do not need to sign it again.
:::

Visit <https://cla.developers.google.com/> to see your current agreements
or to sign a new one.

### Review our community guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Contribution process

### Development setup

1. Install the package with development dependencies:

   ```bash
   uv pip install -e ".[dev]"
   ```

2. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

### Code quality

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Run it before submitting a pull request:

```bash
ruff check . && ruff format --check .
```

### Testing

Kinetic's tests are organised as a ladder of tiers. Each tier is more
realistic than the one below it and needs one more thing installed. Every
tier **skips itself cleanly** when its prerequisite is missing, so a fresh
checkout always gets a green run — just a less thorough one. Install
what you can; CI runs tiers 0 and 1 on every pull request.

| Tier | What it exercises | Needs | Runtime |
| ---- | ----------------- | ----- | ------- |
| Unit | Pure logic, orchestration above mocked seams | nothing | seconds |
| 0 — emulator | Real GCS wire protocol via [fake-gcs-server](https://github.com/fsouza/fake-gcs-server) | the `fake-gcs-server` binary | seconds |
| 1 — docker | The real runner image, run with the exact command the Job spec generates | Tier 0 + a Docker daemon | ~2 min cold, ~15 s warm |
| e2e | Real workloads on a real GKE cluster | a GCP project (see below) | minutes |

Install test dependencies first:

```bash
uv pip install -e ".[test]"
```

#### Tier 0: the GCS emulator

`fake-gcs-server` is the **canonical transport for every test that touches
Cloud Storage** — there are no hand-written GCS mocks. It is a single
static Go binary, not a Python package, so `pip` will not fetch it:

```bash
brew install fake-gcs-server
```

Alternatives: `go install github.com/fsouza/fake-gcs-server@latest`, or
download a [release tarball](https://github.com/fsouza/fake-gcs-server/releases)
and point `FAKE_GCS_SERVER_BIN` at the extracted binary.

The test fixture (`kinetic/utils/fake_gcs_fixture.py`) finds the binary on
your `PATH`, starts it on a free port, and stops it at exit — nothing to
configure. Then run the unit and integration suites:

```bash
python -m unittest discover -s kinetic -p "*_test.py"
```

```bash
python -m unittest discover -s tests/integration -t . -p "*_test.py"
```

Without the binary, emulator-backed tests skip with a message pointing
here; everything else still runs.

:::{note}
The fixture exports `STORAGE_EMULATOR_HOST` for the whole process, so it
refuses to start when `E2E_TESTS` is set. Run the e2e suite in a
separate process, as CI does.
:::

#### Tier 1: the docker roundtrip

This tier builds the actual runner image through kinetic's own build
machinery and executes it with `docker run`, using the command line
derived from the real Job spec, against the emulator. It needs Docker
Desktop (or any Docker daemon) running:

```bash
python -m unittest discover -s tests/docker -t . -p "*_test.py"
```

The first run builds the image (a base pull plus the JAX/Keras/kinetic
install); later runs reuse it by content hash until `remote_runner.py`
or the Dockerfile template changes.

:::{note}
The image installs the *released* `keras-kinetic` version from PyPI —
exactly what Cloud Build does — so a `version.py` bump past the latest
release fails this tier's build until that version is published.
:::

#### E2E tests

End-to-end tests run real workloads against a GKE cluster. They live in
`tests/e2e/` and are skipped unless explicitly enabled.

**Prerequisites:**
- A GCP project with a provisioned Kinetic cluster and an active
  profile — i.e. you have run `kinetic init` (which provisions or joins a
  cluster and saves the profile).
- Google Cloud SDK authenticated (`gcloud auth login` and `gcloud auth application-default login`).
  Kinetic fetches the cluster's kubeconfig itself on first use.

The tests submit jobs through `@kinetic.run`, so they resolve project,
zone, and cluster exactly the way user code does: explicit argument,
then `KINETIC_*` environment variable, then the active profile, then
the built-in default. With a profile set, the only variable you need is
`E2E_TESTS`:

```bash
E2E_TESTS=1 python -m pytest tests/e2e/ -v -n auto
```

**Run a specific test file:**

```bash
E2E_TESTS=1 python -m pytest tests/e2e/cpu_execution_test.py -v
```

**Optional overrides** — useful for pointing the suite at a cluster other
than your active profile's (this is how CI runs it, with no profile on
the runner):

| Variable          | Overrides           | Default without a profile |
| ----------------- | ------------------- | ------------------------- |
| `KINETIC_PROJECT` | profile project     | `GOOGLE_CLOUD_PROJECT`, else required |
| `KINETIC_ZONE`    | profile zone        | `us-central1-a`           |
| `KINETIC_CLUSTER` | profile cluster     | `kinetic-cluster`         |

:::{tip}
Remove `-n auto` to run the tests one at a time. Serial runs are easier to debug.
:::

#### Writing new tests

- Anything that touches Cloud Storage should use the emulator fixture
  (`FakeGcsTestCase` or `shared_server()`), seed real blobs, and assert on
  emulator state — never on log text.
- Patching is fine for **fault injection** (simulating an outage or a
  `Forbidden`) and for **spies** that record calls while the real code
  runs. Do not patch to replace transport.
- Tests above the storage seam (job polling, cleanup routing, batch
  fan-out) may mock kinetic's own `storage` functions as collaborators;
  the seam itself is covered for real by `tests/integration/`.

### Submitting changes

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Commit your changes: `git commit -m 'Add my feature'`.
4. Push the branch: `git push origin feature/my-feature`.
5. Open a pull request.

### Code reviews

Every submission needs a review, including a submission from a project
member. We use GitHub pull requests for reviews. See
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for
more information about pull requests.

## Documentation Contribution Process

Install the documentation dependencies, then build and serve the site
locally:

```bash
uv pip install -e ".[docs]"
sphinx-autobuild docs /tmp/docs
```

The pages are MyST Markdown under `docs/`. Follow the style of the
existing pages: short sentences in the active voice, present tense,
imperative steps, and no contractions.

## Releases

Only maintainers release Kinetic. The process is in
[RELEASE_PROCESS.md](https://github.com/keras-team/kinetic/blob/main/RELEASE_PROCESS.md)
in the repository. In short:

:::{container} kinetic-steps
1. **Bump the version** in `pyproject.toml` and `kinetic/version.py`
   through a pull request.
2. **Create a release branch** named after the version, for example
   `r0.0.5`.
3. **Create a GitHub release** from that branch at
   <https://github.com/keras-team/kinetic/releases/new>.
4. **Wait for the publish workflow.** The release tag starts the
   `publish_to_pypi` GitHub Actions workflow, which uploads the package
   to PyPI. Do not upload with `twine` yourself.
:::
