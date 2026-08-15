# Contributing

We'd love to accept your patches and contributions to this project.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

:::{note}
If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.
:::

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

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
- A GCP project with a provisioned GKE cluster.
- Google Cloud SDK authenticated (`gcloud auth login` and `gcloud auth application-default login`)
- GKE credentials configured: `gcloud container clusters get-credentials <KINETIC_CLUSTER> --zone <KINETIC_ZONE> --project <KINETIC_PROJECT>`

**Required environment variables:**

| Variable          | Required | Default         | Description                    |
| ----------------- | -------- | --------------- | ------------------------------ |
| `E2E_TESTS`       | Yes      | —               | Set to `1` to enable e2e tests |
| `KINETIC_PROJECT` | Yes      | —               | Google Cloud project ID        |
| `KINETIC_ZONE`    | No       | `us-central1-a` | GKE cluster zone               |
| `KINETIC_CLUSTER` | No       | `kinetic-cluster` | GKE cluster name             |

**Run all e2e tests:**

```bash
E2E_TESTS=1 KINETIC_PROJECT=my-project python -m pytest tests/e2e/ -v -n auto
```

**Run a specific test file:**

```bash
E2E_TESTS=1 KINETIC_PROJECT=my-project python -m pytest tests/e2e/cpu_execution_test.py -v
```

:::{tip}
Drop `-n auto` to run tests serially to make it easier to debug.
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

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Documentation Contribution Process

```sh
# Install docs libraries:
pip install .[docs]

# Build and serve docs locally:
sphinx-autobuild docs /tmp/docs
```

## Releases

To release a new version of the package to PyPI, follow these steps:

:::{container} kinetic-steps
1. **Install the release dependencies.**
   ```bash
   uv pip install -e ".[release]"
   ```
2. **Bump the version** in the following files:
   - [pyproject.toml](../pyproject.toml)
   - [version.py](../kinetic/version.py)
3. **Build the source distribution and wheel.**
   ```bash
   python3 -m build
   ```
4. **Upload the packages to PyPI** using `twine`. To avoid `twine` hanging while waiting for interactive input, provide your credentials via environment variables (e.g. using an API token) or a `~/.pypirc` file:
   ```bash
   TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-... python3 -m twine upload dist/*
   ```
5. **Create a new release on GitHub** using the `gh` CLI, e.g.:
   ```bash
   gh release create 0.0.2
   ```
:::

