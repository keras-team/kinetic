# How to contribute

We'd love to accept your patches and contributions to this project.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our community guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Contribution process

### Development setup

1. Install the package with development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

2. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

### Code quality and testing

Before submitting a pull request, please ensure your changes pass linting and unit tests.

- **Linting:** We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Run it with:
  ```bash
  ruff check .
  ```
- **Unit tests:** We use [Pytest](https://docs.pytest.org/) for unit tests. Run them with:

  ```bash
  pytest
  ```

- **E2E tests:** End-to-end tests run real workloads against a GKE cluster. They live in `tests/e2e/` and are skipped by default unless explicitly enabled.

  **Prerequisites:**
  - A GCP project with a provisioned GKE cluster (see [Quick Start](../README.md#quick-start))
  - Google Cloud SDK authenticated (`gcloud auth login` and `gcloud auth application-default login`)
  - GKE credentials configured: `gcloud container clusters get-credentials <KERAS_REMOTE_CLUSTER> --zone <KERAS_REMOTE_ZONE> --project <KERAS_REMOTE_PROJECT>`
  - Test dependencies installed: `pip install -e ".[test,cli]"`

  **Required environment variables:**

  | Variable               | Required | Default         | Description                    |
  | ---------------------- | -------- | --------------- | ------------------------------ |
  | `E2E_TESTS`            | Yes      | —               | Set to `1` to enable e2e tests |
  | `KERAS_REMOTE_PROJECT` | Yes      | —               | Google Cloud project ID        |
  | `KERAS_REMOTE_ZONE`    | No       | `us-central1-a` | GKE cluster zone               |
  | `KERAS_REMOTE_CLUSTER` | No       | `kinetic`  | GKE cluster name               |

  **Run all e2e tests:**

  ```bash
  E2E_TESTS=1 KERAS_REMOTE_PROJECT=my-project python -m pytest tests/e2e/ -v -n auto
  ```

  **Run a specific test file:**

  ```bash
  E2E_TESTS=1 KERAS_REMOTE_PROJECT=my-project python -m pytest tests/e2e/cpu_execution_test.py -v
  ```

  Drop `-n auto` to run tests serially to make it easier to debug.

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
