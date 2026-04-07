# Configuration

Kinetic uses environment variables to manage default settings across the CLI and the Python library.

## Local Environment Variables

| Variable                  | Required | Default            | Description                                                  |
| ------------------------- | -------- | ------------------ | ------------------------------------------------------------ |
| `KINETIC_PROJECT`         | Yes      | —                  | Google Cloud project ID                                      |
| `KINETIC_ZONE`            | No       | `us-central1-a`    | Default compute zone                                         |
| `KINETIC_CLUSTER`         | No       | `kinetic-cluster`  | GKE cluster name                                             |
| `KINETIC_BASE_IMAGE_REPO` | No       | `kinetic`          | Docker repository for prebuilt base images                   |
| `KINETIC_NAMESPACE`       | No       | `default`          | Kubernetes namespace                                         |
| `KINETIC_LOG_LEVEL`       | No       | `INFO`             | Log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `FATAL`) |

You can set these variables in your shell profile (e.g., `~/.bashrc` or `~/.zshrc`) to persist them across sessions.

```bash
export KINETIC_PROJECT="my-gcp-project-id"
export KINETIC_ZONE="us-central1-a"
```

## Precedence

1.  **Decorator Parameters**: Values passed directly to `@kinetic.run()` or `@kinetic.submit()` have the highest precedence.
2.  **Environment Variables**: If a parameter is not provided, Kinetic looks for the corresponding `KINETIC_*` environment variable.
3.  **Defaults**: If neither is present, Kinetic uses its built-in default values.

## Logging

Kinetic uses `absl-py` for logging. You can control the verbosity by setting `KINETIC_LOG_LEVEL`.

- **DEBUG**: Shows detailed information about container builds, artifact uploads, and GKE job submission.
- **INFO**: Shows major milestones in the job lifecycle.
- **WARNING/ERROR**: Shows only critical issues.
