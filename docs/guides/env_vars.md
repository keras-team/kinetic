# Forward Environment Variables

Kinetic allows you to propagate local environment variables to the remote worker environment. This is useful for passing API keys, configuration, or credentials without hardcoding them in your script.

## Forwarding Variables

Use the `capture_env_vars` parameter in the `@kinetic.run()` decorator. It accepts a list of environment variable names or wildcard patterns.

```python
import kinetic

@kinetic.run(
    accelerator="v5litepod-1",
    capture_env_vars=["KAGGLE_USERNAME", "KAGGLE_KEY", "WANDB_*"]
)
def train_model():
    import os
    # These are available in the remote process
    user = os.environ.get("KAGGLE_USERNAME")
    # ...
```

## Wildcard Support

You can use the `*` suffix to capture all environment variables that start with a specific prefix.

- `capture_env_vars=["GOOGLE_CLOUD_*"]`: Captures `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_REGION`, etc.
- `capture_env_vars=["*"]`: (Not recommended) Captures the entire local environment.

## Secure Handling

Kinetic serializes the values of the requested environment variables and sends them to the remote worker as part of the job payload. Ensure you only forward variables that are necessary for the job.

## Precedence

Environment variables set via `capture_env_vars` will override any existing variables with the same name in the remote container's base environment.

## Canonical Environment Variables

Kinetic automatically sets some environment variables in the remote worker environment:

- `KINETIC_OUTPUT_DIR`: The path to the directory where outputs should be saved. By default, this is a GCS path pointing to `gs://{bucket_name}/outputs/{job_id}`. This is useful for passing to checkpointing libraries like Orbax.

> **Important**: By default, Kinetic imposes a 30-day TTL (Time to Live) on the
> GCS buckets it creates. This means anything written to the default
> `KINETIC_OUTPUT_DIR` will be automatically deleted after 30 days. If you need
> to preserve outputs longer, you should copy them to a bucket without a
> lifecycle rule or specify a custom `output_dir` pointing to a different
> location.

