# Checkpointing and Auto-Resume

This guide demonstrates how to use Orbax for checkpointing in Kinetic workloads. Kinetic automatically sets up an output directory and propagates it via the `KINETIC_OUTPUT_DIR` environment variable, making it easy to save and restore state without hardcoding GCS paths or cluster-specific details.

> **Important**: By default, Kinetic imposes a 30-day TTL (Time to Live) on the GCS buckets it creates. This means anything written to the default `KINETIC_OUTPUT_DIR` will be automatically deleted after 30 days. If you need to preserve checkpoints longer, you should copy them to a bucket without a lifecycle rule or specify a custom `output_dir`.


## Example

Here is a complete example showing Orbax checkpointing with Kinetic and Auto-Resume. You can find this file at [`examples/example_checkpoint.py`](https://github.com/keras-team/kinetic/blob/main/examples/example_checkpoint.py) in the repository.

```{literalinclude} ../../examples/example_checkpoint.py
```
