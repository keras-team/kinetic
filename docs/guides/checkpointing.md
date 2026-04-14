# Checkpointing and Auto-Resume

This guide demonstrates how to use Orbax for checkpointing in Kinetic workloads. Kinetic automatically sets up an output directory and propagates it via the `KINETIC_OUTPUT_DIR` environment variable, making it easy to save and restore state without hardcoding GCS paths or cluster-specific details.

> **Important**: By default, Kinetic imposes a 30-day TTL (Time to Live) on the GCS buckets it creates. This means anything written to the default `KINETIC_OUTPUT_DIR` will be automatically deleted after 30 days. If you need to preserve checkpoints longer, you should copy them to a bucket without a lifecycle rule or specify a custom `output_dir`.


## JAX Example

Here is a complete example showing Orbax checkpointing with Kinetic and Auto-Resume. You can find this file at [`examples/example_checkpoint.py`](https://github.com/keras-team/kinetic/blob/main/examples/example_checkpoint.py) in the repository.

```{literalinclude} ../../examples/example_checkpoint.py
```

## Keras Example

The same pattern works for Keras models. Call `model.get_weights()` to produce a PyTree of numpy arrays for Orbax to save, and `model.set_weights()` to restore them on resume. You can find this file at [`examples/example_keras_checkpoint.py`](https://github.com/keras-team/kinetic/blob/main/examples/example_keras_checkpoint.py) in the repository.

```{literalinclude} ../../examples/example_keras_checkpoint.py
```
