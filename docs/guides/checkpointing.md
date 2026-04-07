# Checkpointing and Auto-Resume

This guide demonstrates how to use Orbax for checkpointing in Kinetic workloads. Kinetic automatically sets up an output directory and propagates it via the `KINETIC_OUTPUT_DIR` environment variable, making it easy to save and restore state without hardcoding GCS paths or cluster-specific details.

## Example

Here is a complete example showing Orbax checkpointing with Kinetic and Auto-Resume. You can find this file at `examples/example_checkpoint.py` in the repository.

```{literalinclude} ../../examples/example_checkpoint.py
```
