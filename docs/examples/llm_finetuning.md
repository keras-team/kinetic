# Fine-tuning LLMs

Kinetic integrates seamlessly with [Keras Hub](https://keras.io/keras_hub/) and the [Kaggle](https://www.kaggle.com/) ecosystem, making it easy to fine-tune large language models like Gemma on cloud TPUs.

## Capturing Credentials

When fine-tuning models from Keras Hub or Kaggle, you often need to provide credentials (`KAGGLE_USERNAME`, `KAGGLE_KEY`). Use the `capture_env_vars` parameter to securely forward your local environment variables to the remote worker.

```python
import kinetic

@kinetic.run(
    accelerator="tpu-v5litepod-1",
    capture_env_vars=["KAGGLE_*", "GOOGLE_CLOUD_*"]
)
def train_gemma():
    import keras_hub
    # Credentials are automatically available in the remote environment
    gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_1b")
    # ...
```

## Low-Rank Adaptation (LoRA)

Fine-tuning large models often requires massive memory. LoRA significantly reduces the number of trainable parameters, enabling fine-tuning on smaller accelerator slices.

```python
@kinetic.run(accelerator="tpu-v5litepod-8")
def train_lora():
    import keras_hub
    gemma_lm = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b_en")
    
    # Enable LoRA (rank=4)
    print("Enabling LoRA...")
    gemma_lm.backbone.enable_lora(rank=4)
    
    # Train as usual
    gemma_lm.fit(train_data, epochs=3)
    return "Training complete!"
```

## Distributed Fine-tuning

For larger models or datasets, use the Pathways backend to distribute training across multiple TPU hosts.

```python
@kinetic.run(
    accelerator="tpu-v6e-8",
    backend="pathways"
)
def train_distributed():
    import keras
    import jax
    # Multi-host TPU environment is auto-initialized
    # ...
```

See the [Distributed Training](../guides/distributed_training.md) guide for more details on scaling your workloads.

## Related pages

- [Distributed Training](../guides/distributed_training.md) — multi-host
  Pathways setup that LLM fine-tuning typically needs.
- [Checkpointing](../guides/checkpointing.md) — Orbax + `KINETIC_OUTPUT_DIR`
  for resumable fine-tuning runs.
- [Examples](../examples.md) — the Gemma SFT examples are full
  end-to-end LLM fine-tuning walkthroughs.
throughs.
