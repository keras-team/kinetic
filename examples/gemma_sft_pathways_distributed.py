import os

# JAX must be set as the backend before importing Keras
os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras
import keras_hub

import kinetic


@kinetic.run(
  accelerator="v5litepod-2x4",
  cluster="keras-team-dogfood",
  project="keras-team-gcp",
  backend="pathways",
  capture_env_vars=["KAGGLE_USERNAME", "KAGGLE_KEY"],
)
def finetune_gemma_distributed():
  """
  Distributed Finetuning of Gemma using Keras Remote + ML Pathways on TPUs.
  This mirrors the Kaggle DataParallel fine-tuning methodology, but executed remotely.
  """
  print(
    f"Number of JAX devices available across all hosts: {jax.device_count()}"
  )

  # 1. Setup Data Parallel Distribution using a DeviceMesh
  # We construct a 1D mesh wrapping all available TPUs in the Pathways cluster
  devices = keras.distribution.list_devices()
  device_mesh = keras.distribution.DeviceMesh(
    shape=(len(devices),),
    axis_names=["batch"],
    devices=devices,
  )

  # Set global distribution to DataParallel
  keras.distribution.set_distribution(
    keras.distribution.DataParallel(device_mesh=device_mesh)
  )
  print(f"Global distribution set to DataParallel across {len(devices)} TPUs.")

  # 2. Load Gemma Model via Keras Hub
  print("Loading Gemma 2B model...")
  gemma_lm = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b_en")

  # 3. Enable LoRA (Low-Rank Adaptation)
  # This drastically reduces the number of trainable parameters
  print("Enabling LoRA (rank=4)...")
  gemma_lm.backbone.enable_lora(rank=4)
  gemma_lm.summary()

  # 4. Prepare Fine-tuning Data
  # In a real environment, you might load TFRecords from GCS here.
  data = [
    "Patient: I have a sore throat and slight fever. Doctor: You might have a mild infection. Make sure to rest and drink fluids.",
    "Patient: My ankle hurts when I put weight on it. Doctor: It sounds like a sprain. Try keeping it elevated and apply ice.",
    "Patient: I've been feeling very tired lately. Doctor: Fatigue can be caused by many things. Are you getting enough sleep?",
    "Patient: I have a rash on my arm. Doctor: Is it itchy or painful? I can prescribe a topical cream.",
  ]
  # Duplicate data to fill batches across multiple TPUs
  train_data = data * 8

  # 5. Compile the Model
  optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
  )
  # Exclude LayerNorms and biases from weight decay per standard practice
  optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

  gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
  )

  # 6. Run Distributed Fine-tuning
  print("\nStarting Distributed Fine-Tuning across Pathways Cluster...")
  gemma_lm.fit(train_data, batch_size=len(devices), epochs=3)
  print("\nTraining completed successfully!")

  # 7. Local Evaluation Test
  prompt = (
    "Patient: I've had a headache that won't go away for two days. Doctor: "
  )
  output = gemma_lm.generate([prompt], max_length=64)

  gen_output = f"\nGenerative Output Test:\n{output[0]}"
  print(gen_output)

  return gen_output


if __name__ == "__main__":
  # Make sure you have exported your Kaggle credentials locally so they are sent to the cluster:
  # export KAGGLE_USERNAME="your_username"
  # export KAGGLE_KEY="your_key"

  print("Submitting Gemma Distributed Finetuning job to Pathways Backend...")
  result = finetune_gemma_distributed()
  print("Job Finished With Result:", result)
