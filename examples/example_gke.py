"""
Example: Using kinetic with GKE

This demonstrates running remote functions on a GKE cluster with kinetic.

Prerequisites:
1. A GKE cluster (CPU or with GPU node pools)
2. kubectl configured to access the cluster
3. KERAS_REMOTE_PROJECT environment variable set

Setup (CPU cluster - works out of the box):
    ./setup.sh  # Answer 'yes' when prompted for GKE setup

Setup (GPU cluster - for GPU examples):
    # Add a GPU node pool to existing cluster
    gcloud container node-pools create gpu-pool \\
        --cluster kinetic-cluster \\
        --zone us-central1-a \\
        --machine-type n1-standard-4 \\
        --accelerator type=nvidia-tesla-t4,count=1 \\
        --num-nodes 1 \\
        --scopes gke-default,storage-full

    # Install NVIDIA GPU drivers
    kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

Supported accelerators:
    - cpu: CPU only (no GPU required)
    - nvidia-tesla-t4, t4: NVIDIA T4
    - nvidia-l4, l4: NVIDIA L4
    - nvidia-tesla-v100, v100: NVIDIA V100
    - nvidia-tesla-a100, a100: NVIDIA A100 (40GB)
    - a100-80gb: NVIDIA A100 (80GB)
    - nvidia-h100-80gb, h100: NVIDIA H100
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np

import kinetic


# Example 1: CPU-only execution (works with default cluster)
@kinetic.run(accelerator="cpu")
def simple_computation(x, y):
  """Simple addition that runs on remote CPU."""
  result = x + y
  print(f"Computing {x} + {y} = {result}")
  return result


# Example 2: Keras model training on CPU
@kinetic.run(accelerator="cpu")
def train_simple_model_cpu():
  """Train a simple Keras model on remote CPU."""

  # Create a simple model
  model = keras.Sequential(
    [
      keras.layers.Dense(64, activation="relu", input_shape=(10,)),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(1),
    ]
  )

  model.compile(optimizer="adam", loss="mse")

  # Generate some dummy data
  x_train = np.random.randn(1000, 10)
  y_train = np.random.randn(1000, 1)

  # Train the model
  print("Training model on CPU...")
  history = model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)

  print(f"Final loss: {history.history['loss'][-1]}")
  return history.history["loss"][-1]


# Example 3: GPU training (requires GPU node pool)
@kinetic.run(accelerator="nvidia-tesla-t4")
def train_model_gpu():
  """Train a Keras model on remote GPU. Requires T4 GPU node pool."""
  model = keras.Sequential(
    [
      keras.layers.Dense(128, activation="relu", input_shape=(20,)),
      keras.layers.Dense(128, activation="relu"),
      keras.layers.Dense(1),
    ]
  )

  model.compile(optimizer="adam", loss="mse")

  x_train = np.random.randn(5000, 20)
  y_train = np.random.randn(5000, 1)

  print("Training model on T4 GPU...")
  history = model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1)

  return history.history["loss"][-1]


def main():
  """Run examples."""
  print("=" * 60)
  print("Keras Remote - GKE Examples")
  print("=" * 60)

  # Example 1: Simple computation (CPU)
  print("\n--- Example 1: Simple Computation (CPU) ---")
  print("Running simple_computation(10, 20) on GKE...")
  result = simple_computation(10, 20)
  print(f"Result: {result}")

  # Example 2: Model training on CPU
  print("\n--- Example 2: Keras Model Training (CPU) ---")
  print("Training a simple model on CPU...")
  final_loss = train_simple_model_cpu()
  print(f"Model trained. Final loss: {final_loss}")

  # Example 3: GPU training (requires GPU node pool)
  # Uncomment to run if you have T4 GPU nodes available
  # print("\n--- Example 3: Model Training on T4 GPU ---")
  # final_loss = train_model_gpu()
  # print(f"Model trained. Final loss: {final_loss}")

  print("\n" + "=" * 60)
  print("Examples completed!")
  print("=" * 60)


if __name__ == "__main__":
  # Prerequisites:
  # 1. Set KERAS_REMOTE_PROJECT environment variable to your GCP project ID
  #    (if `project` param is not provided in the decorator)
  # 2. Ensure your GKE cluster has GPU nodes with the required accelerator type
  main()
