import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np
from keras import layers

import kinetic


# A simple model that will be executed remotely on pathways
@kinetic.run(accelerator="v5litepod-1", backend="pathways")
def train_simple_model():
  print("Running Pathways job on JAX Backend!")

  # Create a simple dataset
  x = np.random.rand(1000, 10)
  y = np.random.randint(0, 2, size=(1000, 1))

  # A simple sequential model
  model = keras.Sequential(
    [
      keras.Input(shape=(10,)),
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(1, activation="sigmoid"),
    ]
  )

  model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
  )

  print("Model Architecture:")
  model.summary()

  # Train the model
  print("\nStarting Training...")
  history = model.fit(x, y, epochs=5, batch_size=32, validation_split=0.2)

  print("\nTraining completed successfully on Pathways!")
  return history.history


if __name__ == "__main__":
  print("Submitting Pathways training job...")
  result_history = train_simple_model()
  print("Final validation accuracy:", result_history["val_accuracy"][-1])
