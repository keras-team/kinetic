"""
Fashion MNIST on TPU
====================

Train a simple CNN on the Fashion MNIST dataset using a TPU accelerator.
Demonstrates the basic ``@kinetic.run`` workflow: define a training function,
decorate it, and call it — Kinetic handles the remote execution.
"""

import kinetic


@kinetic.run(accelerator="tpu-v5litepod-1")
def train_fashion_mnist():
  import keras
  import numpy as np

  # Load and preprocess the Fashion MNIST dataset
  (x_train, y_train), (x_test, y_test) = (
    keras.datasets.fashion_mnist.load_data()
  )
  x_train = x_train.astype("float32") / 255.0
  x_test = x_test.astype("float32") / 255.0
  x_train = np.expand_dims(x_train, -1)
  x_test = np.expand_dims(x_test, -1)

  # Build a simple convolutional model
  model = keras.Sequential(
    [
      keras.layers.Input(shape=(28, 28, 1)),
      keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
      keras.layers.MaxPooling2D(pool_size=(2, 2)),
      keras.layers.Flatten(),
      keras.layers.Dense(128, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
    ]
  )

  model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
  )

  # Train for a few epochs on the remote TPU
  model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

  # Evaluate and return results
  score = model.evaluate(x_test, y_test, verbose=0)
  return f"Test loss: {score[0]:.4f}, Test accuracy: {score[1]:.4f}"


if __name__ == "__main__":
  result = train_fashion_mnist()
  print(result)
