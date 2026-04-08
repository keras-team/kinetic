import os

os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras
import numpy as np

import kinetic


@kinetic.run(accelerator="cpu")
def train_keras_jax_model():
  print(f"Keras version: {keras.__version__}")
  print(f"Keras backend: {keras.config.backend()}")
  print(f"JAX version: {jax.__version__}")
  print(f"JAX devices: {jax.devices()}")

  num_classes = 10
  input_shape = (28, 28, 1)

  model = keras.Sequential(
    [
      keras.layers.Input(shape=input_shape),
      keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
      keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
      keras.layers.MaxPooling2D(pool_size=(2, 2)),
      keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
      keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
      keras.layers.GlobalAveragePooling2D(),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(num_classes, activation="softmax"),
    ]
  )
  print("Model defined.")

  model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
      keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
  )
  print("Model compiled.")

  # Dummy data
  num_samples = 1024
  x_train = np.random.rand(num_samples, *input_shape).astype(np.float32)
  y_train = np.random.randint(0, num_classes, size=(num_samples,)).astype(
    np.int32
  )

  print("Starting model.fit...")
  history = model.fit(x_train, y_train, epochs=5, batch_size=32)
  print("Model.fit finished.")

  return history.history["loss"][-1]


if __name__ == "__main__":
  print("Starting Keras JAX demo...")
  loss = train_keras_jax_model()
  print(f"Final training loss: {loss}")
