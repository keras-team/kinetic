# Training Keras Models

Kinetic makes it easy to take a standard Keras training script and execute it on high-performance cloud accelerators with minimal changes.

## Basic Usage

To run a Keras model remotely, wrap your training logic in a function and apply the `@kinetic.run()` decorator.

```python
import kinetic

@kinetic.run(accelerator="tpu-v6e-8")
def train_model():
    import keras
    import numpy as np

    # Define a simple model
    model = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(10,)),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    # Generate or load data
    x_train = np.random.randn(1000, 10)
    y_train = np.random.randn(1000, 1)

    # Train the model
    history = model.fit(x_train, y_train, epochs=5, verbose=0)
    
    # Return any result (it will be serialized back to your local machine)
    return history.history["loss"][-1]

# This call triggers the remote execution pipeline
final_loss = train_model()
print(f"Final loss: {final_loss}")
```

## How it Works

When you call a decorated function:
1. **Packaging**: Kinetic captures your function and any local code dependencies.
2. **Provisioning**: It ensures the requested accelerator (e.g., `tpu-v6e-8` TPU) is available in your GKE cluster.
3. **Execution**: The function runs inside a container on the remote node.
4. **Streaming**: Logs are streamed back to your terminal in real-time.
5. **Return**: The function's return value is serialized and returned to your local process.

## Performance Tips

- **In-function Imports**: Import heavy libraries like `keras`, `jax`, or `tensorflow` *inside* the decorated function. This keeps your local environment light and ensures the remote worker uses its own optimized installations.
- **Batch Size**: Accelerators perform best with large batch sizes. Ensure your `batch_size` in `model.fit()` is tuned for the specific hardware you've requested.
- **Data Loading**: For the best performance, use the :doc:`data` API to handle data dependencies efficiently.
