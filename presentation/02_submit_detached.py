import os

os.environ["KERAS_BACKEND"] = "jax"

import kinetic


@kinetic.run(accelerator="cpu")
def long_running_train():
  import time

  import keras
  import numpy as np

  print("Starting a long-running training job...")
  model = keras.Sequential(
    [
      keras.layers.Dense(128, activation="relu", input_shape=(20,)),
      keras.layers.Dense(128, activation="relu"),
      keras.layers.Dense(1),
    ]
  )
  model.compile(optimizer="adam", loss="mse")

  x = np.random.randn(5000, 20)
  y = np.random.randn(5000, 1)

  final_loss = 0.0
  for epoch in range(20):
    history = model.fit(x, y, epochs=1, batch_size=64, verbose=0)
    final_loss = float(history.history["loss"][-1])
    print(f"  epoch {epoch + 1}/20  loss={final_loss:.4f}")
    time.sleep(2)  # exaggerate length so the demo feels long-running

  return {"final_loss": final_loss}


if __name__ == "__main__":
  handle = long_running_train.run_async()
  print("\n" + "=" * 60)
  print(f"  JOB SUBMITTED — id: {handle.job_id}")
  print("=" * 60)
  print("\nThis process is now exiting. The job keeps running remotely.")
  print("Attach from any other shell with:")
  print(
    f"\n    python presentation/03_attach_from_anywhere.py {handle.job_id}\n"
  )
  print("Or list every live job on the cluster with:")
  print(
    "\n    python -c 'import kinetic; [print(j.job_id, j.func_name) "
    "for j in kinetic.list_jobs()]'\n"
  )
