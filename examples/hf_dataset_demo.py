import time

import kinetic


@kinetic.run(accelerator="tpu-v5litepod-4")
def train_with_hf_dataset(dataset_path: str):
  """
  This is the remote function. It natively receives a local file path
  to the resolved dataset from Kinetic.
  """
  print(f"User code received dataset at: {dataset_path}")

  # User can load it directly from disk using the HF datasets library
  from datasets import load_from_disk

  start = time.time()
  ds = load_from_disk(dataset_path)
  print(f"Loaded dataset from disk in {time.time() - start:.2f}s")

  print(f"Dataset type: {type(ds)}")
  if hasattr(ds, "column_names"):
    print(f"Columns: {ds.column_names}")
  print(f"Number of rows: {len(ds)}")

  # Just printing the first example
  print("First item:", ds[0])

  return "done"


if __name__ == "__main__":
  print("Submitting Kinetic job with Hugging Face Dataset dependency...")

  # Pass the Data object with hf:// URI as an argument.
  dataset_ref = kinetic.Data(
    "hf://smartcat/Amazon_All_Beauty_2018?config_name=reviews&split=train"
  )

  result = train_with_hf_dataset(dataset_ref)

  print("Result:", result)
