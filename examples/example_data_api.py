import json
import os
import tempfile

import kinetic
from kinetic import Data


# Data as function arg (local directory)
@kinetic.run(accelerator="cpu")
def test_data_arg(data_dir):
  files = sorted(os.listdir(data_dir))
  with open(f"{data_dir}/train.csv") as f:
    content = f.read()
  return {"files": files, "content": content}


# Data as function arg (single file)
@kinetic.run(accelerator="cpu")
def test_file_arg(config_path):
  with open(config_path) as f:
    return json.load(f)


# volumes (fixed-path mount). The volumes path is bound at decoration time,
# so we build the decorator inside main() once we know the temp dir.
def make_volumes_test(dataset_dir):
  @kinetic.run(
    accelerator="cpu",
    volumes={"/data": Data(dataset_dir)},
  )
  def test_volumes():
    files = sorted(os.listdir("/data"))
    with open("/data/train.csv") as f:
      content = f.read()
    return {"files": files, "content": content}

  return test_volumes


def make_mixed_test(dataset_dir):
  @kinetic.run(
    accelerator="cpu",
    volumes={"/weights": Data(dataset_dir)},
  )
  def test_mixed(config_path, lr=0.001):
    with open(config_path) as f:
      cfg = json.load(f)
    has_weights = os.path.isdir("/weights")
    return {"config": cfg, "lr": lr, "has_weights": has_weights}

  return test_mixed


# Data in nested structure
@kinetic.run(accelerator="cpu")
def test_nested(datasets):
  return [sorted(os.listdir(d)) for d in datasets]


def main():
  # Setup: create temporary dummy data
  tmp_dir = tempfile.mkdtemp(prefix="kn-data-example-")
  dataset_dir = os.path.join(tmp_dir, "dataset")
  os.makedirs(dataset_dir, exist_ok=True)

  # A small CSV file used by several tests below.
  train_csv = os.path.join(dataset_dir, "train.csv")
  with open(train_csv, "w") as f:
    f.write("feature,label\n1,100\n2,200\n3,300\n")

  # A JSON config file used by the single-file and mixed tests.
  config_json = os.path.join(tmp_dir, "config.json")
  with open(config_json, "w") as f:
    json.dump({"lr": 0.01, "epochs": 10}, f)

  print(f"Created temp data in {tmp_dir}\n")

  result = test_data_arg(Data(dataset_dir))
  print(f"Test 1 (dir arg): {result}")
  assert result["files"] == ["train.csv"]
  assert "1,100" in result["content"]

  result = test_file_arg(Data(config_json))
  print(f"Test 2 (file arg): {result}")
  assert result["lr"] == 0.01

  # Cache hit (re-run same data, check logs for "cache hit")
  result = test_file_arg(Data(config_json))
  print(f"Test 3 (cache hit): {result}")
  assert result["lr"] == 0.01

  test_volumes = make_volumes_test(dataset_dir)
  result = test_volumes()
  print(f"Test 4 (volumes): {result}")
  assert result["files"] == ["train.csv"]

  test_mixed = make_mixed_test(dataset_dir)
  result = test_mixed(Data(config_json), lr=0.01)
  print(f"Test 5 (mixed): {result}")
  assert result["config"]["lr"] == 0.01
  assert result["lr"] == 0.01
  assert result["has_weights"] is True

  result = test_nested(
    datasets=[
      Data(dataset_dir),
      Data(dataset_dir),
    ]
  )
  print(f"Test 6 (nested): {result}")
  assert len(result) == 2

  print("\nAll E2E tests passed!")


if __name__ == "__main__":
  main()
