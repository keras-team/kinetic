import keras_remote
from keras_remote import Data


# --- Test 1: Data as function arg (local directory) ---
@keras_remote.run(accelerator="cpu")
def test_data_arg(data_dir):
  import os

  files = sorted(os.listdir(data_dir))
  with open(f"{data_dir}/train.csv") as f:
    content = f.read()
  return {"files": files, "content": content}


result = test_data_arg(Data("/tmp/kr-data-test/dataset/"))
print(f"Test 1 (dir arg): {result}")
assert result["files"] == ["train.csv"]
assert "1,100" in result["content"]


# --- Test 2: Data as function arg (single file) ---
@keras_remote.run(accelerator="cpu")
def test_file_arg(config_path):
  import json

  with open(config_path) as f:
    return json.load(f)


result = test_file_arg(Data("/tmp/kr-data-test/config.json"))
print(f"Test 2 (file arg): {result}")
assert result["lr"] == 0.01

# --- Test 3: Cache hit (re-run same data, check logs for "cache hit") ---
result = test_file_arg(Data("/tmp/kr-data-test/config.json"))
print(f"Test 3 (cache hit): {result}")
assert result["lr"] == 0.01


# --- Test 4: volumes (fixed-path mount) ---
@keras_remote.run(
  accelerator="cpu",
  volumes={"/data": Data("/tmp/kr-data-test/dataset/")},
)
def test_volumes():
  import os

  files = sorted(os.listdir("/data"))
  with open("/data/train.csv") as f:
    content = f.read()
  return {"files": files, "content": content}


result = test_volumes()
print(f"Test 4 (volumes): {result}")
assert result["files"] == ["train.csv"]


# --- Test 5: Mixed — volumes + Data arg + plain arg ---
@keras_remote.run(
  accelerator="cpu",
  volumes={"/weights": Data("/tmp/kr-data-test/dataset/")},
)
def test_mixed(config_path, lr=0.001):
  import json
  import os

  with open(config_path) as f:
    cfg = json.load(f)
  has_weights = os.path.isdir("/weights")
  return {"config": cfg, "lr": lr, "has_weights": has_weights}


result = test_mixed(Data("/tmp/kr-data-test/config.json"), lr=0.01)
print(f"Test 5 (mixed): {result}")
assert result["config"]["lr"] == 0.01
assert result["lr"] == 0.01
assert result["has_weights"] is True


# --- Test 6: Data in nested structure ---
@keras_remote.run(accelerator="cpu")
def test_nested(datasets):
  import os

  return [sorted(os.listdir(d)) for d in datasets]


result = test_nested(
  datasets=[
    Data("/tmp/kr-data-test/dataset/"),
    Data("/tmp/kr-data-test/dataset/"),
  ]
)
print(f"Test 6 (nested): {result}")
assert len(result) == 2

print("\nAll E2E tests passed!")
