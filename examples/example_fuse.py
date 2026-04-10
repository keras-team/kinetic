"""GCS FUSE examples — lazy-mount data instead of downloading it.

Passing ``fuse=True`` to ``Data(...)`` tells Kinetic to mount the data via the
GCS FUSE CSI driver rather than downloading it into the container.  This is
useful for large datasets where you only need to read a subset of the files at
runtime.

Prerequisites:
  * A GKE cluster with the GCS FUSE CSI driver addon enabled
    (``kinetic up`` enables it by default).
"""

import json
import os
import tempfile

import kinetic
from kinetic import Data

_tmp_root = tempfile.mkdtemp(prefix="kn-fuse-example-")
_counter = 0


def _fresh_dir(name: str) -> str:
  """Return a unique temp directory for each test to avoid cross-test state."""
  global _counter
  _counter += 1
  path = os.path.join(_tmp_root, f"{_counter}_{name}")
  os.makedirs(path, exist_ok=True)
  return path


print(f"Temp root: {_tmp_root}\n")

dataset_dir = _fresh_dir("dataset")
with open(os.path.join(dataset_dir, "train.csv"), "w") as f:
  f.write("feature,label\n1,100\n2,200\n3,300\n")


@kinetic.run(
  accelerator="cpu",
  volumes={"/data": Data(dataset_dir, fuse=True)},
)
def read_fuse_volume():
  """1. FUSE volume — directory mounted at a fixed path."""
  files = sorted(os.listdir("/data"))
  with open("/data/train.csv") as f:
    content = f.read()
  return {"files": files, "content": content}


result = read_fuse_volume()
print(f"Test 1 (fuse volume): {result}")
assert result["files"] == ["train.csv"]
assert "1,100" in result["content"]

nested_dataset = _fresh_dir("nested")
with open(os.path.join(nested_dataset, "root.txt"), "w") as f:
  f.write("root data")
sub = os.path.join(nested_dataset, "subdir")
os.makedirs(sub)
with open(os.path.join(sub, "nested.txt"), "w") as f:
  f.write("nested data")


@kinetic.run(
  accelerator="cpu",
  volumes={"/data": Data(nested_dataset, fuse=True)},
)
def read_nested():
  """2. FUSE volume preserves nested directories."""
  root_files = sorted(os.listdir("/data"))
  with open("/data/subdir/nested.txt") as f:
    nested = f.read()
  return {"root_files": root_files, "nested": nested}


result = read_nested()
print(f"Test 2 (nested dirs): {result}")
assert "subdir" in result["root_files"]
assert "root.txt" in result["root_files"]
assert "nested" in result["nested"]

data_dir = _fresh_dir("data")
with open(os.path.join(data_dir, "data.csv"), "w") as f:
  f.write("data,100")

weights_dir = _fresh_dir("weights")
with open(os.path.join(weights_dir, "model.bin"), "w") as f:
  f.write("pretrained-weights")


@kinetic.run(
  accelerator="cpu",
  volumes={
    "/data": Data(data_dir, fuse=True),
    "/weights": Data(weights_dir, fuse=True),
  },
)
def check_multiple_volumes():
  """3. Multiple FUSE volumes."""
  return {
    "data_files": sorted(os.listdir("/data")),
    "weight_files": sorted(os.listdir("/weights")),
  }


result = check_multiple_volumes()
print(f"Test 3 (multiple fuse volumes): {result}")
assert result["data_files"] == ["data.csv"]
assert result["weight_files"] == ["model.bin"]

arg_dataset = _fresh_dir("arg_dataset")
with open(os.path.join(arg_dataset, "train.csv"), "w") as f:
  f.write("feature,label\n1,100\n2,200\n3,300\n")


@kinetic.run(accelerator="cpu")
def read_fuse_arg(data_path):
  """4. FUSE data as a function argument."""
  files = sorted(os.listdir(data_path))
  with open(f"{data_path}/train.csv") as f:
    content = f.read()
  return {"files": files, "content": content}


result = read_fuse_arg(Data(arg_dataset, fuse=True))
print(f"Test 4 (fuse data arg): {result}")
assert result["files"] == ["train.csv"]
assert "1,100" in result["content"]

config_json = os.path.join(_fresh_dir("config"), "config.json")
with open(config_json, "w") as f:
  json.dump({"lr": 0.01, "epochs": 10}, f)


@kinetic.run(accelerator="cpu")
def read_fuse_file(config_path):
  """5. FUSE data arg — single file."""
  with open(config_path) as f:
    return json.load(f)


result = read_fuse_file(Data(config_json, fuse=True))
print(f"Test 5 (fuse single file): {result}")
assert result["lr"] == 0.01

fuse_dir = _fresh_dir("fuse_data")
with open(os.path.join(fuse_dir, "fuse.txt"), "w") as f:
  f.write("fuse content")

dl_dir = _fresh_dir("dl_data")
with open(os.path.join(dl_dir, "dl.txt"), "w") as f:
  f.write("downloaded content")


@kinetic.run(
  accelerator="cpu",
  volumes={
    "/fuse_data": Data(fuse_dir, fuse=True),
    "/dl_data": Data(dl_dir),
  },
)
def mixed_volumes():
  """6. Mixed — FUSE volume + downloaded volume."""
  with open("/fuse_data/fuse.txt") as f:
    fuse_content = f.read()
  with open("/dl_data/dl.txt") as f:
    dl_content = f.read()
  return {"fuse": fuse_content, "dl": dl_content}


result = mixed_volumes()
print(f"Test 6 (fuse + downloaded volumes): {result}")
assert "fuse" in result["fuse"]
assert "downloaded" in result["dl"]

wt_dir = _fresh_dir("model_weights")
with open(os.path.join(wt_dir, "model.bin"), "w") as f:
  f.write("pretrained-weights")

cfg_json = os.path.join(_fresh_dir("train_config"), "config.json")
with open(cfg_json, "w") as f:
  json.dump({"lr": 0.01, "epochs": 10}, f)


@kinetic.run(
  accelerator="cpu",
  volumes={"/weights": Data(wt_dir, fuse=True)},
)
def train(config_path, lr=0.001):
  """7. Mixed — FUSE volume + Data arg + plain arg."""
  with open(config_path) as f:
    cfg = json.load(f)
  has_weights = os.path.isdir("/weights")
  weight_files = sorted(os.listdir("/weights"))
  return {
    "config": cfg,
    "lr": lr,
    "has_weights": has_weights,
    "weight_files": weight_files,
  }


result = train(Data(cfg_json), lr=0.05)
print(f"Test 7 (fuse volume + data arg + plain arg): {result}")
assert result["config"]["lr"] == 0.01
assert result["lr"] == 0.05
assert result["has_weights"] is True
assert result["weight_files"] == ["model.bin"]

print("\nAll FUSE examples passed!")
