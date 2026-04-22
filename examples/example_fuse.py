"""GCS FUSE examples: lazy-mount data instead of downloading it.

Passing ``fuse=True`` to ``Data(...)`` tells Kinetic to mount the data via the
GCS FUSE CSI driver rather than downloading it into the container. This is
useful for large datasets where you only need to read a subset of the files at
runtime.

Prerequisites:
  * A GKE cluster with the GCS FUSE CSI driver addon enabled
    (``kinetic up`` enables it by default).

Note on local vs. remote execution:
The examples below create local temp directories and pass them to
``@kinetic.run`` functions. Although the data starts on your laptop,
Kinetic uploads it to GCS behind the scenes and then *mounts* it back
into the remote pod via the GCS FUSE CSI driver: the pod never
downloads a full copy. This is the key difference from the default
behaviour (``fuse=False``), where data is downloaded into the container
before execution.
"""

import json
import os
import shutil
import tempfile

import kinetic
from kinetic import Data

# Decorators that do not need to close over any local state can be defined at
# module scope. Test 1 reads from a fixed mount path that is bound when the
# decorator is built, so the decorator must be constructed inside main() once
# the temp dir is known. The same applies to tests 2, 3, 6, and 7.


@kinetic.run(accelerator="cpu")
def read_fuse_arg(data_path):
  """Receive FUSE data as a function argument instead of a volume mount."""
  files = sorted(os.listdir(data_path))
  with open(f"{data_path}/train.csv") as f:
    content = f.read()
  return {"files": files, "content": content}


@kinetic.run(accelerator="cpu")
def read_fuse_file(config_path):
  """Read a single FUSE-mounted file (e.g. a JSON config)."""
  with open(config_path) as f:
    return json.load(f)


def make_read_fuse_volume(dataset_dir):
  @kinetic.run(
    accelerator="cpu",
    volumes={"/data": Data(dataset_dir, fuse=True)},
  )
  def read_fuse_volume():
    """Read a FUSE-mounted directory from a fixed volume path."""
    files = sorted(os.listdir("/data"))
    with open("/data/train.csv") as f:
      content = f.read()
    return {"files": files, "content": content}

  return read_fuse_volume


def make_read_nested(nested_dataset):
  @kinetic.run(
    accelerator="cpu",
    volumes={"/data": Data(nested_dataset, fuse=True)},
  )
  def read_nested():
    """Verify that a FUSE-mounted volume preserves nested directories."""
    root_files = sorted(os.listdir("/data"))
    with open("/data/subdir/nested.txt") as f:
      nested = f.read()
    return {"root_files": root_files, "nested": nested}

  return read_nested


def make_check_multiple_volumes(data_dir, weights_dir):
  @kinetic.run(
    accelerator="cpu",
    volumes={
      "/data": Data(data_dir, fuse=True),
      "/weights": Data(weights_dir, fuse=True),
    },
  )
  def check_multiple_volumes():
    """Mount multiple FUSE volumes at distinct paths.

    A common scenario: your training job reads training data from one
    bucket path and pretrained model weights from another. Both can
    be lazily mounted so the pod only fetches files it actually opens.
    """
    return {
      "data_files": sorted(os.listdir("/data")),
      "weight_files": sorted(os.listdir("/weights")),
    }

  return check_multiple_volumes


def make_mixed_volumes(fuse_dir, dl_dir):
  @kinetic.run(
    accelerator="cpu",
    volumes={
      "/fuse_data": Data(fuse_dir, fuse=True),
      "/dl_data": Data(dl_dir),
    },
  )
  def mixed_volumes():
    """Combine a FUSE-mounted volume with a regular downloaded volume."""
    with open("/fuse_data/fuse.txt") as f:
      fuse_content = f.read()
    with open("/dl_data/dl.txt") as f:
      dl_content = f.read()
    return {"fuse": fuse_content, "dl": dl_content}

  return mixed_volumes


def make_train(wt_dir):
  @kinetic.run(
    accelerator="cpu",
    volumes={"/weights": Data(wt_dir, fuse=True)},
  )
  def train(config_path, lr=0.001):
    """Combine a FUSE volume, a Data argument, and a plain Python argument."""
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

  return train


def main():
  tmp_root = tempfile.mkdtemp(prefix="kn-fuse-example-")
  print(f"Temp root: {tmp_root}\n")

  counter = 0

  def fresh_dir(name):
    nonlocal counter
    counter += 1
    path = os.path.join(tmp_root, f"{counter}_{name}")
    os.makedirs(path, exist_ok=True)
    return path

  try:
    # Test 1: Basic FUSE volume mount.
    # Create a small CSV dataset locally. Kinetic uploads it to GCS and the
    # remote pod mounts it via FUSE at /data, with no full download.
    dataset_dir = fresh_dir("dataset")
    with open(os.path.join(dataset_dir, "train.csv"), "w") as f:
      f.write("feature,label\n1,100\n2,200\n3,300\n")

    read_fuse_volume = make_read_fuse_volume(dataset_dir)
    result = read_fuse_volume()
    print(f"Test 1 (fuse volume): {result}")
    assert result["files"] == ["train.csv"]
    assert "1,100" in result["content"]

    # Test 2: Nested directory structure.
    # FUSE mounts preserve the full directory tree, including subdirectories.
    nested_dataset = fresh_dir("nested")
    with open(os.path.join(nested_dataset, "root.txt"), "w") as f:
      f.write("root data")
    sub = os.path.join(nested_dataset, "subdir")
    os.makedirs(sub)
    with open(os.path.join(sub, "nested.txt"), "w") as f:
      f.write("nested data")

    read_nested = make_read_nested(nested_dataset)
    result = read_nested()
    print(f"Test 2 (nested dirs): {result}")
    assert "subdir" in result["root_files"]
    assert "root.txt" in result["root_files"]
    assert "nested" in result["nested"]

    # Test 3: Multiple FUSE volumes.
    # Useful when a job needs both training data and pretrained weights from
    # separate locations, without downloading either fully.
    data_dir = fresh_dir("data")
    with open(os.path.join(data_dir, "data.csv"), "w") as f:
      f.write("data,100")

    weights_dir = fresh_dir("weights")
    with open(os.path.join(weights_dir, "model.bin"), "w") as f:
      f.write("pretrained-weights")

    check_multiple_volumes = make_check_multiple_volumes(data_dir, weights_dir)
    result = check_multiple_volumes()
    print(f"Test 3 (multiple fuse volumes): {result}")
    assert result["data_files"] == ["data.csv"]
    assert result["weight_files"] == ["model.bin"]

    # Test 4: FUSE data passed as a function argument.
    # Instead of mounting at a fixed volume path, pass Data directly as a
    # function argument. Kinetic resolves it to a local path on the pod.
    arg_dataset = fresh_dir("arg_dataset")
    with open(os.path.join(arg_dataset, "train.csv"), "w") as f:
      f.write("feature,label\n1,100\n2,200\n3,300\n")

    result = read_fuse_arg(Data(arg_dataset, fuse=True))
    print(f"Test 4 (fuse data arg): {result}")
    assert result["files"] == ["train.csv"]
    assert "1,100" in result["content"]

    # Test 5: Single file via FUSE.
    # FUSE works for single files too. Handy for config files or small
    # artifacts you don't want baked into the container image.
    config_json = os.path.join(fresh_dir("config"), "config.json")
    with open(config_json, "w") as f:
      json.dump({"lr": 0.01, "epochs": 10}, f)

    result = read_fuse_file(Data(config_json, fuse=True))
    print(f"Test 5 (fuse single file): {result}")
    assert result["lr"] == 0.01

    # Test 6: Mixing FUSE and downloaded volumes.
    # Combine FUSE-mounted and downloaded volumes in the same job. Use
    # fuse=True for large datasets you only partially read, and the default
    # download mode for smaller data that benefits from local I/O.
    fuse_dir = fresh_dir("fuse_data")
    with open(os.path.join(fuse_dir, "fuse.txt"), "w") as f:
      f.write("fuse content")

    dl_dir = fresh_dir("dl_data")
    with open(os.path.join(dl_dir, "dl.txt"), "w") as f:
      f.write("downloaded content")

    mixed_volumes = make_mixed_volumes(fuse_dir, dl_dir)
    result = mixed_volumes()
    print(f"Test 6 (fuse + downloaded volumes): {result}")
    assert "fuse" in result["fuse"]
    assert "downloaded" in result["dl"]

    # Test 7: Realistic training setup. FUSE volume + Data arg + plain arg.
    # Mount pretrained weights via FUSE, pass a config file as a Data
    # argument, and supply plain Python arguments (lr).
    wt_dir = fresh_dir("model_weights")
    with open(os.path.join(wt_dir, "model.bin"), "w") as f:
      f.write("pretrained-weights")

    cfg_json = os.path.join(fresh_dir("train_config"), "config.json")
    with open(cfg_json, "w") as f:
      json.dump({"lr": 0.01, "epochs": 10}, f)

    train = make_train(wt_dir)
    result = train(Data(cfg_json), lr=0.05)
    print(f"Test 7 (fuse volume + data arg + plain arg): {result}")
    assert result["config"]["lr"] == 0.01
    assert result["lr"] == 0.05
    assert result["has_weights"] is True
    assert result["weight_files"] == ["model.bin"]

    print("\nAll FUSE examples passed!")
  finally:
    shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
  main()
