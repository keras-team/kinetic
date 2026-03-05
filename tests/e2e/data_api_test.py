"""E2E tests for the Data API — data args, volumes, caching, and mixed usage.


Data files are created with unique content per test run (via a run-level
nonce) so that content-hash cache misses are guaranteed on the first use
and cache hits are guaranteed on re-use within the same run.
"""

import json
import logging
import os
import pathlib
import tempfile
import uuid

from absl.testing import absltest

import keras_remote
from keras_remote import Data
from tests.e2e.e2e_utils import skip_unless_e2e

# Per-run nonce ensures fresh content hashes (no stale cache hits from
# previous test runs whose data is still in GCS).
_RUN_NONCE = uuid.uuid4().hex[:12]


def _make_test_dir(test_case):
  """Create a temp directory cleaned up after the test."""
  td = tempfile.TemporaryDirectory()
  test_case.addCleanup(td.cleanup)
  return pathlib.Path(td.name)


@skip_unless_e2e()
class TestDataAsArg(absltest.TestCase):
  """Data objects passed as function arguments."""

  def test_local_directory(self):
    """A local directory is uploaded, downloaded on the pod, and readable."""
    tmp = _make_test_dir(self)
    data_dir = tmp / "dataset"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text(f"id,value\n1,{_RUN_NONCE}\n")

    @keras_remote.run(accelerator="cpu")
    def read_dir(data_path):

      files = sorted(os.listdir(data_path))
      with open(f"{data_path}/train.csv") as f:
        content = f.read()
      return {"files": files, "content": content}

    result = read_dir(Data(str(data_dir)))

    self.assertEqual(result["files"], ["train.csv"])
    self.assertIn(_RUN_NONCE, result["content"])

  def test_local_single_file(self):
    """A single local file resolves to a file path (not a directory)."""
    tmp = _make_test_dir(self)
    config = tmp / "config.json"
    config.write_text(f'{{"nonce": "{_RUN_NONCE}"}}')

    @keras_remote.run(accelerator="cpu")
    def read_file(config_path):
      with open(config_path) as f:
        return json.load(f)

    result = read_file(Data(str(config)))

    self.assertEqual(result["nonce"], _RUN_NONCE)

  def test_multiple_data_args(self):
    """Multiple Data args are each resolved independently."""
    tmp = _make_test_dir(self)
    d1 = tmp / "train"
    d1.mkdir()
    (d1 / "a.csv").write_text(f"train,{_RUN_NONCE}")
    d2 = tmp / "val"
    d2.mkdir()
    (d2 / "b.csv").write_text(f"val,{_RUN_NONCE}")

    @keras_remote.run(accelerator="cpu")
    def read_both(train_dir, val_dir):

      return {
        "train": sorted(os.listdir(train_dir)),
        "val": sorted(os.listdir(val_dir)),
      }

    result = read_both(Data(str(d1)), Data(str(d2)))

    self.assertEqual(result["train"], ["a.csv"])
    self.assertEqual(result["val"], ["b.csv"])


@skip_unless_e2e()
class TestDataCaching(absltest.TestCase):
  """Content-hash caching uploads once and reuses on subsequent calls."""

  def test_cache_hit_returns_same_result(self):
    """Second call with identical data hits the cache and still works."""
    tmp = _make_test_dir(self)
    config = tmp / "config.json"
    config.write_text(f'{{"cached": true, "nonce": "{_RUN_NONCE}"}}')

    @keras_remote.run(accelerator="cpu")
    def read_config(path):
      with open(path) as f:
        return json.load(f)

    # First call — cache miss (upload happens)
    r1 = read_config(Data(str(config)))

    # Second call — capture logs to verify cache hit
    logger = logging.getLogger("absl")
    with self.assertLogs(logger, level="INFO") as cm:
      r2 = read_config(Data(str(config)))

    log_output = "\n".join(cm.output)
    self.assertIn("Data cache hit", log_output)
    self.assertNotIn("Uploading data", log_output)

    self.assertEqual(r1, r2)
    self.assertTrue(r1["cached"])

  def test_modified_data_gets_new_hash(self):
    """Changing file content produces a different hash and fresh upload."""
    tmp = _make_test_dir(self)
    f = tmp / "data.txt"
    f.write_text(f"version_a_{_RUN_NONCE}")

    @keras_remote.run(accelerator="cpu")
    def read_data(path):
      with open(path) as fh:
        return fh.read()

    r1 = read_data(Data(str(f)))
    self.assertIn("version_a", r1)

    # Modify the file — new content hash → should trigger fresh upload
    f.write_text(f"version_b_{_RUN_NONCE}")
    logger = logging.getLogger("absl")
    with self.assertLogs(logger, level="INFO") as cm:
      r2 = read_data(Data(str(f)))

    log_output = "\n".join(cm.output)
    self.assertIn("Uploading data", log_output)

    self.assertIn("version_b", r2)
    self.assertNotEqual(r1, r2)


@skip_unless_e2e()
class TestVolumes(absltest.TestCase):
  """Data declared in the ``volumes`` decorator parameter."""

  def test_volume_at_fixed_path(self):
    """Volume data is available at the declared mount path."""
    tmp = _make_test_dir(self)
    data_dir = tmp / "dataset"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text(f"id,value\n1,{_RUN_NONCE}\n")

    @keras_remote.run(
      accelerator="cpu",
      volumes={"/data": Data(str(data_dir))},
    )
    def read_volume():

      files = sorted(os.listdir("/data"))
      with open("/data/train.csv") as f:
        content = f.read()
      return {"files": files, "content": content}

    result = read_volume()

    self.assertEqual(result["files"], ["train.csv"])
    self.assertIn(_RUN_NONCE, result["content"])

  def test_volume_cache_hit(self):
    """Second call with same volume data hits the cache."""
    tmp = _make_test_dir(self)
    data_dir = tmp / "dataset"
    data_dir.mkdir()
    (data_dir / "info.txt").write_text(f"vol_cache,{_RUN_NONCE}")
    vol_data = Data(str(data_dir))

    @keras_remote.run(
      accelerator="cpu",
      volumes={"/cached_vol": vol_data},
    )
    def read_vol():
      with open("/cached_vol/info.txt") as f:
        return f.read()

    # First call — cache miss (upload happens)
    r1 = read_vol()

    # Second call — same Data, should hit cache
    logger = logging.getLogger("absl")
    with self.assertLogs(logger, level="INFO") as cm:
      r2 = read_vol()

    log_output = "\n".join(cm.output)
    self.assertIn("Data cache hit", log_output)
    self.assertNotIn("Uploading data", log_output)

    self.assertEqual(r1, r2)
    self.assertIn("vol_cache", r1)

  def test_volume_cache_miss_on_change(self):
    """Changing volume data produces a fresh upload."""
    tmp = _make_test_dir(self)
    data_dir = tmp / "dataset"
    data_dir.mkdir()
    data_file = data_dir / "info.txt"

    def run_read_vol():
      @keras_remote.run(
        accelerator="cpu",
        volumes={"/vol": Data(str(data_dir))},
      )
      def read_vol():
        with open("/vol/info.txt") as f:
          return f.read()

      return read_vol()

    data_file.write_text(f"vol_v1,{_RUN_NONCE}")
    r1 = run_read_vol()
    self.assertIn("vol_v1", r1)

    # Modify volume data — new hash triggers a fresh upload.
    data_file.write_text(f"vol_v2,{_RUN_NONCE}")
    logger = logging.getLogger("absl")
    with self.assertLogs(logger, level="INFO") as cm:
      r2 = run_read_vol()

    log_output = "\n".join(cm.output)
    self.assertIn("Uploading data", log_output)

    self.assertIn("vol_v2", r2)
    self.assertNotEqual(r1, r2)

  def test_multiple_volumes(self):
    """Multiple volumes are each mounted at their declared paths."""
    tmp = _make_test_dir(self)
    d1 = tmp / "data"
    d1.mkdir()
    (d1 / "data.csv").write_text(f"data,{_RUN_NONCE}")
    d2 = tmp / "weights"
    d2.mkdir()
    (d2 / "model.bin").write_text(f"weights,{_RUN_NONCE}")

    @keras_remote.run(
      accelerator="cpu",
      volumes={
        "/data": Data(str(d1)),
        "/weights": Data(str(d2)),
      },
    )
    def check_volumes():

      return {
        "data_files": sorted(os.listdir("/data")),
        "weight_files": sorted(os.listdir("/weights")),
      }

    result = check_volumes()

    self.assertEqual(result["data_files"], ["data.csv"])
    self.assertEqual(result["weight_files"], ["model.bin"])


@skip_unless_e2e()
class TestMixed(absltest.TestCase):
  """Combining volumes, Data args, and plain args in a single call."""

  def test_volumes_plus_data_arg_plus_plain_arg(self):
    """All three data patterns work together in one function."""
    tmp = _make_test_dir(self)
    weights_dir = tmp / "weights"
    weights_dir.mkdir()
    (weights_dir / "model.bin").write_text(f"w,{_RUN_NONCE}")
    config = tmp / "config.json"
    config.write_text(f'{{"lr": 0.01, "nonce": "{_RUN_NONCE}"}}')

    @keras_remote.run(
      accelerator="cpu",
      volumes={"/weights": Data(str(weights_dir))},
    )
    def train(config_path, lr=0.001):
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

    result = train(Data(str(config)), lr=0.05)

    self.assertEqual(result["config"]["lr"], 0.01)
    self.assertEqual(result["lr"], 0.05)
    self.assertTrue(result["has_weights"])
    self.assertEqual(result["weight_files"], ["model.bin"])

  def test_volume_data_also_passed_as_arg(self):
    """Data used as both volume and arg resolves to a path in both cases."""
    tmp = _make_test_dir(self)
    data_dir = tmp / "dataset"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text(f"id,value\n1,{_RUN_NONCE}\n")
    vol_data = Data(str(data_dir))

    @keras_remote.run(
      accelerator="cpu",
      volumes={"/mnt/data": vol_data},
    )
    def read_both(data_path):
      # data_path must resolve to a string, not a dict
      if not isinstance(data_path, str):
        raise TypeError(
          f"Expected str path, got {type(data_path)}: {data_path}"
        )
      # Read from the arg path (downloaded copy)
      with open(f"{data_path}/train.csv") as f:
        arg_content = f.read()
      # Read from the volume mount
      with open("/mnt/data/train.csv") as f:
        vol_content = f.read()
      return {"arg": arg_content, "vol": vol_content}

    result = read_both(vol_data)

    self.assertIn(_RUN_NONCE, result["arg"])
    self.assertIn(_RUN_NONCE, result["vol"])
    self.assertEqual(result["arg"], result["vol"])


@skip_unless_e2e()
class TestNestedData(absltest.TestCase):
  """Data objects inside nested structures (lists, dicts)."""

  def test_data_in_list(self):
    """Data objects in a list arg are each resolved to paths."""
    tmp = _make_test_dir(self)
    d1 = tmp / "set_a"
    d1.mkdir()
    (d1 / "a.csv").write_text(f"a,{_RUN_NONCE}")
    d2 = tmp / "set_b"
    d2.mkdir()
    (d2 / "b.csv").write_text(f"b,{_RUN_NONCE}")

    @keras_remote.run(accelerator="cpu")
    def list_dirs(datasets):

      return [sorted(os.listdir(d)) for d in datasets]

    result = list_dirs(datasets=[Data(str(d1)), Data(str(d2))])

    self.assertEqual(result, [["a.csv"], ["b.csv"]])

  def test_data_in_dict_kwarg(self):
    """Data objects as dict values in kwargs are resolved."""
    tmp = _make_test_dir(self)
    d = tmp / "mydata"
    d.mkdir()
    (d / "x.csv").write_text(f"x,{_RUN_NONCE}")

    @keras_remote.run(accelerator="cpu")
    def read_from_dict(sources):

      return {k: sorted(os.listdir(v)) for k, v in sources.items()}

    result = read_from_dict(sources={"primary": Data(str(d))})

    self.assertEqual(result["primary"], ["x.csv"])


if __name__ == "__main__":
  absltest.main()
