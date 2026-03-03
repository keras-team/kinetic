"""Tests for keras_remote.core.accelerators — parser, registry, categories."""

from absl.testing import absltest, parameterized

from keras_remote.core.accelerators import (
  _GPU_ALIASES,
  GPUS,
  TPUS,
  GpuConfig,
  TpuConfig,
  generate_pool_name,
  get_category,
  parse_accelerator,
)


class TestParseGpuDirect(parameterized.TestCase):
  def test_l4(self):
    result = parse_accelerator("l4")
    self.assertIsInstance(result, GpuConfig)
    self.assertEqual(result.name, "l4")
    self.assertEqual(result.count, 1)
    self.assertEqual(result.gke_label, "nvidia-l4")
    self.assertEqual(result.machine_type, "g2-standard-4")

  @parameterized.parameters(*list(GPUS.keys()))
  def test_all_gpu_types_parse_with_count_1(self, gpu_name):
    result = parse_accelerator(gpu_name)
    self.assertIsInstance(result, GpuConfig)
    self.assertEqual(result.count, 1)
    self.assertEqual(result.name, gpu_name)


class TestParseGpuMultiCount(absltest.TestCase):
  def test_a100x4(self):
    result = parse_accelerator("a100x4")
    self.assertIsInstance(result, GpuConfig)
    self.assertEqual(result.name, "a100")
    self.assertEqual(result.count, 4)

  def test_a100_80gbx4(self):
    result = parse_accelerator("a100-80gbx4")
    self.assertIsInstance(result, GpuConfig)
    self.assertEqual(result.name, "a100-80gb")
    self.assertEqual(result.count, 4)


class TestParseGpuAlias(absltest.TestCase):
  def test_nvidia_tesla_t4(self):
    result = parse_accelerator("nvidia-tesla-t4")
    self.assertIsInstance(result, GpuConfig)
    self.assertEqual(result.name, "t4")
    self.assertEqual(result.count, 1)

  def test_nvidia_tesla_v100x4(self):
    result = parse_accelerator("nvidia-tesla-v100x4")
    self.assertIsInstance(result, GpuConfig)
    self.assertEqual(result.name, "v100")
    self.assertEqual(result.count, 4)


class TestParseGpuErrors(absltest.TestCase):
  def test_l4x8_invalid_count(self):
    with self.assertRaisesRegex(ValueError, "not supported"):
      parse_accelerator("l4x8")


class TestParseTpuBare(parameterized.TestCase):
  def test_v5litepod(self):
    result = parse_accelerator("v5litepod")
    self.assertIsInstance(result, TpuConfig)
    self.assertEqual(result.name, "v5litepod")
    self.assertEqual(result.chips, 4)
    self.assertEqual(result.topology, "2x2")

  @parameterized.parameters(*list(TPUS.keys()))
  def test_all_tpu_types_parse_with_default_chips(self, tpu_name):
    result = parse_accelerator(tpu_name)
    self.assertIsInstance(result, TpuConfig)
    self.assertEqual(result.name, tpu_name)
    self.assertEqual(result.chips, TPUS[tpu_name].default_chips)


class TestParseTpuChipCount(absltest.TestCase):
  def test_v3_4(self):
    result = parse_accelerator("v3-4")
    self.assertIsInstance(result, TpuConfig)
    self.assertEqual(result.name, "v3")
    self.assertEqual(result.chips, 4)
    self.assertEqual(result.topology, "2x2")

  def test_v3_32(self):
    result = parse_accelerator("v3-32")
    self.assertIsInstance(result, TpuConfig)
    self.assertEqual(result.name, "v3")
    self.assertEqual(result.chips, 32)
    self.assertEqual(result.topology, "4x8")

  def test_v5litepod_1(self):
    result = parse_accelerator("v5litepod-1")
    self.assertIsInstance(result, TpuConfig)
    self.assertEqual(result.chips, 1)
    self.assertEqual(result.topology, "1x1")


class TestParseTpuTopology(absltest.TestCase):
  def test_v5litepod_2x2(self):
    result = parse_accelerator("v5litepod-2x2")
    self.assertIsInstance(result, TpuConfig)
    self.assertEqual(result.name, "v5litepod")
    self.assertEqual(result.chips, 4)
    self.assertEqual(result.topology, "2x2")

  def test_v5litepod_1x1(self):
    result = parse_accelerator("v5litepod-1x1")
    self.assertIsInstance(result, TpuConfig)
    self.assertEqual(result.chips, 1)
    self.assertEqual(result.topology, "1x1")


class TestParseTpuErrors(absltest.TestCase):
  def test_v3_8_invalid_chips(self):
    with self.assertRaisesRegex(ValueError, "not supported"):
      parse_accelerator("v3-8")

  def test_v5litepod_3x3_invalid_topology(self):
    with self.assertRaisesRegex(ValueError, "not supported"):
      parse_accelerator("v5litepod-3x3")


class TestParseTpuConfigFields(absltest.TestCase):
  def test_v3_4_full_config(self):
    result = parse_accelerator("v3-4")
    self.assertEqual(result.gke_accelerator, "tpu-v3-podslice")
    self.assertEqual(result.machine_type, "ct3-hightpu-4t")
    self.assertEqual(result.num_nodes, 1)

  def test_v5p_default(self):
    result = parse_accelerator("v5p")
    self.assertIsInstance(result, TpuConfig)
    self.assertEqual(result.chips, 8)
    self.assertEqual(result.topology, "2x2x2")

  def test_v5p_3d_topology(self):
    result = parse_accelerator("v5p-2x2x2")
    self.assertIsInstance(result, TpuConfig)
    self.assertEqual(result.name, "v5p")
    self.assertEqual(result.chips, 8)
    self.assertEqual(result.topology, "2x2x2")
    self.assertEqual(result.num_nodes, 2)


class TestParseCpu(absltest.TestCase):
  def test_cpu(self):
    self.assertIsNone(parse_accelerator("cpu"))


class TestParseNormalizationAndErrors(absltest.TestCase):
  def test_whitespace_and_case(self):
    result = parse_accelerator("  A100X4  ")
    self.assertIsInstance(result, GpuConfig)
    self.assertEqual(result.name, "a100")
    self.assertEqual(result.count, 4)

  def test_empty_string(self):
    with self.assertRaisesRegex(ValueError, "Unknown accelerator"):
      parse_accelerator("")

  def test_unknown_accelerator(self):
    with self.assertRaisesRegex(ValueError, "Unknown accelerator"):
      parse_accelerator("unknown")


class TestGetCategory(absltest.TestCase):
  def test_cpu(self):
    self.assertEqual(get_category("cpu"), "cpu")

  def test_gpu(self):
    self.assertEqual(get_category("l4"), "gpu")

  def test_tpu(self):
    self.assertEqual(get_category("v5litepod"), "tpu")


class TestGeneratePoolName(absltest.TestCase):
  def test_gpu_prefix(self):
    gpu = GpuConfig("l4", 1, "nvidia-l4", "g2-standard-4")
    name = generate_pool_name(gpu)
    self.assertTrue(name.startswith("gpu-l4-"), name)

  def test_tpu_prefix(self):
    tpu = TpuConfig("v5p", 8, "2x2x2", "tpu-v5p-slice", "ct5p-hightpu-4t", 2)
    name = generate_pool_name(tpu)
    self.assertTrue(name.startswith("tpu-v5p-"), name)

  def test_suffix_is_4_hex_chars(self):
    gpu = GpuConfig("l4", 1, "nvidia-l4", "g2-standard-4")
    name = generate_pool_name(gpu)
    suffix = name.split("-")[-1]
    self.assertLen(suffix, 4)
    # Verify it's valid hex.
    int(suffix, 16)

  def test_unique_across_calls(self):
    gpu = GpuConfig("l4", 1, "nvidia-l4", "g2-standard-4")
    names = {generate_pool_name(gpu) for _ in range(50)}
    self.assertGreater(len(names), 1)


class TestRegistryIntegrity(absltest.TestCase):
  def test_all_gpus_have_nonempty_counts(self):
    for name, spec in GPUS.items():
      self.assertNotEmpty(spec.counts, f"GPU '{name}' has empty counts")

  def test_all_tpus_have_nonempty_topologies(self):
    for name, spec in TPUS.items():
      self.assertNotEmpty(spec.topologies, f"TPU '{name}' has empty topologies")

  def test_all_tpu_default_chips_valid(self):
    for name, spec in TPUS.items():
      self.assertIn(
        spec.default_chips,
        spec.topologies,
        f"TPU '{name}' default_chips={spec.default_chips} "
        f"not in topologies {list(spec.topologies.keys())}",
      )

  def test_all_gpus_have_gke_label_alias(self):
    for name, spec in GPUS.items():
      self.assertIn(
        spec.gke_label,
        _GPU_ALIASES,
        f"GPU '{name}' gke_label '{spec.gke_label}' not in aliases",
      )
      self.assertEqual(_GPU_ALIASES[spec.gke_label], name)


if __name__ == "__main__":
  absltest.main()
