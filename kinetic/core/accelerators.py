"""Accelerator registry and parsing for kinetic.

Single source of truth for all accelerator metadata — used by both the
runtime (gke_client, container_builder) and the CLI (up, prompts, program).
"""

import re
import uuid
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class GpuConfig:
  """Fully resolved GPU accelerator configuration."""

  name: str  # "l4"
  count: int  # number of GPUs (1, 2, 4, …)
  gke_label: str  # "nvidia-l4" — K8s node selector value
  machine_type: str  # "g2-standard-4" — GKE node pool machine type


@dataclass(frozen=True)
class TpuConfig:
  """Fully resolved TPU accelerator configuration."""

  name: str  # "v5litepod"
  chips: int  # number of TPU chips (4, 8, …)
  topology: str  # "2x2" — TPU topology string
  gke_accelerator: str  # "tpu-v5-lite-podslice"
  machine_type: str  # "ct5lp-hightpu-4t"
  num_nodes: int  # GKE node pool node count


Accelerator = Union[GpuConfig, TpuConfig, None]


@dataclass(frozen=True)
class GpuSpec:
  """Registry entry for a GPU type."""

  gke_label: str
  counts: dict[int, str]  # count -> machine_type


@dataclass(frozen=True)
class TpuTopologySpec:
  """Single topology option for a TPU type."""

  topology: str
  machine_type: str
  num_nodes: int


@dataclass(frozen=True)
class TpuSpec:
  """Registry entry for a TPU type."""

  gke_accelerator: str
  default_chips: int
  topologies: dict[int, TpuTopologySpec]  # chips → topology spec


GPUS: dict[str, GpuSpec] = {
  "l4": GpuSpec(
    "nvidia-l4",
    {
      1: "g2-standard-4",
      2: "g2-standard-24",
      4: "g2-standard-48",
      8: "g2-standard-96",
    },
  ),
  "t4": GpuSpec(
    "nvidia-tesla-t4",
    {
      1: "n1-standard-4",
      2: "n1-standard-8",
      4: "n1-standard-16",
    },
  ),
  "v100": GpuSpec(
    "nvidia-tesla-v100",
    {
      1: "n1-standard-8",
      2: "n1-standard-16",
      4: "n1-standard-32",
      8: "n1-standard-64",
    },
  ),
  "a100": GpuSpec(
    "nvidia-tesla-a100",
    {
      1: "a2-highgpu-1g",
      2: "a2-highgpu-2g",
      4: "a2-highgpu-4g",
      8: "a2-highgpu-8g",
      16: "a2-megagpu-16g",
    },
  ),
  "a100-80gb": GpuSpec(
    "nvidia-a100-80gb",
    {
      1: "a2-ultragpu-1g",
      2: "a2-ultragpu-2g",
      4: "a2-ultragpu-4g",
      8: "a2-ultragpu-8g",
      16: "a2-ultragpu-16g",
    },
  ),
  "h100": GpuSpec(
    "nvidia-h100-80gb",
    {
      1: "a3-highgpu-1g",
      2: "a3-highgpu-2g",
      4: "a3-highgpu-4g",
      8: "a3-highgpu-8g",
    },
  ),
  "p4": GpuSpec(
    "nvidia-tesla-p4",
    {
      1: "n1-standard-4",
      2: "n1-standard-8",
      4: "n1-standard-16",
    },
  ),
  "p100": GpuSpec(
    "nvidia-tesla-p100",
    {
      1: "n1-standard-4",
      2: "n1-standard-8",
      4: "n1-standard-16",
    },
  ),
}

_GPU_ALIASES: dict[str, str] = {
  spec.gke_label: name for name, spec in GPUS.items()
}

# Topology reference — verify new entries against:
#   https://docs.cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus
# Formula: num_nodes = product(topology_dims) / chips_per_VM
# Machine-type suffix "-Nt" → N chips per VM (e.g. ct5p-hightpu-4t → 4 chips).
# v5p uses 3-D topologies (AxBxC); v2, v3, v5litepod, v6e use 2-D (AxB).
TPUS: dict[str, TpuSpec] = {
  "v3": TpuSpec(
    "tpu-v3-podslice",
    4,
    {
      4: TpuTopologySpec("2x2", "ct3-hightpu-4t", 1),
      16: TpuTopologySpec("4x4", "ct3p-hightpu-4t", 4),
      32: TpuTopologySpec("4x8", "ct3p-hightpu-4t", 8),
      64: TpuTopologySpec("8x8", "ct3p-hightpu-4t", 16),
      128: TpuTopologySpec("8x16", "ct3p-hightpu-4t", 32),
      256: TpuTopologySpec("16x16", "ct3p-hightpu-4t", 64),
      512: TpuTopologySpec("16x32", "ct3p-hightpu-4t", 128),
      1024: TpuTopologySpec("32x32", "ct3p-hightpu-4t", 256),
      2048: TpuTopologySpec("32x64", "ct3p-hightpu-4t", 512),
    },
  ),
  "v4": TpuSpec(
    "tpu-v4-podslice",
    4,
    {
      4: TpuTopologySpec("2x2x1", "ct4p-hightpu-4t", 1),
      8: TpuTopologySpec("2x2x2", "ct4p-hightpu-4t", 2),
      16: TpuTopologySpec("2x2x4", "ct4p-hightpu-4t", 4),
      32: TpuTopologySpec("2x4x4", "ct4p-hightpu-4t", 8),
      64: TpuTopologySpec("4x4x4", "ct4p-hightpu-4t", 16),
      128: TpuTopologySpec("4x4x8", "ct4p-hightpu-4t", 32),
      256: TpuTopologySpec("4x8x8", "ct4p-hightpu-4t", 64),
      512: TpuTopologySpec("8x8x8", "ct4p-hightpu-4t", 128),
      1024: TpuTopologySpec("8x8x16", "ct4p-hightpu-4t", 256),
      2048: TpuTopologySpec("8x16x16", "ct4p-hightpu-4t", 512),
      4096: TpuTopologySpec("16x16x16", "ct4p-hightpu-4t", 1024),
    },
  ),
  "v5litepod": TpuSpec(
    "tpu-v5-lite-podslice",
    4,
    {
      1: TpuTopologySpec("1x1", "ct5lp-hightpu-1t", 1),
      4: TpuTopologySpec("2x2", "ct5lp-hightpu-4t", 1),
      8: TpuTopologySpec("2x4", "ct5lp-hightpu-8t", 1),
      16: TpuTopologySpec("4x4", "ct5lp-hightpu-4t", 4),
      32: TpuTopologySpec("4x8", "ct5lp-hightpu-4t", 8),
      64: TpuTopologySpec("8x8", "ct5lp-hightpu-4t", 16),
      128: TpuTopologySpec("8x16", "ct5lp-hightpu-4t", 32),
      256: TpuTopologySpec("16x16", "ct5lp-hightpu-4t", 64),
    },
  ),
  "v5p": TpuSpec(
    "tpu-v5p-slice",
    8,
    {
      8: TpuTopologySpec("2x2x2", "ct5p-hightpu-4t", 2),
      16: TpuTopologySpec("2x2x4", "ct5p-hightpu-4t", 4),
      32: TpuTopologySpec("2x4x4", "ct5p-hightpu-4t", 8),
    },
  ),
  "v6e": TpuSpec(
    "tpu-v6e-slice",
    8,
    {
      8: TpuTopologySpec("2x4", "ct6e-standard-4t", 2),
      16: TpuTopologySpec("4x4", "ct6e-standard-4t", 4),
    },
  ),
}

_TPU_ALIASES: dict[str, str] = {
  "v5e": "v5litepod",
}


_MULTI_GPU_RE = re.compile(r"^([^x]+)(?:x)(\d+)$")  # "a100x4"
_TPU_CHIPS_RE = re.compile(r"^([a-z0-9_]+)-(\d+)$")  # "v3-8"
_TPU_TOPO_RE = re.compile(
  r"^([a-z0-9_]+)-(\d+x\d+(?:x\d+)?)$"
)  # "v5litepod-2x2"

DEFAULT_GPU = "l4"
DEFAULT_TPU = "v5litepod"

_PREFERRED_GPUS = [
  "h100",
  "a100-80gb",
  "a100",
  "l4",
  "v100",
  "t4",
  "p100",
  "p4",
]
_PREFERRED_TPUS = ["v6e", "v5p", "v5litepod", "v4", "v3"]


def _resolve_gpu_alias(name: str) -> str:
  return _GPU_ALIASES.get(name, name)


def _resolve_tpu_alias(name: str) -> str:
  return _TPU_ALIASES.get(name, name)


def parse_accelerator(accel_str: str) -> Accelerator:
  """Parse an accelerator string into a fully resolved config.

  Returns GpuConfig, TpuConfig, or None (for "cpu").

  Accepted formats:
      - Generic: "gpu", "tpu", "cpu" (resolves to defaults)
      - Dynamic Count: "gpu:4", "tpu:8", "cpu:8" (assigns most capable hardware matching the count)
      - Explicit GPU Name: "gpu:l4", "l4", "gpu:a100-80gb" (resolves to 1 instance of the specified GPU)
      - Multi-GPU Name: "gpu:a100x4", "a100x4", "gpu:l4-2" (resolves to N instances of the specified GPU)
      - Explicit TPU Name: "tpu:v5litepod", "v5litepod" (resolves to the default topology/chips for the TPU)
      - Explicit TPU Topology/Chips: "tpu:v3-8", "tpu:v5litepod-2x2", "v3-8" (resolves to the specified TPU slice)

  Note: Prefixes ('gpu:' and 'tpu:') are recommended for complete disambiguation but are completely optional.

  Dynamic Resolution:
      When using generic formats like "gpu:<N>" or "tpu:<N>", the parser
      dynamically assigns the most capable hardware type that supports the
      requested device count `N`. Hardware is selected based on an internal
      preference hierarchy (e.g., H100 > A100 > L4 for GPUs, and
      v6e > v5p > v5litepod for TPUs).
  """
  s = accel_str.strip().lower()

  if s == "cpu" or (s.startswith("cpu:") and s[4:].isdigit()):
    return None

  if s == "gpu":
    return make_gpu(DEFAULT_GPU, 1)

  if s == "tpu":
    return make_tpu(DEFAULT_TPU, TPUS[DEFAULT_TPU].default_chips)

  # 1) Try parsing as GPU
  is_gpu_explicit = s.startswith("gpu:")
  gpu_str = s[4:] if is_gpu_explicit else s

  if gpu_str.isdigit():
    count = int(gpu_str)
    for gpu_name in _PREFERRED_GPUS:
      if gpu_name in GPUS and count in GPUS[gpu_name].counts:
        return make_gpu(gpu_name, count)
    if is_gpu_explicit:
      valid_counts = sorted(
        set(c for spec in GPUS.values() for c in spec.counts)
      )
      raise ValueError(
        f"No GPU supports count {count}. Supported counts: {valid_counts}"
      )

  name = _resolve_gpu_alias(gpu_str)
  if name in GPUS:
    return make_gpu(name, 1)

  m = _MULTI_GPU_RE.match(gpu_str)
  if m:
    name = _resolve_gpu_alias(m.group(1))
    if name in GPUS:
      return make_gpu(name, int(m.group(2)))

  if is_gpu_explicit:
    raise ValueError(f"Unknown GPU accelerator: '{accel_str}'")

  # 2) Try parsing as TPU
  is_tpu_explicit = s.startswith("tpu:")
  tpu_str = s[4:] if is_tpu_explicit else s

  if tpu_str.isdigit():
    chips = int(tpu_str)
    for tpu_name in _PREFERRED_TPUS:
      if tpu_name in TPUS and chips in TPUS[tpu_name].topologies:
        return make_tpu(tpu_name, chips)
    if is_tpu_explicit:
      valid_chips = sorted(
        set(c for spec in TPUS.values() for c in spec.topologies)
      )
      raise ValueError(
        f"No TPU supports {chips} chips. Supported chip counts: {valid_chips}"
      )

  name = _resolve_tpu_alias(tpu_str)
  if name in TPUS:
    return make_tpu(name, TPUS[name].default_chips)

  m = _TPU_TOPO_RE.match(tpu_str)
  if m:
    name = _resolve_tpu_alias(m.group(1))
    if name in TPUS:
      topo_str = m.group(2)
      for chips, topo_spec in TPUS[name].topologies.items():
        if topo_spec.topology == topo_str:
          return make_tpu(name, chips)
      valid = [ts.topology for ts in TPUS[name].topologies.values()]
      raise ValueError(
        f"Topology '{topo_str}' not supported for '{name}'. "
        f"Supported: {', '.join(valid)}."
      )

  m = _TPU_CHIPS_RE.match(tpu_str)
  if m:
    name = _resolve_tpu_alias(m.group(1))
    if name in TPUS:
      return make_tpu(name, int(m.group(2)))

  raise ValueError(
    f"Unknown accelerator: '{accel_str}'. "
    f"GPUs: {', '.join(GPUS)} (use 'gpu:name' or 'gpu:namexN'). "
    f"TPUs: {', '.join(TPUS)} (use 'tpu:name' or 'tpu:name-N')."
  )


def get_category(accel_str: str) -> str:
  """Return 'cpu', 'gpu', or 'tpu' for the given accelerator string."""
  result = parse_accelerator(accel_str)
  if result is None:
    return "cpu"
  if isinstance(result, GpuConfig):
    return "gpu"
  return "tpu"


def generate_pool_name(accel: GpuConfig | TpuConfig) -> str:
  """Generate a unique GKE node pool name for an accelerator config.

  Format: ``gpu-{name}-{hex4}`` or ``tpu-{name}-{hex4}`` where *hex4*
  is a random 4-character hexadecimal suffix.
  """
  suffix = uuid.uuid4().hex[:4]
  if isinstance(accel, GpuConfig):
    return f"gpu-{accel.name}-{suffix}"
  if isinstance(accel, TpuConfig):
    return f"tpu-{accel.name}-{suffix}"
  raise TypeError(f"Expected GpuConfig or TpuConfig, got {type(accel)}")


def make_gpu(name: str, count: int) -> GpuConfig:
  spec = GPUS[name]
  if count not in spec.counts:
    raise ValueError(
      f"GPU count {count} not supported for '{name}'. "
      f"Supported: {', '.join(str(c) for c in spec.counts)}."
    )
  return GpuConfig(
    name=name,
    count=count,
    gke_label=spec.gke_label,
    machine_type=spec.counts[count],
  )


def make_tpu(name: str, chips: int) -> TpuConfig:
  spec = TPUS[name]
  if chips not in spec.topologies:
    raise ValueError(
      f"Chip count {chips} not supported for '{name}'. "
      f"Supported: {', '.join(str(c) for c in spec.topologies)}."
    )
  topo_spec = spec.topologies[chips]
  return TpuConfig(
    name=name,
    chips=chips,
    topology=topo_spec.topology,
    gke_accelerator=spec.gke_accelerator,
    machine_type=topo_spec.machine_type,
    num_nodes=topo_spec.num_nodes,
  )
