"""Accelerator registry and parsing for keras-remote.

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
  machine_type: str
  counts: tuple[int, ...]


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
  "l4": GpuSpec("nvidia-l4", "g2-standard-4", (1, 2, 4)),
  "t4": GpuSpec("nvidia-tesla-t4", "n1-standard-4", (1, 2, 4)),
  "v100": GpuSpec("nvidia-tesla-v100", "n1-standard-8", (1, 2, 4, 8)),
  "a100": GpuSpec("nvidia-tesla-a100", "a2-highgpu-1g", (1, 2, 4, 8)),
  "a100-80gb": GpuSpec("nvidia-a100-80gb", "a2-ultragpu-1g", (1, 2, 4, 8)),
  "h100": GpuSpec("nvidia-h100-80gb", "a3-highgpu-1g", (1, 2, 4, 8)),
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
  "v2": TpuSpec(
    "tpu-v2-podslice",
    4,
    {
      4: TpuTopologySpec("2x2", "ct2-hightpu-4t", 1),
      16: TpuTopologySpec("4x4", "ct2-hightpu-4t", 4),
      32: TpuTopologySpec("4x8", "ct2-hightpu-4t", 8),
    },
  ),
  "v3": TpuSpec(
    "tpu-v3-podslice",
    4,
    {
      4: TpuTopologySpec("2x2", "ct3-hightpu-4t", 1),
      16: TpuTopologySpec("4x4", "ct3p-hightpu-4t", 4),
      32: TpuTopologySpec("4x8", "ct3p-hightpu-4t", 8),
    },
  ),
  "v5litepod": TpuSpec(
    "tpu-v5-lite-podslice",
    4,
    {
      1: TpuTopologySpec("1x1", "ct5lp-hightpu-1t", 1),
      4: TpuTopologySpec("2x2", "ct5lp-hightpu-4t", 1),
      8: TpuTopologySpec("2x4", "ct5lp-hightpu-8t", 1),
    },
  ),
  "v5p": TpuSpec(
    "tpu-v5p-slice",
    8,
    {
      8: TpuTopologySpec("2x2x2", "ct5p-hightpu-4t", 2),
      16: TpuTopologySpec("2x2x4", "ct5p-hightpu-4t", 4),
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


# ── Parser ────────────────────────────────────────────────────────

_MULTI_GPU_RE = re.compile(r"^(.+?)x(\d+)$")  # "a100x4"
_TPU_CHIPS_RE = re.compile(r"^(v\d+\w*)-(\d+)$")  # "v3-8"
_TPU_TOPO_RE = re.compile(
  r"^(v\d+\w*)-(\d+x\d+(?:x\d+)?)$"
)  # "v5litepod-2x2", "v5p-2x2x2"


def parse_accelerator(accel_str: str) -> Accelerator:
  """Parse an accelerator string into a fully resolved config.

  Returns GpuConfig, TpuConfig, or None (for "cpu").

  Accepted formats:
      GPU:  "l4", "nvidia-l4", "a100x4", "a100-80gbx8"
      TPU:  "v3-8" (chip count), "v5litepod-2x2" (topology), "v5litepod" (default)
      CPU:  "cpu"
  """
  s = accel_str.strip().lower()

  if s == "cpu":
    return None

  # Direct GPU name: "l4", "a100-80gb"
  if s in GPUS:
    return make_gpu(s, 1)

  # GPU alias: "nvidia-l4"
  if s in _GPU_ALIASES:
    return make_gpu(_GPU_ALIASES[s], 1)

  # Multi-GPU: "a100x4", "l4x2"
  m = _MULTI_GPU_RE.match(s)
  if m:
    name = m.group(1)
    if name in GPUS:
      return make_gpu(name, int(m.group(2)))
    if name in _GPU_ALIASES:
      return make_gpu(_GPU_ALIASES[name], int(m.group(2)))

  # Direct TPU name (bare): "v5litepod" → default chips
  if s in TPUS:
    return make_tpu(s, TPUS[s].default_chips)

  # TPU with topology string: "v5litepod-2x2", "v5p-2x2x2"
  m = _TPU_TOPO_RE.match(s)
  if m and m.group(1) in TPUS:
    name = m.group(1)
    topo_str = m.group(2)
    for chips, topo_spec in TPUS[name].topologies.items():
      if topo_spec.topology == topo_str:
        return make_tpu(name, chips)
    valid = [ts.topology for ts in TPUS[name].topologies.values()]
    raise ValueError(
      f"Topology '{topo_str}' not supported for '{name}'. "
      f"Supported: {', '.join(valid)}."
    )

  # TPU with chip count: "v3-8", "v5litepod-4"
  m = _TPU_CHIPS_RE.match(s)
  if m and m.group(1) in TPUS:
    return make_tpu(m.group(1), int(m.group(2)))

  raise ValueError(
    f"Unknown accelerator: '{accel_str}'. "
    f"GPUs: {', '.join(GPUS)} (use 'xN' for multi-GPU, e.g. 'a100x4'). "
    f"TPUs: {', '.join(TPUS)} (use '-N' for chips, e.g. 'v3-8', "
    f"or '-NxM' for topology, e.g. 'v5litepod-2x2')."
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
    machine_type=spec.machine_type,
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
