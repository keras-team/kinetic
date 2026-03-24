"""Typed configuration for kinetic CLI infrastructure commands."""

from dataclasses import dataclass, field
from typing import Union

from kinetic.cli.constants import DEFAULT_CLUSTER_NAME, DEFAULT_ZONE
from kinetic.core.accelerators import GpuConfig, TpuConfig


@dataclass
class NodePoolConfig:
  """A named accelerator node pool."""

  name: str  # GKE node pool name, e.g. "gpu-l4-a3f2"
  accelerator: Union[GpuConfig, TpuConfig]
  min_nodes: int = 0


@dataclass
class InfraConfig:
  """Configuration for infrastructure provisioning/teardown."""

  project: str
  zone: str = DEFAULT_ZONE
  cluster_name: str = DEFAULT_CLUSTER_NAME
  node_pools: list[NodePoolConfig] = field(default_factory=list)
