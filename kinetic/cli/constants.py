"""Default values and constants for the kinetic CLI."""

import os

from kinetic.constants import (
  DEFAULT_CLUSTER_NAME,  # noqa: F401 — re-exported
  DEFAULT_ZONE,  # noqa: F401 — re-exported
)

RESOURCE_NAME_PREFIX = "kinetic"
# Kubernetes service account used by kinetic workload pods.
# Bound to the node GCP SA via Workload Identity Federation.
KINETIC_KSA_NAME = "kinetic"
STATE_DIR = os.environ.get(
  "KINETIC_STATE_DIR",
  os.path.expanduser("~/.kinetic/pulumi"),
)
PULUMI_ROOT = os.path.expanduser("~/.kinetic/pulumi-cli")
PROFILES_FILE = os.path.expanduser("~/.kinetic/profiles.json")
REQUIRED_APIS = [
  "compute.googleapis.com",
  "cloudbuild.googleapis.com",
  "artifactregistry.googleapis.com",
  "storage.googleapis.com",
  "container.googleapis.com",
  "secretmanager.googleapis.com",
  "iam.googleapis.com",
]

NVIDIA_DRIVER_DAEMONSET_URL = (
  "https://raw.githubusercontent.com/GoogleCloudPlatform/"
  "container-engine-accelerators/v1.0.20/"
  "nvidia-driver-installer/cos/daemonset-preloaded.yaml"
)

LWS_INSTALL_URL = "https://github.com/kubernetes-sigs/lws/releases/download/v0.5.1/manifests.yaml"

# Autoscaling upper bounds
MAX_CLUSTER_CPU = 1000
MAX_CLUSTER_MEMORY_GB = 64000
NODE_MAX_RUN_DURATION_SECONDS = 24 * 60 * 60  # 24 hours
GPU_NODE_POOL_MAX_SCALE_UP = 10
