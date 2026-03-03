"""Default values and constants for the keras-remote CLI."""

import os

from keras_remote.constants import (
  DEFAULT_CLUSTER_NAME,  # noqa: F401 — re-exported
  DEFAULT_ZONE,  # noqa: F401 — re-exported
)

RESOURCE_NAME_PREFIX = "keras-remote"
STATE_DIR = os.environ.get(
  "KERAS_REMOTE_STATE_DIR",
  os.path.expanduser("~/.keras-remote/pulumi"),
)
PULUMI_ROOT = os.path.expanduser("~/.keras-remote/pulumi-cli")
REQUIRED_APIS = [
  "compute.googleapis.com",
  "cloudbuild.googleapis.com",
  "artifactregistry.googleapis.com",
  "storage.googleapis.com",
  "container.googleapis.com",
]

NVIDIA_DRIVER_DAEMONSET_URL = (
  "https://raw.githubusercontent.com/GoogleCloudPlatform/"
  "container-engine-accelerators/v1.0.20/"
  "nvidia-driver-installer/cos/daemonset-preloaded.yaml"
)

LWS_INSTALL_URL = "https://github.com/kubernetes-sigs/lws/releases/download/v0.5.1/manifests.yaml"
