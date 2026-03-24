"""Post-deploy steps that cannot be managed by Pulumi.

These operations configure local machine state (kubectl) or apply
Kubernetes manifests that depend on the cluster being ready.
"""

import os
import subprocess

from kinetic.cli.constants import (
  LWS_INSTALL_URL,
  NVIDIA_DRIVER_DAEMONSET_URL,
)


def configure_kubectl(cluster_name, zone, project):
  """Configure kubectl to access the GKE cluster.

  Args:
      cluster_name: GKE cluster name.
      zone: GCP zone.
      project: GCP project ID.
  """
  env = {**os.environ, "USE_GKE_GCLOUD_AUTH_PLUGIN": "True"}
  subprocess.run(
    [
      "gcloud",
      "container",
      "clusters",
      "get-credentials",
      cluster_name,
      f"--zone={zone}",
      f"--project={project}",
    ],
    check=True,
    capture_output=True,
    env=env,
  )


def install_gpu_drivers():
  """Install NVIDIA GPU device drivers on GKE GPU nodes.

  Applies the Google-maintained DaemonSet that installs GPU drivers
  on Container-Optimized OS nodes.
  """
  subprocess.run(
    ["kubectl", "apply", "-f", NVIDIA_DRIVER_DAEMONSET_URL],
    check=True,
    capture_output=True,
  )


def install_lws():
  """Install the LeaderWorkerSet custom resource controller.

  This enables Pathways scheduling on the GKE cluster.
  """
  subprocess.run(
    ["kubectl", "apply", "--server-side", "-f", LWS_INSTALL_URL],
    check=True,
    capture_output=True,
  )
