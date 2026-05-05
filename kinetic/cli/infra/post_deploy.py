"""Post-deploy steps that cannot be managed by Pulumi.

Configures local machine state (kubectl context) after the cluster is ready.
Kubernetes resources (KSA, LWS CRD, GPU drivers) are managed declaratively
by the Pulumi program.
"""

import os
import subprocess

from kinetic.credentials import invalidate_credential_cache


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
      "--",
      cluster_name,
      f"--zone={zone}",
      f"--project={project}",
    ],
    check=True,
    capture_output=True,
    env=env,
  )
  # Kubeconfig changed — invalidate so ensure_credentials() re-validates.
  invalidate_credential_cache(project, zone, cluster_name)
