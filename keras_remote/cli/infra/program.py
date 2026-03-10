"""Pulumi inline program for keras-remote infrastructure.

Defines all GCP resources needed for keras-remote: API services,
Artifact Registry, GKE cluster, and optional accelerator node pools.
"""

import pulumi
import pulumi_gcp as gcp

from keras_remote.cli.constants import (
  MAX_CLUSTER_CPU,
  MAX_CLUSTER_MEMORY_GB,
  NODE_MAX_RUN_DURATION_SECONDS,
  REQUIRED_APIS,
  RESOURCE_NAME_PREFIX,
)
from keras_remote.constants import zone_to_ar_location, zone_to_region
from keras_remote.core.accelerators import GpuConfig, TpuConfig

# OAuth scopes required by all node pools (including accelerator pools).
_BASE_OAUTH_SCOPES = [
  # Read/write access to GCS for storing checkpoints, datasets, and logs.
  "https://www.googleapis.com/auth/devstorage.read_write",
  # Write application logs to Cloud Logging.
  "https://www.googleapis.com/auth/logging.write",
  # Export metrics to Cloud Monitoring.
  "https://www.googleapis.com/auth/monitoring",
]

# Additional scopes for the default (system) node pool, which runs GKE
# control-plane components that need deeper platform integration.
_DEFAULT_POOL_OAUTH_SCOPES = _BASE_OAUTH_SCOPES + [
  # Report service status to Google Service Control.
  "https://www.googleapis.com/auth/servicecontrol",
  # Read managed-service configuration from Service Management.
  "https://www.googleapis.com/auth/service.management.readonly",
  # Send distributed traces to Cloud Trace.
  "https://www.googleapis.com/auth/trace.append",
]


def create_program(config):
  """Create a Pulumi inline program function closed over the config.

  Args:
      config: InfraConfig instance.

  Returns:
      A callable suitable for pulumi.automation.create_or_select_stack().
  """

  def pulumi_program():
    project_id = config.project
    zone = config.zone
    ar_location = zone_to_ar_location(zone)
    cluster_name = config.cluster_name
    node_pools = config.node_pools

    # 1. Enable GCP APIs
    enabled_apis = []
    for api in REQUIRED_APIS:
      svc = gcp.projects.Service(
        f"api-{api.split('.')[0]}",
        service=api,
        project=project_id,
        disable_on_destroy=False,
        disable_dependent_services=False,
      )
      enabled_apis.append(svc)

    # 2. Artifact Registry docker repository
    repo = gcp.artifactregistry.Repository(
      "keras-remote-repo",
      repository_id=f"kr-{cluster_name}",
      location=ar_location,
      format="DOCKER",
      description="keras-remote container images",
      project=project_id,
      opts=pulumi.ResourceOptions(depends_on=enabled_apis),
    )

    # 3. Cloud Storage buckets
    region = zone_to_region(zone)

    gcp.storage.Bucket(
      "keras-remote-jobs-bucket",
      name=f"{project_id}-kr-{cluster_name}-jobs",
      location=region,
      project=project_id,
      force_destroy=True,
      opts=pulumi.ResourceOptions(depends_on=enabled_apis),
    )

    gcp.storage.Bucket(
      "keras-remote-builds-bucket",
      name=f"{project_id}-kr-{cluster_name}-builds",
      location=ar_location,
      project=project_id,
      force_destroy=True,
      opts=pulumi.ResourceOptions(depends_on=enabled_apis),
    )

    # 4. GKE Cluster
    cluster = gcp.container.Cluster(
      "keras-remote-cluster",
      name=cluster_name,
      location=zone,
      project=project_id,
      initial_node_count=1,
      remove_default_node_pool=False,
      node_config=gcp.container.ClusterNodeConfigArgs(
        machine_type="e2-standard-4",
        disk_size_gb=50,
        oauth_scopes=_DEFAULT_POOL_OAUTH_SCOPES,
      ),
      # Match setup.sh: --no-enable-autoupgrade
      release_channel=gcp.container.ClusterReleaseChannelArgs(
        channel="UNSPECIFIED",
      ),
      deletion_protection=False,
      cluster_autoscaling=gcp.container.ClusterClusterAutoscalingArgs(
        enabled=True,
        autoscaling_profile="OPTIMIZE_UTILIZATION",
        auto_provisioning_defaults=gcp.container.ClusterClusterAutoscalingAutoProvisioningDefaultsArgs(
          oauth_scopes=_DEFAULT_POOL_OAUTH_SCOPES,
          management=gcp.container.ClusterClusterAutoscalingAutoProvisioningDefaultsManagementArgs(
            auto_upgrade=True,
            auto_repair=True,
          ),
        ),
        resource_limits=[
          gcp.container.ClusterClusterAutoscalingResourceLimitArgs(
            resource_type="cpu",
            maximum=MAX_CLUSTER_CPU,
          ),
          gcp.container.ClusterClusterAutoscalingResourceLimitArgs(
            resource_type="memory",
            maximum=MAX_CLUSTER_MEMORY_GB,
          ),
        ],
      ),
      opts=pulumi.ResourceOptions(depends_on=enabled_apis),
    )

    # 5. Accelerator node pools (zero or more)
    pool_entries = []
    for np in node_pools:
      accel = np.accelerator
      pool_name = np.name
      if isinstance(accel, GpuConfig):
        pool = _create_gpu_node_pool(
          cluster, accel, zone, project_id, pool_name
        )
      elif isinstance(accel, TpuConfig):
        pool = _create_tpu_node_pool(
          cluster, accel, zone, project_id, pool_name
        )
      else:
        continue
      pool_entries.append((accel, pool))

    # 6. Stack exports
    # Exports that reference resource outputs (e.g. cluster.name,
    # repo.name, pool.name) create Pulumi dependencies — the export
    # only resolves when the underlying resource is successfully created.
    pulumi.export("project", project_id)
    pulumi.export("zone", zone)
    pulumi.export("cluster_name", cluster.name)
    pulumi.export("cluster_endpoint", cluster.endpoint)
    pulumi.export(
      "ar_registry",
      repo.name.apply(
        lambda _: f"{ar_location}-docker.pkg.dev/{project_id}/kr-{cluster_name}"
      ),
    )

    # 7. Accelerator node pool exports (list of dicts)
    if not pool_entries:
      pulumi.export("accelerators", [])
    else:
      export_outputs = []
      for accel, pool in pool_entries:
        if isinstance(accel, GpuConfig):
          entry = pool.name.apply(
            lambda pn, a=accel: {
              "type": "GPU",
              "name": a.name,
              "count": a.count,
              "machine_type": a.machine_type,
              "node_pool": pn,
              "node_count": 1,
            }
          )
        else:  # TpuConfig
          entry = pool.name.apply(
            lambda pn, a=accel: {
              "type": "TPU",
              "name": a.name,
              "chips": a.chips,
              "topology": a.topology,
              "machine_type": a.machine_type,
              "node_pool": pn,
              "node_count": a.num_nodes,
            }
          )
        export_outputs.append(entry)
      pulumi.export("accelerators", pulumi.Output.all(*export_outputs))

  return pulumi_program


def _create_gpu_node_pool(cluster, gpu: GpuConfig, zone, project_id, pool_name):
  """Create a GPU-accelerated GKE node pool."""
  return gcp.container.NodePool(
    pool_name,
    name=pool_name,
    cluster=cluster.name,
    location=zone,
    project=project_id,
    initial_node_count=0,
    autoscaling=gcp.container.NodePoolAutoscalingArgs(
      min_node_count=0,
      max_node_count=10,
    ),
    management=gcp.container.NodePoolManagementArgs(
      auto_repair=True,
      auto_upgrade=True,
    ),
    node_config=gcp.container.NodePoolNodeConfigArgs(
      machine_type=gpu.machine_type,
      oauth_scopes=_BASE_OAUTH_SCOPES,
      guest_accelerators=[
        gcp.container.NodePoolNodeConfigGuestAcceleratorArgs(
          type=gpu.gke_label,
          count=gpu.count,
        ),
      ],
      labels={RESOURCE_NAME_PREFIX: "true"},
      max_run_duration=f"{NODE_MAX_RUN_DURATION_SECONDS}s",  # 24 hours
    ),
  )


def _create_tpu_node_pool(cluster, tpu: TpuConfig, zone, project_id, pool_name):
  """Create a TPU GKE node pool."""
  # Single-host TPU slices (1 node) must not specify placement_policy;
  # multi-host slices require COMPACT placement with an explicit topology.
  is_multi_host = tpu.num_nodes > 1
  # Autoscaling is enabled, so we need to set the min_node_count to 0.
  min_nodes = 0

  placement = (
    gcp.container.NodePoolPlacementPolicyArgs(
      type="COMPACT",
      tpu_topology=tpu.topology,
    )
    if is_multi_host
    else None
  )
  return gcp.container.NodePool(
    pool_name,
    name=pool_name,
    cluster=cluster.name,
    location=zone,
    project=project_id,
    initial_node_count=min_nodes,
    autoscaling=gcp.container.NodePoolAutoscalingArgs(
      min_node_count=min_nodes,
      max_node_count=tpu.num_nodes,
    ),
    management=gcp.container.NodePoolManagementArgs(
      auto_repair=True,
      auto_upgrade=True,
    ),
    node_config=gcp.container.NodePoolNodeConfigArgs(
      machine_type=tpu.machine_type,
      oauth_scopes=_BASE_OAUTH_SCOPES,
      labels={RESOURCE_NAME_PREFIX: "true"},
      max_run_duration=f"{NODE_MAX_RUN_DURATION_SECONDS}s",  # 24 hours
    ),
    placement_policy=placement,
  )
