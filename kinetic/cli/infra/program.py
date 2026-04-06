"""Pulumi inline program for kinetic infrastructure.

Defines all GCP resources needed for kinetic: API services,
Artifact Registry, GKE cluster, and optional accelerator node pools.
"""

import json
from collections.abc import Callable

import pulumi
import pulumi_command as command
import pulumi_gcp as gcp
import pulumi_kubernetes as k8s

from kinetic.cli.config import InfraConfig, NodePoolConfig
from kinetic.cli.constants import (
  GPU_NODE_POOL_MAX_SCALE_UP,
  KINETIC_KSA_NAME,
  LWS_INSTALL_URL,
  MAX_CLUSTER_CPU,
  MAX_CLUSTER_MEMORY_GB,
  NODE_MAX_RUN_DURATION_SECONDS,
  NVIDIA_DRIVER_DAEMONSET_URL,
  REQUIRED_APIS,
  RESOURCE_NAME_PREFIX,
)
from kinetic.constants import zone_to_ar_location, zone_to_region
from kinetic.core.accelerators import GpuConfig, TpuConfig

# With a dedicated node SA and IAM roles controlling access, a single
# cloud-platform scope is sufficient — IAM is the sole gatekeeper.
_CLOUD_PLATFORM_SCOPE = ["https://www.googleapis.com/auth/cloud-platform"]

_BUCKET_LIFECYCLE_30D = [
  gcp.storage.BucketLifecycleRuleArgs(
    action=gcp.storage.BucketLifecycleRuleActionArgs(type="Delete"),
    condition=gcp.storage.BucketLifecycleRuleConditionArgs(age=30),
  ),
]


def _build_kubeconfig(
  cluster_name: pulumi.Input[str],
  endpoint: pulumi.Input[str],
  ca_certificate: pulumi.Input[str],
  project_id: pulumi.Input[str],
) -> pulumi.Output[str]:
  """Build a kubeconfig YAML string from GKE cluster outputs.

  Returns a `pulumi.Output[str]` that resolves once the cluster is ready.
  """
  return pulumi.Output.all(
    cluster_name, endpoint, ca_certificate, project_id
  ).apply(
    lambda args: json.dumps(
      {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [
          {
            "name": "cluster",
            "cluster": {
              "server": f"https://{args[1]}",
              "certificate-authority-data": args[2],
            },
          }
        ],
        "contexts": [
          {
            "name": "context",
            "context": {"cluster": "cluster", "user": "user"},
          }
        ],
        "current-context": "context",
        "users": [
          {
            "name": "user",
            "user": {
              "exec": {
                "apiVersion": "client.authentication.k8s.io/v1beta1",
                "command": "gke-gcloud-auth-plugin",
                "installHint": "Install gke-gcloud-auth-plugin",
                "provideClusterInfo": True,
              },
            },
          }
        ],
      }
    )
  )


def _enable_apis(project_id: str) -> list[gcp.projects.Service]:
  """Enable required GCP API services."""
  enabled_apis: list[gcp.projects.Service] = []
  for api in REQUIRED_APIS:
    svc = gcp.projects.Service(
      f"api-{api.split('.')[0]}",
      service=api,
      project=project_id,
      disable_on_destroy=False,
      disable_dependent_services=False,
    )
    enabled_apis.append(svc)
  return enabled_apis


def _create_buckets(
  project_id: str,
  cluster_name: str,
  region: str,
  ar_location: str,
  enabled_apis: list[gcp.projects.Service],
) -> tuple[gcp.storage.Bucket, gcp.storage.Bucket]:
  """Create Cloud Storage buckets for jobs and build artifacts."""
  api_deps = pulumi.ResourceOptions(depends_on=enabled_apis)
  jobs_bucket = gcp.storage.Bucket(
    "kinetic-jobs-bucket",
    name=f"{project_id}-kn-{cluster_name}-jobs",
    location=region,
    project=project_id,
    force_destroy=True,
    uniform_bucket_level_access=True,
    lifecycle_rules=_BUCKET_LIFECYCLE_30D,
    opts=api_deps,
  )
  builds_bucket = gcp.storage.Bucket(
    "kinetic-builds-bucket",
    name=f"{project_id}-kn-{cluster_name}-builds",
    location=ar_location,
    project=project_id,
    force_destroy=True,
    uniform_bucket_level_access=True,
    lifecycle_rules=_BUCKET_LIFECYCLE_30D,
    opts=api_deps,
  )
  return jobs_bucket, builds_bucket


def _bind_sa_iam(
  sa_prefix: str,
  sa: gcp.serviceaccount.Account,
  project_id: str,
  project_roles: list[str],
  buckets: list[tuple[str, gcp.storage.Bucket]],
  repo: gcp.artifactregistry.Repository,
  ar_location: str,
  ar_role: str,
  enabled_apis: list[gcp.projects.Service],
) -> None:
  """Bind IAM roles to a service account at project, bucket, and AR levels.

  Args:
      ar_role: Full AR IAM role, e.g. "roles/artifactregistry.reader".
  """
  for role in project_roles:
    gcp.projects.IAMMember(
      f"{sa_prefix}-{role.split('/')[-1]}",
      project=project_id,
      role=role,
      member=sa.email.apply(lambda e: f"serviceAccount:{e}"),
      opts=pulumi.ResourceOptions(depends_on=enabled_apis),
    )
  for bucket_name, bucket in buckets:
    gcp.storage.BucketIAMMember(
      f"{sa_prefix}-storage-{bucket_name}",
      bucket=bucket.name,
      role="roles/storage.objectAdmin",
      member=sa.email.apply(lambda e: f"serviceAccount:{e}"),
    )
  gcp.artifactregistry.RepositoryIamMember(
    f"{sa_prefix}-ar-{ar_role.rsplit('.', 1)[-1]}",
    repository=repo.name,
    location=ar_location,
    project=project_id,
    role=ar_role,
    member=sa.email.apply(lambda e: f"serviceAccount:{e}"),
  )


def _create_service_accounts(
  project_id: str,
  cluster_name: str,
  ar_location: str,
  repo: gcp.artifactregistry.Repository,
  jobs_bucket: gcp.storage.Bucket,
  builds_bucket: gcp.storage.Bucket,
  enabled_apis: list[gcp.projects.Service],
) -> tuple[gcp.serviceaccount.Account, gcp.serviceaccount.Account]:
  """Create node and build service accounts with their IAM bindings."""
  api_deps = pulumi.ResourceOptions(depends_on=enabled_apis)
  bucket_pairs = [("jobs", jobs_bucket), ("builds", builds_bucket)]

  # Node SA — used by GKE workload pods (GCS, logging, monitoring,
  #   pulling images from AR).
  node_sa = gcp.serviceaccount.Account(
    "kinetic-node-sa",
    account_id=f"kn-{cluster_name}-nodes",
    display_name=f"kinetic {cluster_name} node SA",
    project=project_id,
    opts=api_deps,
  )
  _bind_sa_iam(
    "node-sa",
    node_sa,
    project_id,
    ["roles/logging.logWriter", "roles/monitoring.metricWriter"],
    bucket_pairs,
    repo,
    ar_location,
    "roles/artifactregistry.reader",
    enabled_apis,
  )

  # Build SA — used as the Cloud Build execution SA.
  build_sa = gcp.serviceaccount.Account(
    "kinetic-build-sa",
    account_id=f"kn-{cluster_name}-builds",
    display_name=f"kinetic {cluster_name} build SA",
    project=project_id,
    opts=api_deps,
  )
  _bind_sa_iam(
    "build-sa",
    build_sa,
    project_id,
    ["roles/logging.logWriter", "roles/secretmanager.secretAccessor"],
    bucket_pairs,
    repo,
    ar_location,
    "roles/artifactregistry.writer",
    enabled_apis,
  )

  return node_sa, build_sa


def _create_firewall_cleanup(
  network: gcp.compute.Network,
  project_id: str,
) -> command.local.Command:
  """Delete GKE auto-created firewall rules before network deletion.

  GKE creates firewall rules out-of-band (not managed by Pulumi). These
  can linger after cluster deletion and block VPC network teardown. This
  command runs on `pulumi destroy` to clean them up.

  Destroy order: cluster → firewall cleanup → network.
  """
  return command.local.Command(
    "cleanup-gke-firewalls",
    create="echo 'noop'",
    delete=pulumi.Output.format(
      "for fw in $(gcloud compute firewall-rules list "
      "--filter='network={0}' --format='value(name)' "
      "--project={1} 2>/dev/null); do "
      'gcloud compute firewall-rules delete "$fw" --quiet '
      "--project={1}; done",
      network.self_link,
      project_id,
    ),
    opts=pulumi.ResourceOptions(depends_on=[network]),
  )


def _create_network(
  project_id: str,
  cluster_name: str,
  region: str,
  enabled_apis: list[gcp.projects.Service],
) -> gcp.compute.Network:
  """Create VPC network with Cloud Router and NAT for private nodes."""
  network = gcp.compute.Network(
    "kinetic-network",
    name=f"kn-{cluster_name}",
    project=project_id,
    auto_create_subnetworks=True,
    opts=pulumi.ResourceOptions(
      depends_on=enabled_apis,
      custom_timeouts=pulumi.CustomTimeouts(delete="10m"),
    ),
  )
  router = gcp.compute.Router(
    "kinetic-router",
    name=f"kn-{cluster_name}-router",
    project=project_id,
    region=region,
    network=network.self_link,
  )
  gcp.compute.RouterNat(
    "kinetic-nat",
    name=f"kn-{cluster_name}-nat",
    project=project_id,
    region=region,
    router=router.name,
    nat_ip_allocate_option="AUTO_ONLY",
    source_subnetwork_ip_ranges_to_nat="ALL_SUBNETWORKS_ALL_IP_RANGES",
  )
  return network


def _create_gke_cluster(
  project_id: str,
  cluster_name: str,
  zone: str,
  network: gcp.compute.Network,
  node_sa: gcp.serviceaccount.Account,
  enabled_apis: list[gcp.projects.Service],
  firewall_cleanup: command.local.Command,
) -> gcp.container.Cluster:
  """Create the GKE cluster with autoscaling and private nodes."""
  return gcp.container.Cluster(
    "kinetic-cluster",
    name=cluster_name,
    location=zone,
    project=project_id,
    network=network.self_link,
    initial_node_count=1,
    remove_default_node_pool=False,
    node_config=gcp.container.ClusterNodeConfigArgs(
      machine_type="e2-standard-4",
      disk_size_gb=50,
      service_account=node_sa.email,
      oauth_scopes=_CLOUD_PLATFORM_SCOPE,
      workload_metadata_config=gcp.container.ClusterNodeConfigWorkloadMetadataConfigArgs(
        mode="GKE_METADATA",
      ),
    ),
    workload_identity_config=gcp.container.ClusterWorkloadIdentityConfigArgs(
      workload_pool=f"{project_id}.svc.id.goog",
    ),
    private_cluster_config=gcp.container.ClusterPrivateClusterConfigArgs(
      enable_private_nodes=True,
      enable_private_endpoint=False,
      master_ipv4_cidr_block="172.16.0.0/28",
    ),
    release_channel=gcp.container.ClusterReleaseChannelArgs(
      channel="UNSPECIFIED",
    ),
    deletion_protection=False,
    cluster_autoscaling=gcp.container.ClusterClusterAutoscalingArgs(
      enabled=True,
      autoscaling_profile="OPTIMIZE_UTILIZATION",
      auto_provisioning_defaults=gcp.container.ClusterClusterAutoscalingAutoProvisioningDefaultsArgs(
        service_account=node_sa.email,
        oauth_scopes=_CLOUD_PLATFORM_SCOPE,
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
    opts=pulumi.ResourceOptions(
      depends_on=[*enabled_apis, firewall_cleanup],
    ),
  )


def _create_k8s_resources(
  cluster: gcp.container.Cluster,
  node_sa: gcp.serviceaccount.Account,
  project_id: str,
  node_pools: list[NodePoolConfig],
) -> None:
  """Create Kubernetes resources: provider, KSA, LWS CRD, GPU drivers."""
  k8s_provider = k8s.Provider(
    "k8s-provider",
    kubeconfig=_build_kubeconfig(
      cluster.name,
      cluster.endpoint,
      cluster.master_auth.cluster_ca_certificate,
      project_id,
    ),
  )

  # Workload Identity binding — allow the kinetic KSA to impersonate the
  # node GSA.
  gcp.serviceaccount.IAMMember(
    "wif-kinetic-ksa",
    service_account_id=node_sa.name,
    role="roles/iam.workloadIdentityUser",
    member=pulumi.Output.format(
      "serviceAccount:{0}.svc.id.goog[default/{1}]",
      project_id,
      KINETIC_KSA_NAME,
    ),
    opts=pulumi.ResourceOptions(depends_on=[cluster]),
  )

  # Kinetic KSA with WIF annotation.
  k8s.core.v1.ServiceAccount(
    "kinetic-ksa",
    metadata=k8s.meta.v1.ObjectMetaArgs(
      name=KINETIC_KSA_NAME,
      namespace="default",
      annotations={
        "iam.gke.io/gcp-service-account": node_sa.email,
      },
    ),
    opts=pulumi.ResourceOptions(provider=k8s_provider),
  )

  # LeaderWorkerSet CRD (required for multi-host TPU Pathways).
  k8s.yaml.ConfigFile(
    "lws-crd",
    file=LWS_INSTALL_URL,
    opts=pulumi.ResourceOptions(provider=k8s_provider),
  )

  # NVIDIA GPU driver DaemonSet (only when GPU pools are present).
  if any(isinstance(np.accelerator, GpuConfig) for np in node_pools):
    k8s.yaml.ConfigFile(
      "nvidia-gpu-drivers",
      file=NVIDIA_DRIVER_DAEMONSET_URL,
      opts=pulumi.ResourceOptions(provider=k8s_provider),
    )


def _create_accelerator_pools(
  cluster: gcp.container.Cluster,
  node_pools: list[NodePoolConfig],
  zone: str,
  project_id: str,
  service_account: pulumi.Output[str],
) -> list[tuple[GpuConfig | TpuConfig, gcp.container.NodePool, int]]:
  """Create accelerator node pools and return entries for export."""
  pool_entries: list[
    tuple[GpuConfig | TpuConfig, gcp.container.NodePool, int]
  ] = []
  for np in node_pools:
    accel = np.accelerator
    if isinstance(accel, GpuConfig):
      pool = _create_gpu_node_pool(
        cluster,
        accel,
        zone,
        project_id,
        np.name,
        service_account,
        min_nodes=np.min_nodes,
      )
    elif isinstance(accel, TpuConfig):
      pool = _create_tpu_node_pool(
        cluster,
        accel,
        zone,
        project_id,
        np.name,
        service_account,
        min_nodes=np.min_nodes,
      )
    else:
      continue
    pool_entries.append((accel, pool, np.min_nodes))
  return pool_entries


def _export_stack_outputs(
  project_id: str,
  zone: str,
  cluster: gcp.container.Cluster,
  node_sa: gcp.serviceaccount.Account,
  repo: gcp.artifactregistry.Repository,
  ar_location: str,
  cluster_name: str,
  pool_entries: list[tuple[GpuConfig | TpuConfig, gcp.container.NodePool, int]],
) -> None:
  """Export all Pulumi stack outputs."""
  pulumi.export("project", project_id)
  pulumi.export("zone", zone)
  pulumi.export("cluster_name", cluster.name)
  pulumi.export("cluster_endpoint", cluster.endpoint)
  pulumi.export("node_sa_email", node_sa.email)
  pulumi.export(
    "ar_registry",
    repo.name.apply(
      lambda _: f"{ar_location}-docker.pkg.dev/{project_id}/kn-{cluster_name}"
    ),
  )

  if not pool_entries:
    pulumi.export("accelerators", [])
    return

  export_outputs = []
  for accel, pool, min_nodes in pool_entries:
    if isinstance(accel, GpuConfig):
      entry = pool.name.apply(
        lambda pn, a=accel, mn=min_nodes: {
          "type": "GPU",
          "name": a.name,
          "count": a.count,
          "machine_type": a.machine_type,
          "node_pool": pn,
          "node_count": 1,
          "min_nodes": mn,
        }
      )
    else:  # TpuConfig
      entry = pool.name.apply(
        lambda pn, a=accel, mn=min_nodes: {
          "type": "TPU",
          "name": a.name,
          "chips": a.chips,
          "topology": a.topology,
          "machine_type": a.machine_type,
          "node_pool": pn,
          "node_count": a.num_nodes,
          "min_nodes": mn,
        }
      )
    export_outputs.append(entry)
  pulumi.export("accelerators", pulumi.Output.all(*export_outputs))


def create_program(config: InfraConfig) -> Callable[[], None]:
  """Create a Pulumi inline program function closed over the config.

  Args:
      config: InfraConfig instance.

  Returns:
      A callable suitable for pulumi.automation.create_or_select_stack().
  """

  def pulumi_program() -> None:
    project_id = config.project
    zone = config.zone
    ar_location = zone_to_ar_location(zone)
    cluster_name = config.cluster_name
    region = zone_to_region(zone)

    enabled_apis = _enable_apis(project_id)

    repo = gcp.artifactregistry.Repository(
      "kinetic-repo",
      repository_id=f"kn-{cluster_name}",
      location=ar_location,
      format="DOCKER",
      description="kinetic container images",
      project=project_id,
      opts=pulumi.ResourceOptions(depends_on=enabled_apis),
    )

    jobs_bucket, builds_bucket = _create_buckets(
      project_id,
      cluster_name,
      region,
      ar_location,
      enabled_apis,
    )

    node_sa, _build_sa = _create_service_accounts(
      project_id,
      cluster_name,
      ar_location,
      repo,
      jobs_bucket,
      builds_bucket,
      enabled_apis,
    )

    network = _create_network(project_id, cluster_name, region, enabled_apis)
    firewall_cleanup = _create_firewall_cleanup(network, project_id)

    cluster = _create_gke_cluster(
      project_id,
      cluster_name,
      zone,
      network,
      node_sa,
      enabled_apis,
      firewall_cleanup,
    )

    _create_k8s_resources(cluster, node_sa, project_id, config.node_pools)

    pool_entries = _create_accelerator_pools(
      cluster,
      config.node_pools,
      zone,
      project_id,
      node_sa.email,
    )

    _export_stack_outputs(
      project_id,
      zone,
      cluster,
      node_sa,
      repo,
      ar_location,
      cluster_name,
      pool_entries,
    )

  return pulumi_program


def _create_gpu_node_pool(
  cluster: gcp.container.Cluster,
  gpu: GpuConfig,
  zone: str,
  project_id: str,
  pool_name: str,
  service_account: pulumi.Output[str],
  min_nodes: int = 0,
) -> gcp.container.NodePool:
  """Create a GPU-accelerated GKE node pool."""
  return gcp.container.NodePool(
    pool_name,
    name=pool_name,
    cluster=cluster.name,
    location=zone,
    project=project_id,
    initial_node_count=min_nodes,
    autoscaling=gcp.container.NodePoolAutoscalingArgs(
      min_node_count=min_nodes,
      max_node_count=min_nodes + GPU_NODE_POOL_MAX_SCALE_UP,
    ),
    management=gcp.container.NodePoolManagementArgs(
      auto_repair=True,
      auto_upgrade=True,
    ),
    node_config=gcp.container.NodePoolNodeConfigArgs(
      machine_type=gpu.machine_type,
      service_account=service_account,
      oauth_scopes=_CLOUD_PLATFORM_SCOPE,
      workload_metadata_config=gcp.container.NodePoolNodeConfigWorkloadMetadataConfigArgs(
        mode="GKE_METADATA",
      ),
      guest_accelerators=[
        gcp.container.NodePoolNodeConfigGuestAcceleratorArgs(
          type=gpu.gke_label,
          count=gpu.count,
        ),
      ],
      labels={RESOURCE_NAME_PREFIX: "true"},
      max_run_duration=f"{NODE_MAX_RUN_DURATION_SECONDS}s",  # 24 hours
      spot=gpu.spot,
    ),
  )


def _create_tpu_node_pool(
  cluster: gcp.container.Cluster,
  tpu: TpuConfig,
  zone: str,
  project_id: str,
  pool_name: str,
  service_account: pulumi.Output[str],
  min_nodes: int = 0,
) -> gcp.container.NodePool:
  """Create a TPU GKE node pool."""
  # Single-host TPU slices (1 node) must not specify placement_policy;
  # multi-host slices require COMPACT placement with an explicit topology.
  is_multi_host = tpu.num_nodes > 1
  if is_multi_host and min_nodes % tpu.num_nodes != 0:
    raise ValueError(
      f"min_nodes ({min_nodes}) must be a multiple of the TPU slice size "
      f"({tpu.num_nodes}) for multi-host TPUs."
    )

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
      max_node_count=min_nodes + tpu.num_nodes,
    ),
    management=gcp.container.NodePoolManagementArgs(
      auto_repair=True,
      auto_upgrade=True,
    ),
    node_config=gcp.container.NodePoolNodeConfigArgs(
      machine_type=tpu.machine_type,
      service_account=service_account,
      oauth_scopes=_CLOUD_PLATFORM_SCOPE,
      workload_metadata_config=gcp.container.NodePoolNodeConfigWorkloadMetadataConfigArgs(
        mode="GKE_METADATA",
      ),
      labels={RESOURCE_NAME_PREFIX: "true"},
      max_run_duration=None
      if tpu.spot
      else f"{NODE_MAX_RUN_DURATION_SECONDS}s",  # 24 hours
      spot=tpu.spot,
    ),
    placement_policy=placement,
  )
