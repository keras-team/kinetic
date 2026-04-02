"""Pulumi inline program for kinetic infrastructure.

Defines all GCP resources needed for kinetic: API services,
Artifact Registry, GKE cluster, and optional accelerator node pools.
"""

import json

import pulumi
import pulumi_gcp as gcp
import pulumi_kubernetes as k8s

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


def _build_kubeconfig(cluster_name, endpoint, ca_certificate, project_id):
  """Build a kubeconfig YAML string from GKE cluster outputs.

  Returns a ``pulumi.Output[str]`` that resolves once the cluster is ready.
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
      "kinetic-repo",
      repository_id=f"kn-{cluster_name}",
      location=ar_location,
      format="DOCKER",
      description="kinetic container images",
      project=project_id,
      opts=pulumi.ResourceOptions(depends_on=enabled_apis),
    )

    # 3. Cloud Storage buckets
    region = zone_to_region(zone)

    jobs_bucket = gcp.storage.Bucket(
      "kinetic-jobs-bucket",
      name=f"{project_id}-kn-{cluster_name}-jobs",
      location=region,
      project=project_id,
      force_destroy=True,
      uniform_bucket_level_access=True,
      lifecycle_rules=[
        gcp.storage.BucketLifecycleRuleArgs(
          action=gcp.storage.BucketLifecycleRuleActionArgs(type="Delete"),
          condition=gcp.storage.BucketLifecycleRuleConditionArgs(age=30),
        ),
      ],
      opts=pulumi.ResourceOptions(depends_on=enabled_apis),
    )

    builds_bucket = gcp.storage.Bucket(
      "kinetic-builds-bucket",
      name=f"{project_id}-kn-{cluster_name}-builds",
      location=ar_location,
      project=project_id,
      force_destroy=True,
      uniform_bucket_level_access=True,
      lifecycle_rules=[
        gcp.storage.BucketLifecycleRuleArgs(
          action=gcp.storage.BucketLifecycleRuleActionArgs(type="Delete"),
          condition=gcp.storage.BucketLifecycleRuleConditionArgs(age=30),
        ),
      ],
      opts=pulumi.ResourceOptions(depends_on=enabled_apis),
    )

    # 4. Service accounts.
    # Node SA — used by GKE workload pods (GCS, logging, monitoring,
    #   pulling images from AR).
    node_sa = gcp.serviceaccount.Account(
      "kinetic-node-sa",
      account_id=f"kn-{cluster_name}-nodes",
      display_name=f"kinetic {cluster_name} node SA",
      project=project_id,
      opts=pulumi.ResourceOptions(depends_on=enabled_apis),
    )
    # Project-level roles (inherently project-scoped).
    for role in [
      "roles/logging.logWriter",
      "roles/monitoring.metricWriter",
    ]:
      gcp.projects.IAMMember(
        f"node-sa-{role.split('/')[-1]}",
        project=project_id,
        role=role,
        member=node_sa.email.apply(lambda e: f"serviceAccount:{e}"),
        opts=pulumi.ResourceOptions(depends_on=enabled_apis),
      )

    # Resource-level roles — scoped to the specific buckets and repository.
    for bucket_name, bucket in [
      ("jobs", jobs_bucket),
      ("builds", builds_bucket),
    ]:
      gcp.storage.BucketIAMMember(
        f"node-sa-storage-{bucket_name}",
        bucket=bucket.name,
        role="roles/storage.objectAdmin",
        member=node_sa.email.apply(lambda e: f"serviceAccount:{e}"),
      )

    gcp.artifactregistry.RepositoryIamMember(
      "node-sa-ar-reader",
      repository=repo.name,
      location=ar_location,
      project=project_id,
      role="roles/artifactregistry.reader",
      member=node_sa.email.apply(lambda e: f"serviceAccount:{e}"),
    )

    # Build SA — used as the Cloud Build execution SA, avoiding
    #   reliance on default SAs whose permissions vary across project
    #   vintages and org policies.
    build_sa = gcp.serviceaccount.Account(
      "kinetic-build-sa",
      account_id=f"kn-{cluster_name}-builds",
      display_name=f"kinetic {cluster_name} build SA",
      project=project_id,
      opts=pulumi.ResourceOptions(depends_on=enabled_apis),
    )
    # Project-level roles (inherently project-scoped).
    for role in [
      "roles/logging.logWriter",
      "roles/secretmanager.secretAccessor",
    ]:
      gcp.projects.IAMMember(
        f"build-sa-{role.split('/')[-1]}",
        project=project_id,
        role=role,
        member=build_sa.email.apply(lambda e: f"serviceAccount:{e}"),
        opts=pulumi.ResourceOptions(depends_on=enabled_apis),
      )

    # Resource-level roles — scoped to the specific buckets and repository.
    for bucket_name, bucket in [
      ("jobs", jobs_bucket),
      ("builds", builds_bucket),
    ]:
      gcp.storage.BucketIAMMember(
        f"build-sa-storage-{bucket_name}",
        bucket=bucket.name,
        role="roles/storage.objectAdmin",
        member=build_sa.email.apply(lambda e: f"serviceAccount:{e}"),
      )

    gcp.artifactregistry.RepositoryIamMember(
      "build-sa-ar-writer",
      repository=repo.name,
      location=ar_location,
      project=project_id,
      role="roles/artifactregistry.writer",
      member=build_sa.email.apply(lambda e: f"serviceAccount:{e}"),
    )

    # 5. VPC network (avoids reliance on a "default" network, which
    #    may not exist in projects with org-policy constraints).
    network = gcp.compute.Network(
      "kinetic-network",
      name=f"kn-{cluster_name}",
      project=project_id,
      auto_create_subnetworks=True,
      opts=pulumi.ResourceOptions(depends_on=enabled_apis),
    )

    # 5b. Cloud Router + NAT — private-node clusters have no external
    #     IPs, so outbound traffic (image pulls, etc.) needs Cloud NAT.
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

    # 6. GKE Cluster (private nodes — satisfies
    #    compute.vmExternalIpAccess org-policy constraints).
    cluster = gcp.container.Cluster(
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
      # Match setup.sh: --no-enable-autoupgrade
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
      opts=pulumi.ResourceOptions(depends_on=enabled_apis),
    )

    # 7. Kubernetes resources (provider derived from cluster outputs).
    k8s_provider = k8s.Provider(
      "k8s-provider",
      kubeconfig=_build_kubeconfig(
        cluster.name,
        cluster.endpoint,
        cluster.master_auth.cluster_ca_certificate,
        project_id,
      ),
    )

    # 7a. Workload Identity binding — allow the kinetic KSA to
    #     impersonate the node GSA. Scoped to the node SA so the KSA
    #     can only impersonate this specific GSA.
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

    # 7b. Kinetic KSA with WIF annotation.
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

    # 7c. LeaderWorkerSet CRD (required for multi-host TPU Pathways).
    k8s.yaml.ConfigFile(
      "lws-crd",
      file=LWS_INSTALL_URL,
      opts=pulumi.ResourceOptions(provider=k8s_provider),
    )

    # 7d. NVIDIA GPU driver DaemonSet (only when GPU pools are present).
    if any(isinstance(np.accelerator, GpuConfig) for np in node_pools):
      k8s.yaml.ConfigFile(
        "nvidia-gpu-drivers",
        file=NVIDIA_DRIVER_DAEMONSET_URL,
        opts=pulumi.ResourceOptions(provider=k8s_provider),
      )

    # 8. Accelerator node pools (zero or more)
    pool_entries = []
    for np in node_pools:
      accel = np.accelerator
      pool_name = np.name
      if isinstance(accel, GpuConfig):
        pool = _create_gpu_node_pool(
          cluster,
          accel,
          zone,
          project_id,
          pool_name,
          node_sa.email,
          min_nodes=np.min_nodes,
        )
      elif isinstance(accel, TpuConfig):
        pool = _create_tpu_node_pool(
          cluster,
          accel,
          zone,
          project_id,
          pool_name,
          node_sa.email,
          min_nodes=np.min_nodes,
        )
      else:
        continue
      pool_entries.append((accel, pool, np.min_nodes))

    # 9. Stack exports
    # Exports that reference resource outputs (e.g. cluster.name,
    # repo.name, pool.name) create Pulumi dependencies — the export
    # only resolves when the underlying resource is successfully created.
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

    # 10. Accelerator node pool exports (list of dicts)
    if not pool_entries:
      pulumi.export("accelerators", [])
    else:
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

  return pulumi_program


def _create_gpu_node_pool(
  cluster,
  gpu: GpuConfig,
  zone,
  project_id,
  pool_name,
  service_account,
  min_nodes=0,
):
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
  cluster,
  tpu: TpuConfig,
  zone,
  project_id,
  pool_name,
  service_account,
  min_nodes=0,
):
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
