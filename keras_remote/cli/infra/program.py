"""Pulumi inline program for keras-remote infrastructure.

Defines all GCP resources needed for keras-remote: API services,
Artifact Registry, GKE cluster, optional accelerator node pools,
and per-namespace isolation resources.
"""

import pulumi
import pulumi_gcp as gcp
import pulumi_kubernetes as k8s

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


def create_program(config, resources_to_import=None, skip_iam=False):
  """Create a Pulumi inline program function closed over the config.

  Args:
      config: InfraConfig instance.
      resources_to_import: Optional dict mapping Pulumi logical resource
          name to GCP import ID, for adopting pre-existing cloud
          resources into Pulumi state.
      skip_iam: If True, skip GCP IAM resource declarations so that
          only K8s resources are managed. Useful when the caller lacks
          IAM permissions (403).

  Returns:
      A callable suitable for pulumi.automation.create_or_select_stack().
  """

  def pulumi_program():
    project_id = config.project
    zone = config.zone
    ar_location = zone_to_ar_location(zone)
    cluster_name = config.cluster_name
    node_pools = config.node_pools
    namespaces = getattr(config, "namespaces", [])
    import_ids = resources_to_import or {}

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
      repository_id="keras-remote",
      location=ar_location,
      format="DOCKER",
      description="keras-remote container images",
      project=project_id,
      opts=pulumi.ResourceOptions(
        depends_on=enabled_apis,
        import_=import_ids.get("keras-remote-repo"),
      ),
    )

    # 3. Cloud Storage buckets
    region = zone_to_region(zone)
    jobs_bucket_name = f"{project_id}-keras-remote-jobs"
    builds_bucket_name = f"{project_id}-keras-remote-builds"

    gcp.storage.Bucket(
      "keras-remote-jobs-bucket",
      name=jobs_bucket_name,
      location=region,
      project=project_id,
      force_destroy=True,
      uniform_bucket_level_access=True,
      opts=pulumi.ResourceOptions(
        depends_on=enabled_apis,
        import_=import_ids.get("keras-remote-jobs-bucket"),
      ),
    )

    gcp.storage.Bucket(
      "keras-remote-builds-bucket",
      name=builds_bucket_name,
      location=ar_location,
      project=project_id,
      force_destroy=True,
      opts=pulumi.ResourceOptions(
        depends_on=enabled_apis,
        import_=import_ids.get("keras-remote-builds-bucket"),
      ),
    )

    # 4. GKE Cluster (with Workload Identity and Dataplane V2)
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
      # Workload Identity — required for per-namespace GCP SA mapping
      workload_identity_config=gcp.container.ClusterWorkloadIdentityConfigArgs(
        workload_pool=f"{project_id}.svc.id.goog",
      ),
      # Dataplane V2 (Cilium) — required for NetworkPolicy enforcement
      datapath_provider="ADVANCED_DATAPATH",
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

    # K8s provider using the GKE cluster
    k8s_provider = k8s.Provider(
      "gke-k8s",
      kubeconfig=pulumi.Output.all(
        cluster.name, cluster.endpoint, cluster.master_auth
      ).apply(
        lambda args: _generate_kubeconfig(
          args[0], args[1], args[2], project_id, zone
        )
      ),
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

    # 6. Namespace resources (per non-default namespace)
    for ns_config in namespaces:
      _create_namespace_resources(
        ns_config,
        project_id,
        ar_location,
        jobs_bucket_name,
        builds_bucket_name,
        k8s_provider,
        enabled_apis,
        skip_iam=skip_iam,
      )

    # 7. Stack exports
    pulumi.export("project", project_id)
    pulumi.export("zone", zone)
    pulumi.export("cluster_name", cluster.name)
    pulumi.export("cluster_endpoint", cluster.endpoint)
    pulumi.export(
      "ar_registry",
      repo.name.apply(
        lambda _: f"{ar_location}-docker.pkg.dev/{project_id}/keras-remote"
      ),
    )

    # 8. Accelerator node pool exports (list of dicts)
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

    # 9. Namespace exports
    pulumi.export(
      "namespaces",
      [
        {
          "name": ns.name,
          "members": ns.members,
          "gpus": ns.gpus,
          "tpus": ns.tpus,
          "cpu": ns.cpu,
          "memory": ns.memory,
          "max_jobs": ns.max_jobs,
          "max_lws": ns.max_lws,
        }
        for ns in namespaces
      ],
    )

  return pulumi_program


def _generate_kubeconfig(name, endpoint, master_auth, project, zone):
  """Generate a kubeconfig string for the GKE cluster."""
  import json

  ca_cert = master_auth.get("cluster_ca_certificate", "")
  return json.dumps(
    {
      "apiVersion": "v1",
      "kind": "Config",
      "clusters": [
        {
          "name": name,
          "cluster": {
            "server": f"https://{endpoint}",
            "certificate-authority-data": ca_cert,
          },
        }
      ],
      "contexts": [
        {
          "name": name,
          "context": {"cluster": name, "user": name},
        }
      ],
      "current-context": name,
      "users": [
        {
          "name": name,
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


def _create_namespace_resources(
  ns_config,
  project_id,
  ar_location,
  jobs_bucket_name,
  builds_bucket_name,
  k8s_provider,
  depends_on,
  skip_iam=False,
):
  """Create all K8s and GCP resources for a non-default namespace."""
  ns_name = ns_config.name
  k8s_opts = pulumi.ResourceOptions(provider=k8s_provider)

  # K8s Namespace
  namespace = k8s.core.v1.Namespace(
    f"ns-{ns_name}",
    metadata=k8s.meta.v1.ObjectMetaArgs(
      name=ns_name,
      labels={"app": "keras-remote"},
    ),
    opts=k8s_opts,
  )

  ns_opts = pulumi.ResourceOptions(
    provider=k8s_provider, depends_on=[namespace]
  )

  # Collect K8s resources so IAM resources can depend on them,
  # ensuring K8s objects are fully created before IAM is attempted.
  k8s_resources = [namespace]

  # K8s ServiceAccount with Workload Identity annotation
  k8s_resources.append(
    k8s.core.v1.ServiceAccount(
      f"sa-{ns_name}",
      metadata=k8s.meta.v1.ObjectMetaArgs(
        name=f"{ns_name}-runner",
        namespace=ns_name,
        annotations={
          "iam.gke.io/gcp-service-account": (
            f"kr-{ns_name}@{project_id}.iam.gserviceaccount.com"
          ),
        },
      ),
      opts=ns_opts,
    )
  )

  # K8s Role
  k8s_resources.append(
    k8s.rbac.v1.Role(
      f"role-{ns_name}",
      metadata=k8s.meta.v1.ObjectMetaArgs(
        name="keras-remote-runner",
        namespace=ns_name,
      ),
      rules=[
        k8s.rbac.v1.PolicyRuleArgs(
          api_groups=["batch"],
          resources=["jobs"],
          verbs=["create", "get", "list", "watch", "delete"],
        ),
        k8s.rbac.v1.PolicyRuleArgs(
          api_groups=["leaderworkerset.x-k8s.io"],
          resources=["leaderworkersets"],
          verbs=["create", "get", "list", "watch", "delete"],
        ),
        k8s.rbac.v1.PolicyRuleArgs(
          api_groups=[""],
          resources=["pods", "pods/log"],
          verbs=["get", "list", "watch"],
        ),
        k8s.rbac.v1.PolicyRuleArgs(
          api_groups=[""],
          resources=["resourcequotas"],
          verbs=["get"],
        ),
      ],
      opts=ns_opts,
    )
  )

  # K8s RoleBindings for each member
  for member in ns_config.members:
    sanitized = member.replace("@", "-").replace(".", "-")
    # Determine subject kind based on prefix
    if member.startswith("group:"):
      subject_kind = "Group"
      subject_name = member.removeprefix("group:")
    else:
      subject_kind = "User"
      subject_name = member.removeprefix("user:")

    k8s_resources.append(
      k8s.rbac.v1.RoleBinding(
        f"rb-{ns_name}-{sanitized}",
        metadata=k8s.meta.v1.ObjectMetaArgs(
          name=f"{ns_name}-member-{sanitized}",
          namespace=ns_name,
        ),
        subjects=[
          k8s.rbac.v1.SubjectArgs(
            kind=subject_kind,
            name=subject_name,
            api_group="rbac.authorization.k8s.io",
          )
        ],
        role_ref=k8s.rbac.v1.RoleRefArgs(
          kind="Role",
          name="keras-remote-runner",
          api_group="rbac.authorization.k8s.io",
        ),
        opts=ns_opts,
      )
    )

  # K8s ResourceQuota
  hard = {}
  if ns_config.gpus is not None:
    hard["requests.nvidia.com/gpu"] = str(ns_config.gpus)
    hard["limits.nvidia.com/gpu"] = str(ns_config.gpus)
  if ns_config.tpus is not None:
    hard["requests.google.com/tpu"] = str(ns_config.tpus)
    hard["limits.google.com/tpu"] = str(ns_config.tpus)
  if ns_config.cpu is not None:
    hard["requests.cpu"] = str(ns_config.cpu)
    hard["limits.cpu"] = str(ns_config.cpu)
  if ns_config.memory is not None:
    hard["requests.memory"] = ns_config.memory
    hard["limits.memory"] = ns_config.memory
  if ns_config.max_jobs is not None:
    hard["count/jobs.batch"] = str(ns_config.max_jobs)
  if ns_config.max_lws is not None:
    hard["count/leaderworkersets.leaderworkerset.x-k8s.io"] = str(
      ns_config.max_lws
    )

  if hard:
    k8s_resources.append(
      k8s.core.v1.ResourceQuota(
        f"quota-{ns_name}",
        metadata=k8s.meta.v1.ObjectMetaArgs(
          name=f"{ns_name}-quota",
          namespace=ns_name,
        ),
        spec=k8s.core.v1.ResourceQuotaSpecArgs(hard=hard),
        opts=ns_opts,
      )
    )

  # K8s NetworkPolicy
  k8s_resources.append(
    k8s.networking.v1.NetworkPolicy(
      f"netpol-{ns_name}",
      metadata=k8s.meta.v1.ObjectMetaArgs(
        name="namespace-isolation",
        namespace=ns_name,
      ),
      spec=k8s.networking.v1.NetworkPolicySpecArgs(
        pod_selector=k8s.meta.v1.LabelSelectorArgs(),
        policy_types=["Ingress", "Egress"],
        ingress=[
          k8s.networking.v1.NetworkPolicyIngressRuleArgs(
            from_=[
              k8s.networking.v1.NetworkPolicyPeerArgs(
                pod_selector=k8s.meta.v1.LabelSelectorArgs(),
              )
            ],
          )
        ],
        egress=[
          # Intra-namespace
          k8s.networking.v1.NetworkPolicyEgressRuleArgs(
            to=[
              k8s.networking.v1.NetworkPolicyPeerArgs(
                pod_selector=k8s.meta.v1.LabelSelectorArgs(),
              )
            ],
          ),
          # HTTPS egress (GCS, AR, pip)
          k8s.networking.v1.NetworkPolicyEgressRuleArgs(
            to=[
              k8s.networking.v1.NetworkPolicyPeerArgs(
                ip_block=k8s.networking.v1.IPBlockArgs(cidr="0.0.0.0/0"),
              )
            ],
            ports=[
              k8s.networking.v1.NetworkPolicyPortArgs(protocol="TCP", port=443)
            ],
          ),
          # DNS to kube-system
          k8s.networking.v1.NetworkPolicyEgressRuleArgs(
            to=[
              k8s.networking.v1.NetworkPolicyPeerArgs(
                namespace_selector=k8s.meta.v1.LabelSelectorArgs(
                  match_labels={"kubernetes.io/metadata.name": "kube-system"},
                ),
              )
            ],
            ports=[
              k8s.networking.v1.NetworkPolicyPortArgs(protocol="UDP", port=53)
            ],
          ),
          # GKE metadata server (Workload Identity)
          k8s.networking.v1.NetworkPolicyEgressRuleArgs(
            to=[
              k8s.networking.v1.NetworkPolicyPeerArgs(
                ip_block=k8s.networking.v1.IPBlockArgs(
                  cidr="169.254.169.254/32"
                ),
              )
            ],
            ports=[
              k8s.networking.v1.NetworkPolicyPortArgs(protocol="TCP", port=80)
            ],
          ),
        ],
      ),
      opts=ns_opts,
    )
  )

  if skip_iam:
    return

  # GCP Service Account — depends on all K8s resources so that namespace
  # isolation is fully established before IAM is attempted.
  namespace_sa = gcp.serviceaccount.Account(
    f"sa-gcp-{ns_name}",
    account_id=f"kr-{ns_name}",
    display_name=f"keras-remote runner for {ns_name}",
    project=project_id,
    opts=pulumi.ResourceOptions(depends_on=depends_on + k8s_resources),
  )

  # GCS prefix-scoped access via IAM Condition
  gcp.storage.BucketIAMMember(
    f"{ns_name}-gcs-access",
    bucket=jobs_bucket_name,
    role="roles/storage.objectAdmin",
    member=namespace_sa.email.apply(lambda e: f"serviceAccount:{e}"),
    condition=gcp.storage.BucketIAMMemberConditionArgs(
      title=f"{ns_name}-prefix-only",
      description=f"Restrict to {ns_name}/ prefix",
      expression=(
        f'resource.name.startsWith("projects/_/buckets/'
        f'{jobs_bucket_name}/objects/{ns_name}/")'
      ),
    ),
  )

  # Custom list role (bucket-level, no condition)
  custom_list_role = gcp.projects.IAMCustomRole(
    f"gcs-lister-{ns_name}",
    role_id=f"kerasRemoteGcsLister{ns_name.replace('-', '')}",
    title=f"keras-remote GCS lister for {ns_name}",
    permissions=["storage.objects.list"],
    project=project_id,
  )
  gcp.storage.BucketIAMMember(
    f"{ns_name}-gcs-list",
    bucket=jobs_bucket_name,
    role=custom_list_role.id,
    member=namespace_sa.email.apply(lambda e: f"serviceAccount:{e}"),
  )

  # Artifact Registry reader
  gcp.artifactregistry.RepositoryIamMember(
    f"{ns_name}-ar-reader",
    repository="keras-remote",
    location=ar_location,
    role="roles/artifactregistry.reader",
    member=namespace_sa.email.apply(lambda e: f"serviceAccount:{e}"),
    project=project_id,
  )

  # Workload Identity binding
  gcp.serviceaccount.IAMMember(
    f"{ns_name}-workload-identity",
    service_account_id=namespace_sa.name,
    role="roles/iam.workloadIdentityUser",
    member=pulumi.Output.concat(
      "serviceAccount:",
      project_id,
      ".svc.id.goog",
      f"[{ns_name}/{ns_name}-runner]",
    ),
  )

  # Per-member build permissions
  for member in ns_config.members:
    _create_member_iam(
      ns_name,
      member,
      project_id,
      ar_location,
      builds_bucket_name,
      extra_depends=k8s_resources,
    )


def _create_member_iam(
  ns_name,
  member,
  project_id,
  ar_location,
  builds_bucket_name,
  extra_depends=None,
):
  """Grant per-member build permissions (Cloud Build, AR, builds bucket)."""
  sanitized = member.replace("@", "-").replace(".", "-")
  iam_member = f"user:{member}" if ":" not in member else member
  iam_opts = pulumi.ResourceOptions(depends_on=extra_depends or [])

  # Cloud Build editor
  gcp.projects.IAMMember(
    f"{ns_name}-{sanitized}-cloudbuild",
    project=project_id,
    role="roles/cloudbuild.builds.editor",
    member=iam_member,
    opts=iam_opts,
  )

  # Storage objectAdmin on builds bucket
  gcp.storage.BucketIAMMember(
    f"{ns_name}-{sanitized}-builds-storage",
    bucket=builds_bucket_name,
    role="roles/storage.objectAdmin",
    member=iam_member,
    opts=iam_opts,
  )

  # Artifact Registry writer
  gcp.artifactregistry.RepositoryIamMember(
    f"{ns_name}-{sanitized}-ar-writer",
    repository="keras-remote",
    location=ar_location,
    role="roles/artifactregistry.writer",
    member=iam_member,
    project=project_id,
    opts=iam_opts,
  )


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
          count=1,
        ),
      ],
      labels={RESOURCE_NAME_PREFIX: "true"},
      max_run_duration=f"{NODE_MAX_RUN_DURATION_SECONDS}s",  # 24 hours
      workload_metadata_config=gcp.container.NodePoolNodeConfigWorkloadMetadataConfigArgs(
        mode="GKE_METADATA",
      ),
    ),
  )


def _create_tpu_node_pool(cluster, tpu: TpuConfig, zone, project_id, pool_name):
  """Create a TPU GKE node pool."""
  # Single-host TPU slices (1 node) must not specify placement_policy;
  # multi-host slices require COMPACT placement with an explicit topology.
  is_multi_host = tpu.num_nodes > 1
  min_nodes = tpu.num_nodes if is_multi_host else 0

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
      workload_metadata_config=gcp.container.NodePoolNodeConfigWorkloadMetadataConfigArgs(
        mode="GKE_METADATA",
      ),
    ),
    placement_policy=placement,
  )
