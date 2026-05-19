# Kinetic Security & Threat Model

Kinetic is a tool designed to seamlessly execute user-defined Python code on
remote infrastructure (GCP/Kubernetes). Because its primary function is
arbitrary code execution, understanding the security boundaries is critical.

## The Security Boundary

**Kinetic assumes that any user authorized to submit jobs is a trusted
entity.** If a user is granted the IAM permissions to upload to the Kinetic GCS
buckets and the Kubernetes RBAC permissions to create Jobs/Pods, they have the
ability to execute arbitrary code on the cluster.

Kinetic **does not** attempt to sandbox, restrict, or monitor the Python code
written by authorized users. Securing the cluster against authorized users must
be done at the infrastructure level (e.g., network policies, minimal IAM roles
for the GKE nodes, namespace isolation).

## In-Scope Threats (What Kinetic Protects Against)

1. **Payload Tampering (Man-in-the-Middle):** Kinetic relies on `cloudpickle`
   to serialize functions. Deserializing untrusted pickle data is dangerous
(CWE-502). Kinetic protects against scenarios where an attacker gains write
access to the GCS bucket but lacks K8s access.
   * **Mitigation:** The client computes a SHA-256 hash of the payload at
     submission time and embeds it immutably into the Kubernetes Pod Spec. The
remote runner verifies the payload matches this hash before deserialization,
ensuring the payload wasn't tampered with in transit or at rest in GCS.

2. **Data Exfiltration via GCS:** Kinetic provisions cluster-scoped GCS
buckets.
   * **Mitigation:** Buckets should be locked down using IAM so that only
     authorized developers and the cluster's service account can read/write.

## Out-of-Scope Threats (What Kinetic Does Not Protect Against)

1. **Malicious Insiders:** Kinetic does not prevent an authorized user from
writing malicious code inside their `@kinetic.run()` function.
2. **Container Escapes:** If a user exploits a container runtime vulnerability
to escape the Pod, this is an infrastructure/GKE concern, not a Kinetic
concern.
3. **Compromised K8s Credentials:** If an attacker compromises a user's
`kubeconfig` or GCP credentials, they inherit that user's ability to run code.
