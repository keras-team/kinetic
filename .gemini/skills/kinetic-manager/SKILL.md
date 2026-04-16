---
name: kinetic-manager
description: |
  Use this skill to manage Keras and JAX remote jobs using Kinetic.
  It provides expertise in infrastructure provisioning, job submission, monitoring, and result retrieval on GCP.
---

# Kinetic Manager

This skill guides the agent in using Kinetic to run Keras/JAX workloads on cloud TPUs and GPUs.

## Workflow

### 1. Infrastructure Setup
If infrastructure is not yet provisioned:
- Use `kinetic up` to create the GKE cluster and required resources.
- Use `kinetic status` to check the current state.

### 2. Job Submission
To run a workload:
- Use `@kinetic.run()` for synchronous execution (blocks until completion).
- Use `@kinetic.submit()` for asynchronous execution (returns a `JobHandle`).

Example:
```python
import kinetic

@kinetic.submit(accelerator="tpu-v5e-1")
def train():
  import keras
  # ... training code ...
  return history.history
```

### 3. Job Management
For jobs submitted with `@kinetic.submit()`:
- **List jobs**: `kinetic jobs list`
- **Check status**: `kinetic jobs status <job-id>`
- **Stream logs**: `kinetic jobs logs <job-id> --follow`
- **Collect result**: `kinetic jobs result <job-id>`
- **Cancel job**: `kinetic jobs cancel <job-id>`

### 4. Data Management
Use `kinetic.Data` to pass local or GCS data to your remote functions.
```python
@kinetic.run(accelerator="tpu-v5e-1")
def train(data_dir):
  # ...
  pass

train(kinetic.Data("./my_dataset/"))
```

## Best Practices
- Prefer `@kinetic.submit()` for long-running jobs to avoid blocking the local session.
- Always use `kinetic down` when finished to avoid ongoing cloud costs.
- Use `kinetic doctor` to diagnose environment or credential issues.
