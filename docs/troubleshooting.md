# Troubleshooting

## Common Issues

### "Project must be specified" error

```bash
export KINETIC_PROJECT="your-project-id"
```

### "404 Requested entity was not found" error

Enable required APIs and create the Artifact Registry repository:

```bash
gcloud services enable compute.googleapis.com \
    cloudbuild.googleapis.com artifactregistry.googleapis.com \
    storage.googleapis.com container.googleapis.com \
    --project=$KINETIC_PROJECT

gcloud artifacts repositories create kinetic \
    --repository-format=docker \
    --location=us \
    --project=$KINETIC_PROJECT
```

### Permission denied errors

Grant required IAM roles:

```bash
gcloud projects add-iam-policy-binding $KINETIC_PROJECT \
    --member="user:your-email@example.com" \
    --role="roles/storage.admin"
```

### Container build failures

Check Cloud Build logs:

```bash
gcloud builds list --project=$KINETIC_PROJECT --limit=5
```

## Verify Setup

Run `kinetic status` to check the health of your infrastructure. For manual verification:

```bash
# Check authentication
gcloud auth list

# Check project
echo $KINETIC_PROJECT

# Check APIs
gcloud services list --enabled --project=$KINETIC_PROJECT \
    | grep -E "(cloudbuild|artifactregistry|storage|container)"
```
