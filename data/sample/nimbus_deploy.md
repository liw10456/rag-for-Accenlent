# Deploying on Nimbus Cloud

## Deployment methods

Applications can be deployed from a Git repository, from a container image, or with the nimbus CLI command nimbus deploy. Git deployments trigger automatically on every push to the configured branch.

## Rollbacks

Every deployment creates an immutable release. To roll back, run nimbus rollback with the release ID, or click Rollback next to any of the last 20 releases in the dashboard. A rollback takes effect in under a minute because the previous container image is already cached.

## Environment variables

Secrets should be stored as environment variables, not committed to the repository. Variables are encrypted at rest with AES-256 and are injected into the container at start time. Changing a variable triggers a restart of the service.

## Health checks

Nimbus sends an HTTP GET request to the /healthz path every 10 seconds. If three consecutive checks fail, the instance is replaced. A new deployment is only promoted to live traffic after it passes its health checks.
