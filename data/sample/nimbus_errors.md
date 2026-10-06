# Nimbus Cloud Error Codes

## NB-401 Authentication failed

NB-401 means the API token is missing, expired or revoked. Tokens expire after 90 days. Generate a new token under Settings, then API Tokens, and update your CI secrets.

## NB-429 Rate limit exceeded

NB-429 is returned when a client sends more than 600 requests per minute to the public API. The response includes a Retry-After header. Clients should back off exponentially with jitter instead of retrying immediately.

## NB-503 Region unavailable

NB-503 indicates that the selected region is in maintenance or experiencing an outage. Check the status page and fail over to a secondary region if your deployment is multi-region.

## NB-507 Storage quota exceeded

NB-507 occurs when a project uses more storage than its plan allows. Delete unused snapshots or upgrade the plan. Writes are blocked until usage is below the quota, but reads continue to work.
