#!/usr/bin/env bash


set -euo pipefail

: "${TAG:?TAG is required}"
: "${PRERELEASE:=false}"

echo "publish-repo.sh: TAG=${TAG} PRERELEASE=${PRERELEASE}"
echo "publish-repo.sh: TODO(task-20) — apt repo assembly not yet implemented." >&2
exit 1
