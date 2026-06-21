#!/usr/bin/env bash
# Convenience launcher for a local Azurite instance, mirroring the container
# the in-process test harness (tests/common/azurite.rs) spawns via
# testcontainers. Useful for ad-hoc psql sessions outside `cargo test`.
#
# Endpoints (after start):
#   Blob   http://127.0.0.1:10000/devstoreaccount1
#   Queue  http://127.0.0.1:10001/devstoreaccount1
#   Table  http://127.0.0.1:10002/devstoreaccount1
#
# Account name: devstoreaccount1
# Account key:  Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==
set -euo pipefail

docker rm -f azurite >/dev/null 2>&1 || true
docker run -d --name azurite -p 10000-10002:10000-10002 \
  mcr.microsoft.com/azure-storage/azurite:latest >/dev/null

echo "Azurite listening on http://127.0.0.1:10000/devstoreaccount1"
