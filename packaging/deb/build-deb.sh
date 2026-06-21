#!/usr/bin/env bash
# packaging/deb/build-deb.sh
#
# Assemble a .deb from the `cargo pgrx package` staging tree.
#
# Inputs (environment):
#   PG_VER  PostgreSQL major version (14..18)
#   TAG     Release tag (e.g. v0.1.0); the leading "v" is stripped for the deb version.
#
# Output:
#   dist/parquet-azure-fdw-pg${PG_VER}_${TAG#v}_amd64.deb
#
# TODO(task-20): implement the actual .deb assembly. Sketch:
#   1. Source staging dir:  target/release/parquet_azure_fdw-pg${PG_VER}/
#      contains usr/lib/postgresql/${PG_VER}/lib/parquet_azure_fdw.so
#      and     usr/share/postgresql/${PG_VER}/extension/parquet_azure_fdw*
#   2. Copy under a fakeroot tree: build/deb/parquet-azure-fdw-pg${PG_VER}/...
#   3. Synthesize DEBIAN/control with Package, Version=${TAG#v}, Architecture=amd64,
#      Depends=postgresql-${PG_VER}, Maintainer, Description.
#   4. dpkg-deb --build --root-owner-group build/deb/parquet-azure-fdw-pg${PG_VER} dist/
#
# This stub exists so workflow wiring can be reviewed end-to-end; it intentionally
# fails so an unfinished apt rollout cannot silently ship empty packages.

set -euo pipefail

: "${PG_VER:?PG_VER is required}"
: "${TAG:?TAG is required}"

echo "build-deb.sh: PG_VER=${PG_VER} TAG=${TAG}"
echo "build-deb.sh: TODO(task-20) — .deb assembly not yet implemented." >&2
exit 1
