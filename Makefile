# parquet_azure_fdw Makefile
#
# Adapted from isdaniel/redis_fdw_rs. Azurite is launched on-demand by
# testcontainers inside the in-crate `#[pg_test]` cases, so unlike the Redis
# FDW we don't need a docker-compose for tests — only a running Docker daemon.

.PHONY: help build build-release install test test-all test-unit \
        test-pg14 test-pg15 test-pg16 test-pg17 test-pg18 \
        test-live test-live-pg14 test-live-pg15 test-live-pg16 test-live-pg17 test-live-pg18 \
        build-all clean format lint check stop-pg cleanup-azurite docker-status \
        before-git-push before-git-push-all

# All supported PostgreSQL major versions.
VERSIONS = pg14 pg15 pg16 pg17 pg18

# Default PG version for single-target commands; override with `make test PG=pg17`.
PG ?= pg14

# Env file for the opt-in live Azure smoke test (gitignored).
ENV_FILE ?= .env.local

# Use --no-default-features so PG selection is always explicit and reproducible.
FEATURES_LIB   = $(PG)
FEATURES_TEST  = $(PG),pg_test
CARGO_LIB      = cargo build --no-default-features --features $(FEATURES_LIB)
CARGO_CLIPPY   = cargo clippy --no-default-features --all-targets --features $(FEATURES_TEST) -- -D warnings
CARGO_CHECK    = cargo check  --no-default-features --features $(FEATURES_LIB)
CARGO_PGRX     = cargo pgrx test $(PG)

# Default target
help:
	@echo "parquet_azure_fdw — Available Make Targets:"
	@echo ""
	@echo "  make test                Run pgrx tests for default PG version (PG=$(PG))"
	@echo "  make test PG=pg17        Run pgrx tests for a specific PG version"
	@echo "  make test-pg14..pg18     Per-version shortcuts"
	@echo "  make test-all            Run pgrx tests on every PG version ($(VERSIONS))"
	@echo "  make test-unit           cargo check + clippy (Docker NOT required)"
	@echo "  make test-live           Run live-Azure smoke (sources $(ENV_FILE) for AZURE_TEST_SAS_URL)"
	@echo ""
	@echo "  make build               Debug build for current PG ($(PG))"
	@echo "  make build-release       Release build for current PG ($(PG))"
	@echo "  make build-all           Build every PG version — quick sanity check"
	@echo "  make install             cargo pgrx install --release (default PG)"
	@echo ""
	@echo "  make format              cargo fmt"
	@echo "  make lint                cargo clippy on $(FEATURES_TEST)"
	@echo "  make check               cargo check on $(FEATURES_LIB)"
	@echo "  make clean               cargo clean"
	@echo ""
	@echo "  make stop-pg             Stop a stale pgrx test PostgreSQL instance"
	@echo "  make cleanup-azurite     Remove any leftover Azurite testcontainers"
	@echo "  make docker-status       Show Azurite + pgrx-test containers"
	@echo ""
	@echo "  make before-git-push     Run all CI checks locally (fmt-check, clippy, pgrx test) for PG=$(PG)"
	@echo "  make before-git-push-all Run the above for every PG version"

# ─── Build ────────────────────────────────────────────────────────────────────

build:
	$(CARGO_LIB)

build-release:
	cargo build --release --no-default-features --features $(FEATURES_LIB)

install:
	cargo pgrx install --release --no-default-features --features $(FEATURES_LIB)

# Quick smoke that every PG version still compiles. Stops on first failure.
build-all:
	@for pg in $(VERSIONS); do \
		echo "=== build $$pg ==="; \
		cargo build --no-default-features --features $$pg || exit 1; \
	done
	@echo "=== all PG versions build clean ==="

# ─── Testing ──────────────────────────────────────────────────────────────────

# Run pgrx test for the default PG version (override with PG=pg16 etc.).
test: stop-pg
	$(CARGO_PGRX)

test-pg14:
	$(MAKE) test PG=pg14

test-pg15:
	$(MAKE) test PG=pg15

test-pg16:
	$(MAKE) test PG=pg16

test-pg17:
	$(MAKE) test PG=pg17

test-pg18:
	$(MAKE) test PG=pg18

# Quick compile + lint check — does not start any container.
test-unit:
	$(CARGO_CHECK)
	$(CARGO_CLIPPY)

# Run pgrx tests on every PG version. Each version gets its own stale-PG cleanup.
test-all:
	@for pg in $(VERSIONS); do \
		echo "=== pgrx test $$pg ==="; \
		$(MAKE) stop-pg PG=$$pg; \
		cargo pgrx test $$pg || exit 1; \
	done
	@echo "=== all PG versions pass ==="

# Live Azure smoke test (opt-in). Reads AZURE_TEST_SAS_URL from $(ENV_FILE) if
# present; falls back to the inherited env. The test no-ops cleanly when the
# SAS URL is missing, so this is safe in CI without secrets configured.
test-live:
	@if [ -f "$(ENV_FILE)" ]; then \
		echo "Sourcing $(ENV_FILE)"; \
		set -a; . ./$(ENV_FILE); set +a; \
		cargo pgrx test $(PG) -- live_smoke; \
	else \
		echo "$(ENV_FILE) not found — running with inherited env"; \
		cargo pgrx test $(PG) -- live_smoke; \
	fi

test-live-pg14:
	$(MAKE) test-live PG=pg14
test-live-pg15:
	$(MAKE) test-live PG=pg15
test-live-pg16:
	$(MAKE) test-live PG=pg16
test-live-pg17:
	$(MAKE) test-live PG=pg17
test-live-pg18:
	$(MAKE) test-live PG=pg18

# ─── Infrastructure cleanup ───────────────────────────────────────────────────

PG_DATA_DIR := $(HOME)/.pgrx/data-$(subst pg,,$(PG))
PG_CTL      := $(HOME)/.pgrx/$(subst pg,,$(PG)).*/pgrx-install/bin/pg_ctl

# Kill a stale pgrx-test PostgreSQL instance left over from a prior run.
stop-pg:
	@if [ -f "$(PG_DATA_DIR)/postmaster.pid" ]; then \
		echo "Stopping stale PostgreSQL in $(PG_DATA_DIR)"; \
		( ls $(PG_CTL) 2>/dev/null | head -1 | xargs -r -I{} {} stop -D "$(PG_DATA_DIR)" 2>/dev/null ) || \
			rm -f "$(PG_DATA_DIR)/postmaster.pid"; \
	fi

# Remove any leftover Azurite containers from interrupted test runs.
# testcontainers names them with a random suffix, so match by image.
cleanup-azurite:
	@docker ps -a --filter "ancestor=mcr.microsoft.com/azure-storage/azurite" \
		--format "{{.ID}}" | xargs -r docker rm -f 2>/dev/null || true
	@echo "Azurite leftovers removed."

docker-status:
	@docker ps --filter "ancestor=mcr.microsoft.com/azure-storage/azurite" \
		--format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || true
	@docker ps --filter "name=pgrx" \
		--format "table {{.Names}}\t{{.Status}}" || true

# ─── Development ──────────────────────────────────────────────────────────────

format:
	cargo fmt

lint:
	$(CARGO_CLIPPY)

check:
	$(CARGO_CHECK)

clean:
	cargo clean

# ─── Pre-push verification ────────────────────────────────────────────────────

# Run the same checks CI will run: fmt check, clippy with -D warnings, and the
# full pgrx test suite for $(PG). Use this before `git push`.
before-git-push: stop-pg
	cargo fmt -- --check
	$(CARGO_CLIPPY)
	$(CARGO_PGRX)

# Same as above, but iterates every supported PG version. Slow but exhaustive.
before-git-push-all:
	cargo fmt -- --check
	@for pg in $(VERSIONS); do \
		echo "=== before-git-push for $$pg ==="; \
		$(MAKE) before-git-push PG=$$pg || exit 1; \
	done
	@echo "=== all PG versions verified ==="

.DEFAULT_GOAL := help
