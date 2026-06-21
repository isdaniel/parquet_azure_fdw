# CLAUDE.md

PostgreSQL FDW (foreign data wrapper) in Rust for Parquet blobs in Azure Blob Storage. SELECT / INSERT / UPDATE / DELETE / COPY. pgrx 0.18.1, PG 14–18.

## Code map

```
src/
  lib.rs                          extension entry points + pg_test SQL suite
  error.rs                        FdwError + raise() ereport bridge (redacts on the way out)
  redact.rs                       strips sig=/Bearer/etc. from SDK error text
  runtime.rs                      thread-local current-thread tokio (block_on)
  azure/
    mod.rs                        AzureBlobClient (list_with_prefix_etags, get_body_if_match,
                                  put_if_match, delete_if_match, head_etag). MAX_LIST_RESULTS / MAX_BLOB_BYTES.
    auth.rs                       Credential / parse_auth_method (managed_identity, aad_sp, sas_url)
    reader.rs                     AsyncFileReader over range-GET
    writer.rs                     single-shot block upload + generate_blob_name
  convert/
    arrow_to_pg.rs                read path: arrow array → PG Datum
    pg_to_arrow.rs                write path: PG Datum → Arrow builders; RecordBatchBuilders
  parquet_io/
    reader.rs / writer.rs         ParquetReadOptions, ParquetBatchWriter, Compression
  fdw/
    mod.rs                        FdwRoutine wiring + version-portable tupdesc_attr shim
    options.rs                    SERVER / USER MAPPING / TABLE option parsers + SSRF validators
    pushdown.rs                   (stub) qual pushdown placeholder
    scan.rs                       7 read-path callbacks; ScanState; publishes etag handoff
    modify/
      mod.rs                      FdwModifyState (Insert | Update), BlobIdEntry { name, chunk_base_row, etag }
      insert.rs                   INSERT / COPY accumulator + flush
      update.rs                   UPDATE/DELETE: build_plan, exec_*, commit_plan (If-Match GET+PUT)
      kernel.rs                   apply_edits (pure; deletes via packed BooleanBufferBuilder, updates via overrides)
      rowid.rs                    RowId ↔ ctid encoding (CHUNK_ROWS = 65_536)
      scan_handoff.rs             thread-local (relid → Vec<(name, etag)>) scan→modify handoff
  test_harness/fake_blob_store.rs wiremock-backed fake (etag-aware, models 412)
tests/                            non-pg integration tests (gated on `pg_test` feature for some)
```

## Architectural invariants

- **`#![forbid(unsafe_code)]` on every module except `fdw/scan.rs`, `fdw/modify/{insert,update}.rs`, `fdw/mod.rs::alloc_node`.** Each `unsafe { ... }` block has a `// SAFETY:` comment. `#![deny(unsafe_op_in_unsafe_fn)]` is on in the carve-outs.
- **Never `pgrx::error!` / panic with an unredacted SDK error.** Build via `FdwError::azure(e)` / `azure_ctx(ctx, e)`. `raise()` redacts again as defense in depth.
- **UPDATE/DELETE lost-update guard:** scan captures etag at LIST/HEAD → publishes via `scan_handoff` → `build_plan` consumes (no re-listing in release builds) → `commit_plan` uses that etag for `If-Match` on GET, PUT, DELETE. 412 → `ConcurrentUpdate` → SQLSTATE 40001. `re_scan_foreign_scan` re-publishes the handoff so a rescan-then-modify sequence within one statement still sees the pinned snapshot.
- **No fallback to re-listing in release.** `build_plan` returns `SchemaMismatch` if no scan-time handoff is present. The `expand_glob_for_modify` fallback is compiled in only under `cfg(any(test, feature = "pg_test"))` so production builds physically cannot bypass the etag guard. The end-to-end SQLSTATE-40001 wiring is fenced by the integration test `tests/update_delete.rs::scan_handoff_to_commit_plan_lost_update_guard`, which publishes a real handoff, drives `commit_plan` against the wiremock fake, and asserts `ConcurrentUpdate` (`error::raise` maps that to SQLSTATE 40001).
- **`BlobIdEntry { name, chunk_base_row, etag }`** is the contract between scan-stamped ctids and the modify path. Three fields, all three required everywhere it's constructed.
- **Row identity:** synthetic ctid = `(blob_id, row_offset)`. `blob_id` indexes `blob_table` directly. Rows beyond `u32::MAX` per blob are rejected up front in `build_plan`.
- **Memory caps:** `MAX_LIST_RESULTS = 100_000`, `MAX_BLOB_BYTES = 512 MiB` enforced in `azure/mod.rs`. The cap is symmetric (read AND write paths: `get_with_etag`, `get_body_if_match`, `put_if_match`, `AzureBlobWriter::upload`). The 512 MiB ceiling accounts for `apply_edits`' peak ~3× decoded-blob multiplier (original concat + per-column rebuild pieces + final), keeping worst-case RSS comfortably below Postgres `MaxAllocSize` (~1 GiB). The delete mask in `kernel.rs` uses a packed `BooleanBufferBuilder` (1 bit/row), not `Vec<bool>` (8 bits/row), so a `u32::MAX`-row blob's mask is ~512 MiB instead of 4 GiB.
- **SSRF validators in `fdw/options.rs`:** account/container regex, endpoint allowlist, SAS host check (rejects link-local, RFC1918, IMDS), filename `..` traversal.

## Conventions

- Errors: `FdwError` enum in `error.rs`. Use `?` everywhere; `raise(e)` only at the FFI boundary (callbacks).
- Async: every `await` is wrapped in `runtime::block_on` inside a callback. Don't nest.
- Tests: `tests/*.rs` is the unit/integration suite (some `#![cfg(feature = "pg_test")]` for fake-blob-store-backed cases). In-crate `#[pg_test]` lives in `src/lib.rs` under `mod tests` — these drive Postgres.
- Compatibility: targets PG 14–18 via pgrx feature flags `pg14..pg18`. Use `crate::fdw::tupdesc_attr` (not `tupdesc.attrs`) — pg18 reshaped the layout.

## Workflows

```bash
make build                 # cargo build --no-default-features --features pg14
make test                  # cargo pgrx test pg14 (default PG)
make test-unit             # cargo check + clippy, no Postgres / no Docker
make test-all              # pgrx test on every PG version
make before-git-push       # what CI runs: fmt --check + clippy -D warnings + pgrx test (PG14)
make before-git-push-all   # same, every PG version
```

The wiremock fake is in-process — `cargo test` and `cargo pgrx test pg14` need no Docker. The opt-in `make test-live` smoke against real Azurite / Azure honors `AZURE_TEST_SAS_URL`.

## Pre-PR checklist

1. `make before-git-push` (must be exit 0).
2. If you touched `azure/mod.rs`, `fdw/scan.rs`, `fdw/modify/*`, or `redact.rs` — re-read the architectural invariants above; those are load-bearing.
3. New `unsafe { ... }` blocks: `// SAFETY:` comment is required by `unsafe_op_in_unsafe_fn`.
4. New `FdwError::Azure(...)` constructions: route through `FdwError::azure[_ctx]` so secrets get redacted.

## Known limitations (don't "fix" these in a side PR)

- Statement-atomicity is per-blob, not per-statement. Documented; retry on `serialization_failure`.
- Glob: single trailing `*` only.
- Qual pushdown / row-group pruning is stubbed (`pushdown.rs`).
- No Hive partition discovery, no MERGE.
- `account_key` auth is unsupported (SDK constraint) — use SAS.
- UPDATE/DELETE rewrites use the foreign-table `compression` option, NOT the codec of the source blob. A blob originally written with `gzip` will silently come back `snappy` (the default) after an UPDATE unless the table option says otherwise.
- `apply_edits` is non-streaming. Bound is `MAX_BLOB_BYTES = 512 MiB` (encoded). For larger blobs the kernel must be reworked to operate per-batch with a running offset before raising the cap.
