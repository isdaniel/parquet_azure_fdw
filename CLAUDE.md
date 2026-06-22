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
    partition_datum.rs            parse_text_to_datum: cast partition path text → PG Datum at scan-begin
  parquet_io/
    reader.rs / writer.rs         ParquetReadOptions, ParquetBatchWriter, Compression
    multifile.rs                  K-way heap merge over N parquet streams; NULLS LAST; iteration-time invariant check
  fdw/
    mod.rs                        FdwRoutine wiring + version-portable tupdesc_attr shim
    options.rs                    SERVER / USER MAPPING / TABLE option parsers + SSRF validators
    pushdown.rs                   is_pushable + build_row_filter (row-level) + prune_row_groups (row-group)
    pushdown_walk.rs              PG expression walker → PushedExpr; LIKE-prefix translation; collation guard
    glob.rs                       regex-backed full-glob parser (parse_glob → GlobPattern)
    import_schema.rs              IMPORT FOREIGN SCHEMA callback + directory-prefix grouping
    partition.rs                  Hive partition: path parsing + tuple keying + qual split + LIST-layer pruning
    scan.rs                       7 read-path callbacks; ScanState; publishes etag handoff
    parallel.rs                   parallel scan: DSM-backed cursor + 5 FFI callbacks + ParallelRanges RangeProducer
    modify/
      mod.rs                      FdwModifyState (Insert | Update), BlobIdEntry { name, chunk_base_row, etag }
      insert.rs                   INSERT / COPY accumulator + flush
      update.rs                   UPDATE/DELETE: build_plan, exec_*, commit_plan (If-Match GET+PUT)
      kernel.rs                   apply_edits_batch (pure per-batch) + apply_edits streaming wrapper; deletes via per-batch packed BooleanBufferBuilder, updates via overrides
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
- **Scan seams (SP-0):** `fdw/scan.rs` exposes three `pub(crate)` trait seams — `PlanSource`, `QualFilter`, `RangeProducer` — with default impls (`SingleGlobSource`, `PushedExprFilter`, `SequentialRanges`) that preserve pre-SP-0 behavior. SP-1..SP-3 each replace exactly one seam; do not collapse them back into free functions without rerunning the SP-1..SP-3 file-boundary analysis in the umbrella design doc.
- **Pushdown soundness (SP-1):** pushed-down logic may filter a row only when its three-valued result is `TRUE`. `UNKNOWN` and column-chunk-statistics absence both default to "keep". A row-group survives pruning iff some row could satisfy the quals. `is_pushable` rejects `TIMESTAMPTZ` and `Decimal128(_, scale > 18)`; the walker drops quals on non-default, non-C collations. Pushdown is advisory — PG always re-evaluates above the scan.
- **DDL preserves parquet column order (SP-2):** `IMPORT FOREIGN SCHEMA` and `import_parquet_azure_explicit` emit DDL whose column order matches the parquet file's schema position-for-position. This preserves the SP-1 I2 invariant that `PushedExprFilter::keep_row_groups` and `build_row_filter` index by the same column position across foreign-table schema and parquet schema.
- **Parallel scan SELECT-only (SP-3a):** `IsForeignScanParallelSafe` returns `true` only for `CMD_SELECT` AND when the table's `parallel_workers` option is not `Some(0)`. UPDATE/DELETE/INSERT always run sequentially so the leader-only `scan_handoff` chain (and the lost-update guard) is unchanged. Workers never publish to `scan_handoff`.
- **Hive partition (SP-3b):** declared via `partition_columns 'k1,k2'` + `partition_keys 'k1:type,k2:type'` (must agree on names + order). Partition cols are foreign-table attrs not present in the parquet file — synthesized per row from the blob path (`key=value/`). `storage_attno_to_parquet_idx: Vec<Option<usize>>` map remaps storage quals to parquet positions; partition quals are evaluated at LIST time and prune whole blobs. UPDATE that touches a partition col is rejected. Blob whose path is malformed or whose value fails the declared cast is skipped with a `NOTICE` — sound default, never crashes the scan.
- **Sorted merge SELECT-only (SP-3c):** declared via paired `sorted 'col1,col2'` + `files_in_order 'true'` table options (parse error if not paired). ASC + NULLS LAST only in v1. Sort cols MUST be storage cols (not partition cols). K ≤ 256 concurrently-open blobs. Sorted mode is parallel-unsafe: `IsForeignScanParallelSafe` returns false AND `get_foreign_paths` skips `add_partial_path`. Iteration-time invariant: every popped row's key MUST be >= last_emitted; violation raises `SchemaMismatch` naming the offending blob.
- **Streaming rewrite kernel (SP-4):** `kernel::apply_edits_batch(batch, base_offset, schema, edits)` is the pure per-batch primitive; `apply_edits` is a thin streaming wrapper over it (no `concat_batches`). `commit_plan` constructs the writer (with the SP-0 source codec) before the loop and streams source batches through `apply_edits_batch` → `writer.write`, with a running absolute offset, a cumulative decoded-byte `MAX_BLOB_BYTES` counter (cap unchanged), and a post-loop check that every `updates` key is `< total rows`. Empty result routes to the existing empty-delete (`If-Match` DELETE) path.
- **`enable_multifile` GUC (SP-4):** `parquet_fdw.enable_multifile` (Userset, default on). When off, `build_sorted_stream_if_active` returns `Ok(None)` at the top, forcing the sequential per-blob scan even for `sorted`/`files_in_order` tables. Correctness is preserved; only the K-way merge optimization is skipped.

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
- `account_key` auth is unsupported (SDK constraint) — use SAS.
- `apply_edits` streams per-batch (peak RSS ~1 source + ~1 output batch). The `MAX_BLOB_BYTES = 512 MiB` ceiling is still enforced by a cumulative decoded-byte counter in `commit_plan` — raising it is a separate, measured SP now that the kernel no longer holds the whole blob.
- Per-INSERT memory cap is per-partition-group, not per-statement. A partitioned INSERT routing into `G` distinct partition tuples concurrently buffers up to `G × MAX_BLOB_BYTES` (worst case) before flush, since each group has its own builders + parquet writer. No MERGE.
- Sorted merge in v1: ASC + NULLS LAST only; sort cols must be storage (not partition); K ≤ 256 concurrently-open blobs; PathKey emission is deferred — `get_foreign_paths` returns no path key, so PG adds a redundant `Sort` node above the scan (output is still correctly ordered). Sorted mode disables projection pushdown (reads all columns), and also disables SP-1 row-group pruning (`keep_row_groups`) and row-level filtering (`with_row_filter`): each blob's stream is opened with a bare `ParquetRecordBatchStreamBuilder::new` — sound because PG re-evaluates all quals above the scan, just less efficient. The partitioned + sorted combination is rejected in v1.
