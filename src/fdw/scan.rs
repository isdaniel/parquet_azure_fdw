#![deny(unsafe_op_in_unsafe_fn)]
//! Postgres FDW read-path callbacks.
//!
//! This is the FFI boundary between Postgres' executor and our async Azure +
//! parquet stack. Every `unsafe { ... }` block in this file is annotated with
//! a `// SAFETY:` comment naming the pgrx / Postgres invariant it relies on.
//!
//! The 7 callbacks implemented here match the function-pointer signatures
//! pgrx exposes via `pg_sys::*_function` types and are intended to be
//! installed into an `FdwRoutine` by the (Task 14) handler glue.
//!
//! ## State lifetime
//!
//! `BeginForeignScan` boxes a [`ScanState`] and stashes the raw pointer in
//! `ForeignScanState.fdw_state`. `IterateForeignScan` borrows it back as
//! `&mut ScanState`. `EndForeignScan` reclaims and drops the box. PG only
//! invokes these on the same `ForeignScanState`, so the pointer round-trip
//! is sound.
//!
//! ## Async / blocking model
//!
//! Parquet reads are async (`AsyncFileReader`). The scan callbacks are
//! synchronous PG functions, so each async operation is wrapped in
//! [`crate::runtime::block_on`] (a thread-local current-thread tokio
//! runtime). All `await` points happen BEFORE any datum allocation — by the
//! time we touch `slot->tts_values` we hold a fully materialised
//! `RecordBatch` and never re-enter the runtime until the next row request.

use crate::azure::{build_credential, AzureBlobClient, AzureBlobReader};
use crate::convert::arrow_to_pg::arrow_value_to_datum;
use crate::error::{raise, FdwError, FdwResult};
use crate::fdw::options::{
    parse_server_options_from_slice, parse_table_options_from_slice,
    parse_user_mapping_options_from_slice, validate_combo, ServerOptions, TableOptions,
    UserMappingOptions,
};
use crate::fdw::pushdown::{build_row_filter, PushedExpr};
use crate::fdw::pushdown_walk::walk_quals;
use crate::parquet_io::reader::{open_stream, ParquetReadOptions};
use crate::runtime;

use arrow::array::RecordBatch;
use futures::StreamExt;
use parquet::arrow::async_reader::{ParquetRecordBatchStream, ParquetRecordBatchStreamBuilder};
use parquet::arrow::ProjectionMask;
use pgrx::pg_sys;
use std::ffi::{c_int, c_void, CStr};

/// Per-scan executor state, owned by PG via `Box::into_raw` /
/// `Box::from_raw` round-trip through `ForeignScanState.fdw_state`.
pub struct ScanState {
    /// Container-scoped Azure client (cheap to clone).
    client: AzureBlobClient,
    /// Blob (name, etag) pairs enumerated from the foreign table's `filename`
    /// option (with optional `*` prefix-glob expanded server-side). The etag
    /// is captured at scan time and threaded through the modify path so a
    /// subsequent UPDATE/DELETE preconditions its GET/PUT on the same etag —
    /// any concurrent writer trips SQLSTATE 40001.
    blobs: Vec<(String, String)>,
    /// Index into `blobs` of the blob currently being streamed.
    cur_blob: usize,
    /// Active parquet record-batch stream (if any).
    cur_stream: Option<ParquetRecordBatchStream<AzureBlobReader>>,
    /// Last batch fetched off `cur_stream`; we walk it row by row.
    cur_batch: Option<RecordBatch>,
    /// Next row index to emit within `cur_batch`.
    cur_row: usize,
    /// PG attribute OIDs in TupleDesc order. Used to look up the target
    /// type when materialising each Datum.
    pg_oids: Vec<pg_sys::Oid>,
    /// Top-level parquet column indices we read. Populated from
    /// `baserel->reltarget` at planner time and threaded through
    /// `fdw_private`. Empty list → unset (legacy path); but in practice
    /// the planner always supplies at least one column (count(*) is
    /// rewritten to project col 0 in `get_foreign_plan`).
    projection: Option<Vec<usize>>,
    /// Parallel to the projected RecordBatch's columns: `attno_map[arrow_batch_col_idx]`
    /// is the 0-based relation attnum that arrow column receives. Iterate
    /// walks this to fill the slot at the right positions; relation columns
    /// NOT present in the map get `tts_isnull = true`. Same length as
    /// `projection` when projection is `Some`.
    attno_map: Vec<usize>,
    /// SP-0 seam: how to build a parquet `RowFilter` for the next blob.
    /// SP-1 swaps the impl from `PushedExprFilter` (the SP-0 default) to a
    /// pushdown-aware one. Stays `None` only on the no-pushdown path so
    /// `next_row` can skip the build call entirely.
    qual_filter: Option<Box<dyn QualFilter>>,
    /// SP-0 seam: how to discover the next blob to open. SP-3 (parallel
    /// scan) swaps the impl to a DSM-cursor-backed one.
    range_producer: Box<dyn RangeProducer>,
    /// Cached parsed table options (for re-opening streams on `ReScan`).
    table_opts: TableOptions,
    /// Per-blob chunk table. Grows as blobs are opened; entries are appended
    /// in scan order so `blob_id` indexes are stable across the whole scan.
    blob_id_table: Vec<crate::fdw::modify::BlobIdEntry>,
    /// Base `blob_id` of the source blob currently being streamed (the first
    /// chunk's index in `blob_id_table`).
    cur_blob_base_id: u32,
    /// Absolute row index within the *current source blob* (not the chunk).
    cur_row_in_blob: u64,
    /// 0-based attnums of partition cols. Empty when partition_columns
    /// is unset.
    pub(crate) partition_attnums: Vec<usize>,
    /// Declaration list (name + type) in declared order; mirrors
    /// `table_opts.partition_keys` but cached for hot-path access.
    pub(crate) partition_keys_decl: Vec<(String, crate::fdw::options::PgPartitionType)>,
    /// For each foreign-table attno (0-based): Some(p) if it's a storage
    /// column at parquet index p; None if it's a partition column.
    #[allow(dead_code)]
    pub(crate) storage_attno_to_parquet_idx: Vec<Option<usize>>,
    /// Parsed + cast partition datums for the currently-open blob.
    /// `Vec<Datum>` indexed in declared partition order (mirrors
    /// `partition_attnums` order). Empty when no blob is open.
    pub(crate) partition_datums_for_current_blob: Vec<pg_sys::Datum>,
    /// SP-3c: K-way merge stream when sorted mode is active; mutually
    /// exclusive with cur_stream's per-blob iteration. `None` when sorted
    /// mode is OFF (the common case).
    pub(crate) sorted_stream:
        Option<crate::parquet_io::multifile::MultiFileSortedStream<crate::azure::AzureBlobReader>>,
}

impl ScanState {
    /// Borrow the leader's `(name, etag)` blob list. Used by the parallel
    /// DSM callbacks to size and populate the shared segment.
    pub(crate) fn blobs_for_dsm(&self) -> &[(String, String)] {
        &self.blobs
    }
}

// ---------- public callbacks ------------------------------------------------

/// `GetForeignRelSize_function` — naive v1 cost estimate.
///
/// We don't HEAD the blob (would require an async round-trip in planner
/// context); the planner gets a sane 1000-row default with 32-byte tuples.
/// A future task can refine this by consulting parquet metadata.
///
/// # Safety
///
/// PG passes a valid `*mut RelOptInfo` to a registered FDW callback.
pub unsafe extern "C-unwind" fn get_foreign_rel_size(
    _root: *mut pg_sys::PlannerInfo,
    baserel: *mut pg_sys::RelOptInfo,
    _foreigntableid: pg_sys::Oid,
) {
    // SAFETY: PG guarantees `baserel` points to a live `RelOptInfo` for the
    // duration of the planner callback; its `reltarget` is also live.
    unsafe {
        (*baserel).rows = 1000.0;
        if !(*baserel).reltarget.is_null() {
            (*(*baserel).reltarget).width = 32;
        }
    }
}

/// Build the `pathkeys` list advertising the sorted-mode output ordering.
///
/// SP-3c v1 deliberately returns `std::ptr::null_mut()` (no pathkeys). The
/// `MultiFileSortedStream` installed at scan-begin (see `build_state`) still
/// produces globally-sorted output, so results are CORRECT; PG simply adds a
/// redundant `Sort` node above the foreign scan when a downstream operator
/// needs the ordering, losing only the optimization (not correctness).
///
/// Emitting real pathkeys requires constructing, per sort column, a `Var` +
/// an `EquivalenceClass` (`get_eclass_for_sort_expr`) + a canonical `PathKey`
/// (`make_canonical_pathkey`). Both helpers are exposed by pgrx 0.18.1 but
/// their signatures drift across PG14–18: `get_eclass_for_sort_expr` carries
/// an extra `nullable_relids: Relids` argument on pg14/pg15 (dropped in pg16+),
/// and `make_canonical_pathkey` takes `strategy: c_int` on pg14–17 but
/// `cmptype: CompareType` on pg18. Each also needs per-column btree opfamily /
/// opcintype / sort-operator resolution. That cross-version FFI is deferred to
/// a future SP; the `pathkeys` arg is wired through `create_foreignscan_path`
/// now so the follow-up is a one-function change with no call-site churn.
///
/// # Safety
/// `root` / `baserel` are valid planner pointers; `foreigntableid` is a valid
/// foreign-table OID. Returning null is always sound.
unsafe fn build_sorted_pathkeys(
    _root: *mut pg_sys::PlannerInfo,
    _baserel: *mut pg_sys::RelOptInfo,
    _foreigntableid: pg_sys::Oid,
) -> *mut pg_sys::List {
    // SP-3c v1 fallback: no pathkeys (documented above). The merge stream
    // already guarantees correct ordering; PG adds a redundant Sort if needed.
    std::ptr::null_mut()
}

/// `GetForeignPaths_function` — add a single un-parameterised path.
///
/// # Safety
///
/// `root` / `baserel` are valid planner pointers; `create_foreignscan_path`
/// is the documented entry point for FDWs to register a scan path.
pub unsafe extern "C-unwind" fn get_foreign_paths(
    root: *mut pg_sys::PlannerInfo,
    baserel: *mut pg_sys::RelOptInfo,
    foreigntableid: pg_sys::Oid,
) {
    // SAFETY: PG-supplied pointers are valid for the duration of the call.
    // We pass `null_mut` for every optional list / outer-path argument: a
    // simple base-rel scan with no parameterisation, no extra pathkeys, no
    // fdw_private payload (v1 — pushdown is stubbed).
    //
    // The `create_foreignscan_path` signature changes across PG versions:
    //   * pg14, pg15, pg16: 10 args.
    //   * pg17:             11 args (adds `fdw_restrictinfo`).
    //   * pg18:             12 args (adds `disabled_nodes` after `rows`).
    // We cfg-gate each call so a build under any one pgrx feature compiles
    // against exactly the right C signature.
    unsafe {
        // SP-3c: when sorted mode is active, the scan-begin branch installs a
        // K-way `MultiFileSortedStream` that yields rows globally ordered by
        // the `sorted` columns (ASC, NULLS LAST). We advertise that ordering
        // to the planner via the `pathkeys` arg below so a downstream
        // ORDER BY / merge-join can skip a redundant Sort.
        let pathkeys: *mut pg_sys::List = build_sorted_pathkeys(root, baserel, foreigntableid);

        #[cfg(any(feature = "pg14", feature = "pg15", feature = "pg16"))]
        let path = pg_sys::create_foreignscan_path(
            root,
            baserel,
            std::ptr::null_mut(), // target = default (baserel->reltarget)
            (*baserel).rows,
            0.0,                  // startup_cost
            (*baserel).rows,      // total_cost — placeholder
            pathkeys,             // pathkeys (SP-3c: sorted mode)
            std::ptr::null_mut(), // required_outer
            std::ptr::null_mut(), // fdw_outerpath
            std::ptr::null_mut(), // fdw_private
        );
        #[cfg(feature = "pg17")]
        let path = pg_sys::create_foreignscan_path(
            root,
            baserel,
            std::ptr::null_mut(), // target = default (baserel->reltarget)
            (*baserel).rows,
            0.0,                  // startup_cost
            (*baserel).rows,      // total_cost — placeholder
            pathkeys,             // pathkeys (SP-3c: sorted mode)
            std::ptr::null_mut(), // required_outer
            std::ptr::null_mut(), // fdw_outerpath
            std::ptr::null_mut(), // fdw_restrictinfo (pg17+)
            std::ptr::null_mut(), // fdw_private
        );
        #[cfg(feature = "pg18")]
        let path = pg_sys::create_foreignscan_path(
            root,
            baserel,
            std::ptr::null_mut(), // target = default (baserel->reltarget)
            (*baserel).rows,
            0,                    // disabled_nodes (pg18+) — none disabled
            0.0,                  // startup_cost
            (*baserel).rows,      // total_cost — placeholder
            pathkeys,             // pathkeys (SP-3c: sorted mode)
            std::ptr::null_mut(), // required_outer
            std::ptr::null_mut(), // fdw_outerpath
            std::ptr::null_mut(), // fdw_restrictinfo
            std::ptr::null_mut(), // fdw_private
        );
        pg_sys::add_path(baserel, path as *mut pg_sys::Path);

        // Partial path for parallel execution. PG only generates a Gather
        // over a base rel when the FDW publishes a `parallel_safe = true`
        // *partial* path via `add_partial_path` AND sets
        // `baserel->consider_parallel = true`. The 5 parallel FDW callbacks
        // wired into `FdwRoutine` are dead code without this. SELECT-only:
        // `is_foreign_scan_parallel_safe` already rejects non-SELECT and
        // `parallel_workers = 0`, but we gate here too so we don't pollute
        // the planner with partial paths it will never select.
        //
        // SP-3c: sorted mode also forbids partial paths — the single-
        // coordinator K-way heap merge cannot be split across workers, and
        // `is_foreign_scan_parallel_safe` already returns false for it. We
        // gate here too so the planner never sees a partial path for a sorted
        // scan (which would otherwise contradict the emitted pathkeys).
        let cmd_is_select = !root.is_null()
            && !(*root).parse.is_null()
            && (*(*root).parse).commandType == pg_sys::CmdType::CMD_SELECT;
        if cmd_is_select && !crate::fdw::parallel::read_sorted_opt(foreigntableid) {
            let pw_opt = crate::fdw::parallel::read_parallel_workers_opt(foreigntableid);
            if pw_opt != Some(0) {
                // When the table option is absent, use the cluster GUC
                // `max_parallel_workers_per_gather` (PG global int) so we
                // honor the operator's parallelism cap instead of a magic 4.
                // SAFETY (subsumed by enclosing `unsafe`): reading a
                // documented static PG int.
                let cluster_cap = pg_sys::max_parallel_workers_per_gather;
                let n_workers: i32 = pw_opt.unwrap_or(cluster_cap).max(1);
                (*baserel).consider_parallel = true;
                // Divide the per-worker cost so the planner's Gather wrapping
                // beats the sequential path. PG itself does NOT divide the
                // partial path's cost by `parallel_workers` for FDWs — that
                // responsibility is on the FDW. Without this the partial and
                // sequential paths tie on cost and `add_path` keeps the
                // sequential one (first-added wins on ties), so no Gather is
                // ever chosen.
                let per_worker_rows = ((*baserel).rows / n_workers as f64).max(1.0);
                let per_worker_cost = ((*baserel).rows / n_workers as f64).max(1.0);

                #[cfg(any(feature = "pg14", feature = "pg15", feature = "pg16"))]
                let ppath = pg_sys::create_foreignscan_path(
                    root,
                    baserel,
                    std::ptr::null_mut(), // target = default
                    per_worker_rows,
                    0.0,                  // startup_cost
                    per_worker_cost,      // total_cost
                    std::ptr::null_mut(), // pathkeys
                    std::ptr::null_mut(), // required_outer
                    std::ptr::null_mut(), // fdw_outerpath
                    std::ptr::null_mut(), // fdw_private
                );
                #[cfg(feature = "pg17")]
                let ppath = pg_sys::create_foreignscan_path(
                    root,
                    baserel,
                    std::ptr::null_mut(),
                    per_worker_rows,
                    0.0,
                    per_worker_cost,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(), // fdw_restrictinfo (pg17+)
                    std::ptr::null_mut(),
                );
                #[cfg(feature = "pg18")]
                let ppath = pg_sys::create_foreignscan_path(
                    root,
                    baserel,
                    std::ptr::null_mut(),
                    per_worker_rows,
                    0, // disabled_nodes (pg18+)
                    0.0,
                    per_worker_cost,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                );
                (*ppath).path.parallel_aware = true;
                (*ppath).path.parallel_safe = true;
                (*ppath).path.parallel_workers = n_workers;
                (*ppath).path.rows = per_worker_rows;
                pg_sys::add_partial_path(baserel, ppath as *mut pg_sys::Path);
            }
        }
    }
}

/// `GetForeignPlan_function` — wrap the chosen path into a `ForeignScan`.
///
/// Also captures `baserel->reltarget->exprs` here and encodes the projected
/// relation attnums as an `IntList` into `fdw_private`. `BeginForeignScan`
/// later decodes that list to set `ParquetReadOptions::projection`, so a
/// `SELECT one_col FROM t` only fetches the one column from each parquet
/// blob. See `docs/superpowers/specs/2026-06-20-phaseC-read-path-projection-design.md`.
///
/// # Safety
///
/// PG owns the input lists and pointers; we only call documented planner
/// helpers (`extract_actual_clauses`, `make_foreignscan`, `pull_varattnos`,
/// `bms_next_member`, `lappend_int`). Passing the `scan_clauses` through
/// `extract_actual_clauses(_, false)` strips `RestrictInfo` wrappers as
/// required by `make_foreignscan`.
pub unsafe extern "C-unwind" fn get_foreign_plan(
    root: *mut pg_sys::PlannerInfo,
    baserel: *mut pg_sys::RelOptInfo,
    _foreigntableid: pg_sys::Oid,
    _best_path: *mut pg_sys::ForeignPath,
    tlist: *mut pg_sys::List,
    scan_clauses: *mut pg_sys::List,
    outer_plan: *mut pg_sys::Plan,
) -> *mut pg_sys::ForeignScan {
    // SAFETY: `baserel`, `scan_clauses`, `tlist`, `outer_plan` are valid PG
    // planner pointers; `relid` is the executor scanrelid (an `Index`).
    unsafe {
        let scan_clauses = pg_sys::extract_actual_clauses(scan_clauses, false);
        // Only emit a projection for SELECT. For UPDATE/DELETE, the
        // planner's `reltarget->exprs` doesn't reliably enumerate every
        // column the modify pipeline needs (e.g. plain
        // `UPDATE t SET col=const WHERE id=k` may leave columns out
        // because the SET expression is constant). Projecting too narrow
        // there silently breaks the rewrite. The savings would be
        // negligible anyway — `commit_plan` reads the whole blob on a
        // separate I/O path. NULL fdw_private is interpreted in
        // `build_state` as "project every column".
        let fdw_private = if !root.is_null()
            && !(*root).parse.is_null()
            && (*(*root).parse).commandType == pg_sys::CmdType::CMD_SELECT
        {
            build_projection_intlist(baserel)
        } else {
            std::ptr::null_mut()
        };
        pg_sys::make_foreignscan(
            tlist,
            scan_clauses,
            (*baserel).relid,
            std::ptr::null_mut(), // fdw_exprs
            fdw_private,
            std::ptr::null_mut(), // fdw_scan_tlist
            std::ptr::null_mut(), // fdw_recheck_quals
            outer_plan,
        )
    }
}

/// Walk `baserel->reltarget->exprs`, collect every `Var.varattno` referenced
/// at this relation, and encode them (1-based AttrNumbers) as a PG `IntList`
/// to ride along on the `ForeignScan` node's `fdw_private` slot.
///
/// `pull_varattnos` stores attnos in the bitmap offset by
/// `-FirstLowInvalidHeapAttributeNumber` so system columns (ctid = -1,
/// xmin = -3, etc.) can sit at non-negative indices. We subtract the offset
/// back out and drop any system-column attnums (< 1) — we never project
/// those into the parquet read.
///
/// The count(*) edge case (no Vars at all) projects column 0 (attnum 1)
/// instead of an empty list so the parquet reader always has at least one
/// column to row-iterate over.
///
/// # Safety
///
/// `baserel` is a live `RelOptInfo` for the duration of the planner
/// callback; `reltarget` and `reltarget->exprs` are live. The `Bitmapset`
/// returned by `pull_varattnos` is palloc'd in the planner's memory
/// context. The returned `*mut List` is palloc'd by `lappend_int`.
unsafe fn build_projection_intlist(baserel: *mut pg_sys::RelOptInfo) -> *mut pg_sys::List {
    // SAFETY: see fn-level doc.
    unsafe {
        if baserel.is_null() || (*baserel).reltarget.is_null() {
            // Pathological — give the executor an empty list and let
            // `build_state` fall back to projecting column 0.
            return std::ptr::null_mut();
        }
        let relid: pg_sys::Index = (*baserel).relid;
        let offset: i32 = pg_sys::FirstLowInvalidHeapAttributeNumber;

        let mut bms: *mut pg_sys::Bitmapset = std::ptr::null_mut();

        // (1) Columns named in the scan's output target list (the SELECT
        // projection plus anything the planner promoted up from above).
        let exprs = (*(*baserel).reltarget).exprs;
        pg_sys::pull_varattnos(exprs as *mut pg_sys::Node, relid, &mut bms);

        // (2) Columns referenced only by quals applied AT the scan level
        // (`baserestrictinfo`). reltarget.exprs does NOT include these on
        // its own — without walking them too, `SELECT name FROM t WHERE
        // id = 1` would project only [name] and leave `id` NULL, breaking
        // the qual evaluation above the scan.
        let ri_list = (*baserel).baserestrictinfo;
        if !ri_list.is_null() {
            let n = pg_sys::list_length(ri_list);
            for i in 0..n {
                let cell = pg_sys::list_nth(ri_list, i) as *mut pg_sys::RestrictInfo;
                if cell.is_null() {
                    continue;
                }
                let clause = (*cell).clause;
                if !clause.is_null() {
                    pg_sys::pull_varattnos(clause as *mut pg_sys::Node, relid, &mut bms);
                }
            }
        }

        let mut attnums: Vec<i32> = Vec::new();
        let mut prev: i32 = -1;
        loop {
            let next = pg_sys::bms_next_member(bms, prev);
            if next < 0 {
                break;
            }
            let attno = next + offset;
            // Skip system columns (attno < 1) — we don't project those
            // into the parquet read.
            if attno >= 1 {
                attnums.push(attno);
            }
            prev = next;
        }

        // count(*) / WHERE-only-on-system-cols edge case: project column 0
        // (attnum 1) so the parquet reader always has at least one column
        // to iterate.
        if attnums.is_empty() {
            attnums.push(1);
        }

        // Build a PG `IntList`. lappend_int starts a fresh list when the
        // input is NIL; subsequent calls append in place.
        let mut list: *mut pg_sys::List = std::ptr::null_mut();
        for a in &attnums {
            list = pg_sys::lappend_int(list, *a);
        }
        list
    }
}

/// `BeginForeignScan_function` — parse options, open Azure client, expand
/// glob, box state.
///
/// # Safety
///
/// `node` is a live `ForeignScanState`; its `ss.ss_currentRelation` is the
/// foreign relation being scanned. We never touch `node` after storing the
/// state pointer except through the documented `fdw_state` slot.
pub unsafe extern "C-unwind" fn begin_foreign_scan(
    node: *mut pg_sys::ForeignScanState,
    eflags: c_int,
) {
    // Skip work when PG is only running EXPLAIN (no actual scan happens, so
    // we don't need an Azure connection — keeps EXPLAIN free of network IO
    // and credential errors).
    if (eflags as u32) & pg_sys::EXEC_FLAG_EXPLAIN_ONLY != 0 {
        return;
    }

    // Skip work when running in a parallel worker. PG calls
    // BeginForeignScan in each worker BEFORE InitializeWorkerForeignScan,
    // which is the actual worker-side setup entry point. Returning here
    // avoids:
    //   - allocating a doomed `Box<ScanState>` that the worker init
    //     overwrites and leaks
    //   - the redundant blob LIST round-trip per worker
    //   - publishing to scan_handoff in a worker (CLAUDE.md invariant:
    //     workers never publish)
    // SAFETY: `ParallelWorkerNumber` is a documented PG global int — `-1` in
    // the leader, `0..` in workers. Reading a static int through a raw deref
    // is sound from any FFI callback. `IsParallelWorker()` is not exposed in
    // pgrx 0.18.1 bindings; this is the equivalent check.
    if unsafe { pg_sys::ParallelWorkerNumber } >= 0 {
        return;
    }

    // SAFETY: PG guarantees `node` and its `ss.ss_currentRelation` are valid
    // during executor startup.
    let rel = unsafe { (*node).ss.ss_currentRelation };
    // SAFETY: relation pointer is live; `rd_id` is its OID.
    let relid = unsafe { (*rel).rd_id };

    let state = match unsafe {
        // SAFETY: `node` is live (see above); `relid` was extracted from
        // its currentRelation, so `build_state`'s contract is satisfied.
        build_state(node, relid)
    } {
        Ok(s) => s,
        Err(e) => raise(e),
    };

    // SAFETY: store the boxed state into the executor's per-scan slot.
    // `EndForeignScan` will reclaim it via `Box::from_raw`.
    unsafe {
        (*node).fdw_state = Box::into_raw(Box::new(state)) as *mut c_void;
    }
}

/// `IterateForeignScan_function` — emit one tuple per call, or an empty
/// slot when all blobs are exhausted.
///
/// # Safety
///
/// `node->fdw_state` was populated by `begin_foreign_scan` and is still
/// alive (PG won't call `end_foreign_scan` before the last `iterate`).
/// `ss_ScanTupleSlot` is owned by the executor and has `tts_values` /
/// `tts_isnull` arrays sized to the relation's TupleDesc.
pub unsafe extern "C-unwind" fn iterate_foreign_scan(
    node: *mut pg_sys::ForeignScanState,
) -> *mut pg_sys::TupleTableSlot {
    // SAFETY: see fn-level safety; `fdw_state` is non-null between begin/end.
    let state: &mut ScanState = unsafe { &mut *((*node).fdw_state as *mut ScanState) };
    // SAFETY: `ss_ScanTupleSlot` is set up by the executor before iterate.
    let slot = unsafe { (*node).ss.ss_ScanTupleSlot };
    // SAFETY: `ExecClearTuple` resets the slot to empty; pgrx exposes the
    // shim symbol with a matching signature.
    unsafe {
        pg_sys::ExecClearTuple(slot);
    }

    match next_row(state) {
        Ok(Some((batch, row))) => {
            // SP-3c C2: in sorted mode `next_row` drives the K-way merge and
            // returns BEFORE any `blob_id_table.push`, so `blob_id_table`
            // stays EMPTY while `cur_row_in_blob` keeps incrementing. The
            // (non-sorted) chunk-registration path below does
            // `blob_id_table[base_idx]` at every CHUNK_ROWS (65_536) boundary;
            // on an empty Vec that is an out-of-bounds index → panic across
            // the `extern "C-unwind"` FFI boundary → backend crash. Any sorted
            // SELECT returning > 65_536 rows hit this. The synthetic ctid is
            // also meaningless in sorted mode (rows don't map to a single
            // blob_id, and SP-3c is SELECT-only so no modify path reads it).
            // So we SKIP both chunk registration AND ctid stamping for the
            // sorted path, leaving `tts_tid` at its default.
            let sorted_mode = state.sorted_stream.is_some();
            if !sorted_mode {
                // Lazy chunk registration: see `register_chunk_if_boundary`.
                register_chunk_if_boundary(
                    &mut state.blob_id_table,
                    state.cur_blob_base_id,
                    state.cur_row_in_blob,
                );
            }
            // SAFETY: `tts_values` / `tts_isnull` are arrays of length
            // `tupdesc->natts`. When `attno_map` is a strict subset of
            // the relation's columns (Phase C projection), we NULL the
            // whole slot first then overlay the projected positions;
            // unprojected columns stay NULL because PG won't read them.
            // When `attno_map` covers every column (UPDATE/DELETE, or
            // SELECT * with all-column reltarget) we skip the NULL pass
            // and write each slot position directly — preserving the
            // pre-Phase-C bit-for-bit semantics on those paths.
            //
            // The arrow batch has `attno_map.len()` columns; `batch.column(k)`
            // carries the value of relation column `attno_map[k]` (0-based
            // attnum). See `build_state` for the matching invariant.
            unsafe {
                let values = (*slot).tts_values;
                let isnulls = (*slot).tts_isnull;
                let natts = state.pg_oids.len();
                let projection_is_subset = state.attno_map.len() < natts;
                if projection_is_subset {
                    for k in 0..natts {
                        *values.add(k) = pg_sys::Datum::null();
                        *isnulls.add(k) = true;
                    }
                }
                for (arrow_col, &rel_col) in state.attno_map.iter().enumerate() {
                    if rel_col >= natts {
                        raise(FdwError::SchemaMismatch(format!(
                            "attno_map[{arrow_col}] = {rel_col} >= natts {natts}"
                        )));
                    }
                    let arr = batch.column(arrow_col).as_ref();
                    let oid = state.pg_oids[rel_col];
                    match arrow_value_to_datum(arr, row, oid) {
                        Ok(Some(d)) => {
                            *values.add(rel_col) = d;
                            *isnulls.add(rel_col) = false;
                        }
                        Ok(None) => {
                            *values.add(rel_col) = pg_sys::Datum::null();
                            *isnulls.add(rel_col) = true;
                        }
                        Err(e) => raise(e),
                    }
                }
                // Inject partition virtual columns from the per-blob cache.
                // Done AFTER the arrow walk so partition cols win for any
                // accidental overlap with attno_map (should be empty by
                // construction — partition attnums are disjoint from storage
                // attnums — but be defensive).
                for (i, &attno) in state.partition_attnums.iter().enumerate() {
                    if attno >= natts {
                        continue;
                    }
                    if i >= state.partition_datums_for_current_blob.len() {
                        // No cached datum (shouldn't happen if a blob was
                        // opened; defensive).
                        continue;
                    }
                    *values.add(attno) = state.partition_datums_for_current_blob[i];
                    *isnulls.add(attno) = false;
                }
                // SAFETY (still inside the same `unsafe` block as the
                // `tts_values`/`tts_isnull` writes above): `slot` is a live
                // `TupleTableSlot` owned by the executor for the lifetime
                // of this callback; `tts_tid` is a plain struct field that
                // FDW callbacks are expected to write per PG's executor
                // contract — `ExecForeignUpdate`/`ExecForeignDelete` will
                // decode it back into a `RowId`.
                //
                // SP-3c C2: skip ctid stamping in sorted mode — the synthetic
                // ctid is meaningless there (sorted rows don't map to a single
                // blob_id) and SP-3c is SELECT-only so nothing reads it. Leave
                // `tts_tid` at its default.
                if !sorted_mode {
                    let rid = crate::fdw::modify::rowid::RowId::from_absolute(
                        state.cur_blob_base_id,
                        state.cur_row_in_blob,
                    );
                    (*slot).tts_tid = rid.to_ctid();
                }
                pg_sys::ExecStoreVirtualTuple(slot);
            }
            state.cur_row_in_blob += 1;
            slot
        }
        Ok(None) => {
            // End of data — cleared slot already signals EOF.
            slot
        }
        Err(e) => raise(e),
    }
}

/// `ReScanForeignScan_function` — rewind to the first blob, drop any
/// in-flight stream.
///
/// Also re-publishes the scan-time blob list (with etags) into the
/// `scan_handoff` so a modify that runs after a rescan still observes the
/// scan's pinned snapshot. Without this, `begin_foreign_modify` could see a
/// stale handoff entry (or, after pop, fall through to the unguarded
/// fallback in test builds — or hard-error in release).
///
/// # Safety
///
/// `node->fdw_state` is alive (PG only calls rescan between begin/end).
pub unsafe extern "C-unwind" fn re_scan_foreign_scan(node: *mut pg_sys::ForeignScanState) {
    // NOTE: publishing to scan_handoff here is safe in BOTH sequential and
    // parallel paths: (a) parallel SELECTs are never followed by a modify
    // (the parallel-safe gate restricts parallel scans to SELECT), and
    // (b) the leader is where a modify WOULD land if one did. Workers
    // re-entering rescan never reach this function — the worker setup path
    // is `initialize_worker_foreign_scan`, not `begin_foreign_scan`/this.
    // SAFETY: fdw_state was set by begin_foreign_scan and not yet dropped.
    let state: &mut ScanState = unsafe { &mut *((*node).fdw_state as *mut ScanState) };
    state.cur_stream = None;
    state.cur_batch = None;
    state.cur_row = 0;
    state.cur_blob = 0;
    state.blob_id_table.clear();
    state.cur_blob_base_id = 0;
    state.cur_row_in_blob = 0;
    state.range_producer = Box::new(SequentialRanges {
        blobs: state.blobs.clone(),
        cursor: 0,
    });

    // SP-3c C1: the K-way `MultiFileSortedStream` is SINGLE-PASS — after the
    // first scan its N underlying parquet streams are drained. A rescan (e.g.
    // a sorted foreign table on the inner side of a nested-loop join) would
    // otherwise hit the sorted branch in `next_row` with an exhausted stream
    // and return `Ok(None)` immediately, yielding ZERO rows on every rescan
    // (silent wrong results). So when sorted mode is active we REBUILD the
    // merge stream from scratch here — re-opening the N parquet streams and
    // rebuilding the heap so the second scan returns the full sorted result
    // again. All inputs come from the ScanState fields stamped at build time.
    // The identity projection/attno_map were set in `build_state` and are
    // unchanged across rescans, so we don't touch them here.
    if state.sorted_stream.is_some() {
        // SAFETY: `node` is the live ForeignScanState (PG only calls rescan
        // between begin/end); `build_sorted_stream_if_active` only reads its
        // currentRelation TupleDesc.
        let rebuilt = unsafe {
            build_sorted_stream_if_active(
                node,
                &state.client,
                &state.blobs,
                &state.table_opts,
                &state.partition_attnums,
                &state.storage_attno_to_parquet_idx,
                &state.pg_oids,
            )
        };
        match rebuilt {
            Ok(s) => state.sorted_stream = s,
            Err(e) => raise(e),
        }
    }

    // SAFETY: `ss_currentRelation` is live for the duration of the rescan
    // callback — same invariant `end_foreign_scan` relies on.
    unsafe {
        let rel = (*node).ss.ss_currentRelation;
        if !rel.is_null() {
            let relid = (*rel).rd_id;
            // Clear any previous handoff entry for this relid so we don't
            // stack a stale duplicate on top, then republish the fresh list.
            crate::fdw::modify::scan_handoff::discard(relid);
            crate::fdw::modify::scan_handoff::publish(relid, state.blobs.clone());
        }
    }
}

/// `EndForeignScan_function` — reclaim and drop the boxed state.
///
/// # Safety
///
/// We pair this with the `Box::into_raw` performed by `begin_foreign_scan`.
/// PG calls `end_foreign_scan` exactly once per `begin_foreign_scan`.
pub unsafe extern "C-unwind" fn end_foreign_scan(node: *mut pg_sys::ForeignScanState) {
    // SAFETY: see fn-level safety. After this drop, the `fdw_state` pointer
    // is dangling; we null it out to make any accidental reuse crash loudly.
    unsafe {
        let p = (*node).fdw_state as *mut ScanState;
        if !p.is_null() {
            // Discard any handoff that the modify path didn't consume — e.g.
            // a SELECT without an UPDATE. Keeps thread-local state tidy
            // across statements in the same backend.
            let rel = (*node).ss.ss_currentRelation;
            if !rel.is_null() {
                let relid = (*rel).rd_id;
                crate::fdw::modify::scan_handoff::discard(relid);
            }
            drop(Box::from_raw(p));
            (*node).fdw_state = std::ptr::null_mut();
        }
    }
}

// ---------- internals -------------------------------------------------------

/// Shared phase-1 setup result: every piece [`ScanState`] needs EXCEPT the
/// `RangeProducer` (which differs between sequential and parallel entry
/// points) AND without publishing the `scan_handoff` (which only the
/// sequential entry point does).
pub(crate) struct ScanStateCore {
    pub(crate) client: AzureBlobClient,
    pub(crate) blobs: Vec<(String, String)>,
    pub(crate) pg_oids: Vec<pg_sys::Oid>,
    pub(crate) projection: Option<Vec<usize>>,
    pub(crate) attno_map: Vec<usize>,
    pub(crate) qual_filter: Option<Box<dyn QualFilter>>,
    pub(crate) table_opts: TableOptions,
    pub(crate) partition_attnums: Vec<usize>,
    pub(crate) partition_keys_decl: Vec<(String, crate::fdw::options::PgPartitionType)>,
    pub(crate) storage_attno_to_parquet_idx: Vec<Option<usize>>,
}

/// Internal shared-setup helper for `ScanState` construction. See
/// [`ScanStateCore`]. Does not publish to `scan_handoff` and does not build
/// a [`RangeProducer`].
///
/// # Safety
/// Same contract as [`build_state`].
unsafe fn build_scan_state_core(
    node: *mut pg_sys::ForeignScanState,
    relid: pg_sys::Oid,
) -> FdwResult<ScanStateCore> {
    // --- options off the catalog ---------------------------------------
    // SAFETY: documented PG catalog accessors; results are palloc'd and
    // owned by the current memory context.
    let (server_opts, um_opts, table_opts) = unsafe { read_all_options(relid) }?;
    validate_combo(&server_opts, &um_opts)?;

    // --- credential + client -------------------------------------------
    let cred = build_credential(
        &server_opts.auth_method,
        &server_opts.account_name,
        um_opts.sas_url.as_deref(),
    )?;
    let client = AzureBlobClient::new(
        &server_opts.endpoint,
        &server_opts.account_name,
        cred,
        &table_opts.container,
    )?;

    // --- glob expansion ------------------------------------------------
    let plan_source: Box<dyn PlanSource> =
        if !table_opts.filename.contains('*') && !table_opts.filename.contains('?') {
            Box::new(HeadEtagSource {
                name: table_opts.filename.clone(),
            })
        } else {
            Box::new(GlobSource {
                glob: crate::fdw::glob::parse_glob(&table_opts.filename)?,
            })
        };
    let blobs = plan_source.list(&client)?;

    // NOTE: publishing the (relid -> blobs) handoff is DEFERRED until the
    // very end of this function (after all fallible work). The previous
    // ordering (publish here, before the attribute-OIDs + projection
    // validation) leaked the handoff entry across statements: if the
    // dropped-column rejection or projection-out-of-range branch returned
    // Err, raise() longjmps and end_foreign_scan never sees an
    // initialised fdw_state, so discard() is skipped and the entry
    // stranded in the thread-local stack for the lifetime of the backend.

    // --- attribute OIDs in tupdesc order -------------------------------
    // SAFETY: ss_currentRelation is live; rd_att is its TupleDesc. We use
    // the version-portable `crate::fdw::tupdesc_attr` accessor instead of
    // poking the `attrs` field directly because pg18 replaced it with
    // `compact_attrs`.
    let pg_oids = unsafe {
        let rel = (*node).ss.ss_currentRelation;
        let tupdesc = (*rel).rd_att;
        let natts = (*tupdesc).natts as usize;
        let mut out = Vec::with_capacity(natts);
        for i in 0..natts {
            let att = crate::fdw::tupdesc_attr(tupdesc, i);
            if (*att).attisdropped {
                // We don't yet support dropped columns; surface a clear error
                // rather than silently emitting the wrong column count.
                return Err(FdwError::SchemaMismatch(format!(
                    "dropped column at attnum {} not supported",
                    (*att).attnum
                )));
            }
            out.push((*att).atttypid);
        }
        out
    };

    // --- partition column resolution ----------------------------------
    // Walk the tupdesc once to map declared partition column names →
    // 0-based attnums (case-insensitive). Then build the
    // `storage_attno_to_parquet_idx` map.
    let partition_attnums: Vec<usize> = if table_opts.partition_columns.is_empty() {
        Vec::new()
    } else {
        // SAFETY: ss_currentRelation is live; rd_att is its TupleDesc; we
        // reuse the same `tupdesc_attr` accessor as the pg_oids walk above.
        unsafe {
            let rel = (*node).ss.ss_currentRelation;
            let tupdesc = (*rel).rd_att;
            let natts = (*tupdesc).natts as usize;
            let mut out = Vec::with_capacity(table_opts.partition_columns.len());
            for name in &table_opts.partition_columns {
                let mut found: Option<usize> = None;
                for i in 0..natts {
                    let att = crate::fdw::tupdesc_attr(tupdesc, i);
                    // SAFETY: `attname` is a fixed-length `NameData` whose
                    // `.data` is a NUL-terminated C string within bounds —
                    // same access pattern as the pg_oids walk above.
                    let nm =
                        CStr::from_ptr((*att).attname.data.as_ptr() as *const _).to_string_lossy();
                    if nm.eq_ignore_ascii_case(name) {
                        found = Some(i);
                        break;
                    }
                }
                let attno = found.ok_or_else(|| {
                    FdwError::SchemaMismatch(format!(
                        "partition_columns names '{name}' but no such column in foreign table"
                    ))
                })?;
                out.push(attno);
            }
            out
        }
    };

    let mut storage_attno_to_parquet_idx: Vec<Option<usize>> = Vec::with_capacity(pg_oids.len());
    {
        let mut next_parquet_idx = 0usize;
        for i in 0..pg_oids.len() {
            if partition_attnums.contains(&i) {
                storage_attno_to_parquet_idx.push(None);
            } else {
                storage_attno_to_parquet_idx.push(Some(next_parquet_idx));
                next_parquet_idx += 1;
            }
        }
    }

    let partition_keys_decl = table_opts.partition_keys.clone();

    // --- projection from fdw_private ----------------------------------
    // `get_foreign_plan` stashed an `IntList` of projected AttrNumbers
    // (1-based) on the ForeignScan node for SELECT statements; UPDATE/
    // DELETE leave `fdw_private` NULL, which we treat here as "project
    // every column" (the modify pipeline reads the full blob anyway).
    //
    // Decode into 0-based parquet column indices plus the matching
    // `attno_map` so iterate knows where to write each projected arrow
    // column in the relation's tuple slot.
    //
    // INVARIANT (cross-ref to `pg_attrs_to_arrow_schema` and the
    // dropped-column rejection above): parquet column index k corresponds
    // to relation column k (0-based attnum). If you ever relax the
    // dropped-column rejection, the `attno - 1` arithmetic here must be
    // remapped through the live-column projection — see Phase C spec for
    // details.
    let natts = pg_oids.len();
    let (projection, attno_map) = unsafe {
        // SAFETY: `node->ss.ps.plan` is the executor's pointer to the
        // ForeignScan we built in get_foreign_plan; live for the call.
        let plan = (*node).ss.ps.plan as *mut pg_sys::ForeignScan;
        let fdw_private = if plan.is_null() {
            std::ptr::null_mut()
        } else {
            (*plan).fdw_private
        };
        if fdw_private.is_null() {
            // No projection requested → emit every column (UPDATE/DELETE
            // path, or any future code path that doesn't set
            // `fdw_private`).
            let all: Vec<usize> = (0..natts).collect();
            (Some(all.clone()), all)
        } else {
            let n = pg_sys::list_length(fdw_private);
            let mut attnums: Vec<i32> = Vec::with_capacity(n as usize);
            for i in 0..n {
                attnums.push(pg_sys::list_nth_int(fdw_private, i));
            }
            if attnums.is_empty() {
                // Pathological: planner gave us an empty list. Fall back
                // to projecting column 0 so iterate has something to read.
                attnums.push(1);
            }
            let mut idxs: Vec<usize> = Vec::with_capacity(attnums.len());
            for a in &attnums {
                let i = (*a as i64) - 1;
                if !(0..(natts as i64)).contains(&i) {
                    return Err(FdwError::SchemaMismatch(format!(
                        "projected attnum {a} out of range (natts={natts})"
                    )));
                }
                idxs.push(i as usize);
            }
            // `idxs` doubles as both the parquet ProjectionMask::roots
            // input AND the arrow_batch_col → relation_col_idx map:
            // ProjectionMask preserves the input ordering, so arrow batch
            // column k carries the value of the parquet column with index
            // `idxs[k]`, which by our schema-by-position invariant is
            // also relation column `idxs[k]`.
            (Some(idxs.clone()), idxs)
        }
    };

    // NOTE: `scan_handoff::publish` is intentionally NOT called here —
    // the sequential entry point [`build_state`] performs that step after
    // this helper returns. The parallel-worker entry point
    // [`build_state_for_parallel_worker`] MUST NOT publish (workers never
    // own the lost-update guard; the parallel-safe gate ensures a parallel
    // SELECT is never followed by a modify on the same plan).

    // --- pushdown: walk PG's executor-residual quals -------------------
    // Pushdown is advisory: PG still re-evaluates every original qual above
    // the scan, so a `Vec::new()` fallback is always safe. Gated by
    // `enable_pushdown` (default true) on the server options.
    let pushed_exprs: Vec<PushedExpr> = if server_opts.enable_pushdown {
        // SAFETY: `node->ss.ps.plan` is the executor's pointer to the
        // ForeignScan node built in `get_foreign_plan`; live for this call.
        // Its `scan.plan.qual` is a `List*<Expr*>` (possibly NIL/null).
        // `walk_quals` accepts a null list and returns an empty vec.
        unsafe {
            let plan = (*node).ss.ps.plan as *mut pg_sys::ForeignScan;
            if plan.is_null() {
                Vec::new()
            } else {
                walk_quals((*plan).scan.plan.qual)
            }
        }
    } else {
        Vec::new()
    };

    // SP-3b: split pushed_exprs into partition-only and storage-only.
    // Mixed-leaf expressions are dropped from both (PG re-evaluates above).
    let (partition_quals, storage_quals_raw) =
        crate::fdw::partition::split_quals_by_target(pushed_exprs.clone(), &partition_attnums);

    // SP-3b LIST-layer partition pruning. Filter the blob list by
    // partition quals BEFORE the caller publishes to scan_handoff so the
    // lost-update guard pins exactly the post-prune set.
    let blobs = if partition_quals.is_empty() || partition_keys_decl.is_empty() {
        blobs
    } else {
        blobs
            .into_iter()
            .filter(|(name, _etag)| {
                let parsed = match crate::fdw::partition::partition_values_from_path(
                    name,
                    &partition_keys_decl,
                ) {
                    Ok(p) => p,
                    Err(_) => return true, // malformed path: keep so the per-blob NOTICE fires later
                };
                crate::fdw::partition::evaluate_partition_quals_against_blob(
                    &partition_quals,
                    &partition_attnums,
                    &parsed,
                    &partition_keys_decl,
                )
            })
            .collect()
    };

    // SP-3b: translate storage quals from foreign-table attno → parquet
    // column index via the storage_attno_to_parquet_idx map. Any expression
    // that fails to translate (defensive — split should have caught partition
    // leaves) is filtered out.
    let storage_quals: Vec<PushedExpr> = storage_quals_raw
        .into_iter()
        .filter_map(|e| {
            crate::fdw::pushdown::translate_qual_to_parquet_idx(e, &storage_attno_to_parquet_idx)
        })
        .collect();

    let qual_filter: Option<Box<dyn QualFilter>> = if storage_quals.is_empty() {
        None
    } else {
        Some(Box::new(PushedExprFilter {
            exprs: storage_quals,
        }))
    };

    let _ = relid; // relid is consumed by callers (publish), not by core.
    Ok(ScanStateCore {
        client,
        blobs,
        pg_oids,
        projection,
        attno_map,
        qual_filter,
        table_opts,
        partition_attnums,
        partition_keys_decl,
        storage_attno_to_parquet_idx,
    })
}

/// Resolve a foreign-table column name to its 0-based attno by walking the
/// relation's TupleDesc (case-insensitive). Used by the SP-3c sorted-mode
/// branch to translate `sorted` option names to attnums.
///
/// # Safety
/// `node` is a live `ForeignScanState`; its `ss.ss_currentRelation` and that
/// relation's `rd_att` TupleDesc are valid for the duration of the call.
unsafe fn resolve_attno_by_name(
    node: *mut pg_sys::ForeignScanState,
    name: &str,
) -> FdwResult<usize> {
    // SAFETY: see fn-level doc — currentRelation + rd_att are live; we use
    // the version-portable `tupdesc_attr` accessor (pg18 reshaped `attrs`).
    unsafe {
        let rel = (*node).ss.ss_currentRelation;
        let tupdesc = (*rel).rd_att;
        let natts = (*tupdesc).natts as usize;
        for i in 0..natts {
            let att = crate::fdw::tupdesc_attr(tupdesc, i);
            let nm = CStr::from_ptr((*att).attname.data.as_ptr() as *const _).to_string_lossy();
            if nm.eq_ignore_ascii_case(name) {
                return Ok(i);
            }
        }
        Err(FdwError::SchemaMismatch(format!(
            "column '{name}' not found on foreign table"
        )))
    }
}

/// Sequential entry point: shared setup + publish handoff + build a
/// [`SequentialRanges`] producer.
///
/// # Safety
/// `node` is a valid `ForeignScanState`. `relid` is a valid foreign-table
/// OID — passed straight from PG.
pub(crate) unsafe fn build_state(
    node: *mut pg_sys::ForeignScanState,
    relid: pg_sys::Oid,
) -> FdwResult<ScanState> {
    // SAFETY: caller invariants per `build_scan_state_core`.
    let mut core = unsafe { build_scan_state_core(node, relid)? };
    // Publish the scan-time (name, etag) list so `begin_foreign_modify`
    // can consume it instead of re-listing. Keyed by relid.
    // `end_foreign_scan` calls `scan_handoff::discard` as a safety net for
    // SELECT-without-UPDATE.
    crate::fdw::modify::scan_handoff::publish(relid, core.blobs.clone());
    let producer: Box<dyn RangeProducer> = Box::new(SequentialRanges {
        blobs: core.blobs.clone(),
        cursor: 0,
    });

    // SP-3c: build the K-way merge stream when sorted mode is active. This is
    // SELECT-only / read-only — the etag-handoff publish above is unchanged,
    // and `is_foreign_scan_parallel_safe` keeps modify statements sequential.
    //
    // The construction is factored into `build_sorted_stream_if_active` so
    // both `build_state` (initial) and `re_scan_foreign_scan` (rebuild on
    // rescan) can call it — the merge stream is single-pass, so a rescan MUST
    // rebuild it from scratch (see C1 in `re_scan_foreign_scan`).
    //
    // SP-3c C1: when sorted mode is ACTIVE the merge stream yields batches
    // carrying ALL parquet columns in parquet order, and `iterate_foreign_scan`
    // indexes `batch.column(arrow_col)` by `attno_map`, so we force a full
    // identity projection — but ONLY when the stream was actually built.
    // `build_sorted_stream_if_active` returns None unless the GUC is on, the
    // table is sorted, AND the command is SELECT (sorted mode skips synthetic
    // ctid stamping, which UPDATE/DELETE rely on). Gating the projection on
    // `sorted_stream.is_some()` keeps it exactly in step with the stream: a
    // modify scan — or a partitioned+sorted table with enable_multifile=off —
    // keeps its real projection/attno_map and the sequential ctid-stamping path.
    // Known v1 limitation: sorted mode disables projection pushdown (reads all
    // columns).
    // SAFETY: `node` is the live ForeignScanState; the helper only reads its
    // currentRelation TupleDesc (via `resolve_attno_by_name`) and command type.
    let sorted_stream = unsafe {
        build_sorted_stream_if_active(
            node,
            &core.client,
            &core.blobs,
            &core.table_opts,
            &core.partition_attnums,
            &core.storage_attno_to_parquet_idx,
            &core.pg_oids,
        )?
    };
    if sorted_stream.is_some() {
        let natts = core.pg_oids.len();
        let identity: Vec<usize> = (0..natts).collect();
        core.projection = Some(identity.clone());
        core.attno_map = identity;
    }

    let mut state = assemble_scan_state(core, producer);
    state.sorted_stream = sorted_stream;
    Ok(state)
}

/// Build the sorted-merge stream for a scan, if sorted mode is active.
/// Returns `Ok(None)` when sorted mode is off (`sorted` option empty). Used by
/// [`build_state`] (initial construction) and [`re_scan_foreign_scan`]
/// (rebuild on rescan — the merge stream is single-pass and its underlying
/// parquet streams are drained after the first scan, so a rescan must rebuild
/// from scratch or it yields zero rows: SP-3c critical bug C1).
///
/// The helper performs all the scan-begin validation/setup inline:
///   - partition + sorted rejection (I1: merge can't synthesize partition cols)
///   - sort-col-is-storage validation
///   - K-cap (256 blobs)
///   - same-type guard across blobs
///   - opening N unprojected parquet streams
///   - constructing the `MultiFileSortedStream`
///
/// NOTE: it does NOT mutate projection/attno_map — the caller forces the
/// identity projection (see `build_state`). On rescan the projection is
/// already identity from the first build, so nothing more is needed.
///
/// # Safety
/// `node` is a live `ForeignScanState`; its `ss.ss_currentRelation` TupleDesc
/// is read (via `resolve_attno_by_name`) for the duration of the call.
unsafe fn build_sorted_stream_if_active(
    node: *mut pg_sys::ForeignScanState,
    client: &AzureBlobClient,
    blobs: &[(String, String)],
    table_opts: &TableOptions,
    partition_attnums: &[usize],
    storage_attno_to_parquet_idx: &[Option<usize>],
    pg_oids: &[pg_sys::Oid],
) -> FdwResult<Option<crate::parquet_io::multifile::MultiFileSortedStream<AzureBlobReader>>> {
    // SP-4: the enable_multifile GUC can force the sequential path even when
    // the table declares sorted/files_in_order. Returning Ok(None) here is
    // the same "sorted off" signal the caller already handles.
    if !crate::ENABLE_MULTIFILE.get() {
        return Ok(None);
    }
    if table_opts.sorted.is_empty() {
        return Ok(None);
    }
    // Sorted merge is SELECT-only: `iterate_foreign_scan` deliberately skips
    // synthetic ctid stamping in sorted mode, which UPDATE/DELETE rely on to
    // identify rows. For any non-SELECT command, return None so the sequential
    // per-blob path (which stamps ctids) runs instead. This is the single gate
    // covering both call sites (`build_state` and `re_scan_foreign_scan`).
    // SAFETY: `node` is the live ForeignScanState; `ss.ps.state` and its
    // `es_plannedstmt` are populated for the duration of execution.
    let is_select = unsafe {
        let estate = (*node).ss.ps.state;
        !estate.is_null()
            && !(*estate).es_plannedstmt.is_null()
            && (*(*estate).es_plannedstmt).commandType == pg_sys::CmdType::CMD_SELECT
    };
    if !is_select {
        return Ok(None);
    }
    let _ = pg_oids; // kept in the signature for symmetry with the caller's
                     // identity-projection sizing; not needed here directly.

    // SP-3c I1: sorted merge cannot synthesize per-blob partition virtual
    // columns. `next_row` early-returns from the K-way merge and never
    // populates `partition_datums_for_current_blob`, so a partitioned
    // table under sorted mode would emit NULL for every partition column.
    // Threading per-blob partition datums through the merge is out of
    // scope for v1 — reject the combination with a clear error instead.
    if !partition_attnums.is_empty() {
        return Err(FdwError::SchemaMismatch(
            "sorted merge is not supported on partitioned tables in v1; \
             combine partition pruning OR sorted merge, not both"
                .to_string(),
        ));
    }
    // Validate sort cols are storage (not partition): partition values
    // are constant per blob, so sorting by them is meaningless and the
    // storage→parquet index map has no entry for them.
    for sort_col_name in &table_opts.sorted {
        // SAFETY: `node` is the live ForeignScanState; `resolve_attno_by_name`
        // only reads its currentRelation's TupleDesc.
        let attno = unsafe { resolve_attno_by_name(node, sort_col_name)? };
        if partition_attnums.contains(&attno) {
            return Err(FdwError::SchemaMismatch(format!(
                "sort column '{sort_col_name}' is a partition column; partition values are constant per blob"
            )));
        }
    }
    // Scan-begin K-cap check (the stream's `new()` also enforces this,
    // but checking here gives a clearer scan-begin error).
    if blobs.len() > 256 {
        return Err(FdwError::SchemaMismatch(format!(
            "sorted-merge cannot open {} blobs at once (cap 256) — narrow the filename glob or partition filter",
            blobs.len()
        )));
    }
    // Resolve sort cols to PARQUET column indices via the SP-3b
    // storage_attno_to_parquet_idx map.
    let sort_col_indices: Vec<usize> = table_opts
        .sorted
        .iter()
        .map(|name| {
            // SAFETY: see above — node is live.
            let attno = unsafe { resolve_attno_by_name(node, name)? };
            storage_attno_to_parquet_idx
                .get(attno)
                .and_then(|x| *x)
                .ok_or_else(|| {
                    FdwError::SchemaMismatch(format!(
                        "sort column '{name}' has no parquet index (storage_attno map miss)"
                    ))
                })
        })
        .collect::<FdwResult<Vec<_>>>()?;

    // Open N parquet streams (one per blob). We open them WITHOUT a
    // projection so the parquet column index equals the arrow batch
    // column index — `sort_col_indices` and `build_key` both index by
    // parquet position. I2 (v1 limitation): the streams are opened with a
    // bare `ParquetRecordBatchStreamBuilder::new` — no SP-1 row-group
    // pruning, no row-level filter, no projection. Sound (PG re-evaluates
    // quals above the scan), just less efficient.
    let mut streams = Vec::with_capacity(blobs.len());
    let mut names = Vec::with_capacity(blobs.len());
    // SAME-TYPE GUARD (Task 2 review): `SortKeyValue::cmp`'s mixed-type
    // arm returns `Equal`, which would silently mis-order rows if two
    // merged blobs declared different physical arrow types for the same
    // sort column. The iteration-time invariant can't catch it because
    // `Equal` is not `<`. We therefore read each builder's arrow schema
    // BEFORE `.build()` and require every blob to agree on the arrow
    // `DataType` of each sort column.
    // Open the blob streams concurrently but with a BOUNDED fan-out. Sequential
    // `block_on`s would mean K serial Azure round-trips at scan init; opening
    // all K (≤256) at once risks socket exhaustion / Azure throttling. A
    // `buffered(16)` stream caps in-flight opens at 16 while still overlapping
    // the IO. `buffered` preserves input order, so `streams`/`names` stay
    // aligned with `blobs`.
    use futures::stream::StreamExt as _;
    const OPEN_CONCURRENCY: usize = 16;
    let opened = runtime::block_on(
        futures::stream::iter(blobs.iter().map(|(blob, _etag)| {
            let reader = client.open_blob(blob);
            let sort_col_indices = &sort_col_indices;
            async move {
                let b = ParquetRecordBatchStreamBuilder::new(reader).await?;
                let schema = b.schema().clone();
                let mut this_types = Vec::with_capacity(sort_col_indices.len());
                for &col_idx in sort_col_indices {
                    if col_idx >= schema.fields().len() {
                        return Err(FdwError::SchemaMismatch(format!(
                            "sort column index {col_idx} out of range for blob '{blob}' (parquet has {} columns)",
                            schema.fields().len()
                        )));
                    }
                    this_types.push(schema.field(col_idx).data_type().clone());
                }
                FdwResult::Ok((blob.clone(), this_types, b.build()?))
            }
        }))
        .buffered(OPEN_CONCURRENCY)
        .collect::<Vec<_>>(),
    );
    // Process in blob order: first blob's sort-column types win, every other
    // blob must agree (same-type guard, see above).
    let mut first_sort_types: Option<Vec<arrow::datatypes::DataType>> = None;
    for res in opened {
        let (blob, this_types, stream) = res?;
        match &first_sort_types {
            None => first_sort_types = Some(this_types),
            Some(first) => {
                for (k, &col_idx) in sort_col_indices.iter().enumerate() {
                    if this_types[k] != first[k] {
                        return Err(FdwError::SchemaMismatch(format!(
                            "blob '{blob}' disagrees on the type of sort column at parquet index {col_idx}: {:?} vs {:?} — all merged blobs must share the same physical type",
                            this_types[k], first[k]
                        )));
                    }
                }
            }
        }
        streams.push(stream);
        names.push(blob);
    }
    Ok(Some(runtime::block_on(
        crate::parquet_io::multifile::MultiFileSortedStream::new(streams, names, sort_col_indices),
    )?))
}

/// Worker-side entry. Same shared setup as [`build_state`] but the caller
/// supplies the `RangeProducer` (a `ParallelRanges`) and we SKIP the
/// `scan_handoff::publish` call — workers never publish, and the
/// parallel-safe gate in `is_foreign_scan_parallel_safe` ensures a
/// parallel SELECT is never followed by an UPDATE that would need the
/// handoff.
///
/// # Safety
/// Same contract as [`build_state`].
pub(crate) unsafe fn build_state_for_parallel_worker(
    node: *mut pg_sys::ForeignScanState,
    relid: pg_sys::Oid,
    range_producer: Box<dyn RangeProducer>,
) -> FdwResult<ScanState> {
    // SAFETY: caller invariants per `build_scan_state_core`.
    let core = unsafe { build_scan_state_core(node, relid)? };
    Ok(assemble_scan_state(core, range_producer))
}

fn assemble_scan_state(core: ScanStateCore, range_producer: Box<dyn RangeProducer>) -> ScanState {
    ScanState {
        client: core.client,
        blobs: core.blobs,
        cur_blob: 0,
        cur_stream: None,
        cur_batch: None,
        cur_row: 0,
        pg_oids: core.pg_oids,
        projection: core.projection,
        attno_map: core.attno_map,
        qual_filter: core.qual_filter,
        range_producer,
        table_opts: core.table_opts,
        blob_id_table: Vec::new(),
        cur_blob_base_id: 0,
        cur_row_in_blob: 0,
        partition_attnums: core.partition_attnums,
        partition_keys_decl: core.partition_keys_decl,
        storage_attno_to_parquet_idx: core.storage_attno_to_parquet_idx,
        partition_datums_for_current_blob: Vec::new(),
        sorted_stream: None,
    }
}

/// Read SERVER / USER MAPPING / TABLE options off the catalog and parse
/// each into the typed `Options` structs.
///
/// # Safety
///
/// `relid` is a live foreign-table OID. All pointers returned by
/// `GetForeignTable` / `GetForeignServer` / `GetUserMapping` are valid for
/// the duration of the current memory context.
unsafe fn read_all_options(
    relid: pg_sys::Oid,
) -> FdwResult<(ServerOptions, UserMappingOptions, TableOptions)> {
    // SAFETY: documented PG accessors; non-null on success or they ereport.
    let (table, server, um) = unsafe {
        let table = pg_sys::GetForeignTable(relid);
        if table.is_null() {
            return Err(FdwError::InvalidOption(format!(
                "foreign table oid {} not found",
                relid.to_u32()
            )));
        }
        let server = pg_sys::GetForeignServer((*table).serverid);
        if server.is_null() {
            return Err(FdwError::InvalidOption("foreign server not found".into()));
        }
        let um = pg_sys::GetUserMapping(pg_sys::GetUserId(), (*server).serverid);
        (table, server, um)
    };

    // SAFETY: `options` is either null (NIL list) or a valid `*List` of
    // `DefElem*`. `pg_list_to_kv` only reads cells and `defname`/`arg`.
    let server_kv = unsafe { pg_list_to_kv((*server).options) };
    // SAFETY: same as above for the foreign table's options list.
    let table_kv = unsafe { pg_list_to_kv((*table).options) };
    let um_kv = if um.is_null() {
        Vec::new()
    } else {
        // SAFETY: same as above; checked non-null.
        unsafe { pg_list_to_kv((*um).options) }
    };

    let server_opts = parse_server_options_from_slice(
        &server_kv
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect::<Vec<_>>(),
    )?;
    let um_opts = parse_user_mapping_options_from_slice(
        &um_kv
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect::<Vec<_>>(),
    )?;
    let table_opts = parse_table_options_from_slice(
        &table_kv
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect::<Vec<_>>(),
    )?;

    Ok((server_opts, um_opts, table_opts))
}

/// Lower a `*List` of `DefElem*` into owned `(name, value)` strings.
///
/// # Safety
///
/// `list` is either null or a valid `*pg_sys::List` whose cells are
/// `*mut DefElem`. We use `pgrx::PgList` to walk it. `defGetString` is
/// the documented helper that copies a `DefElem`'s payload into a
/// palloc'd C string; we immediately clone it into a Rust `String`.
unsafe fn pg_list_to_kv(list: *mut pg_sys::List) -> Vec<(String, String)> {
    if list.is_null() {
        return Vec::new();
    }
    // SAFETY: `from_pg` documented requirement is a valid `*mut List`.
    let pg_list: pgrx::PgList<pg_sys::DefElem> = unsafe { pgrx::PgList::from_pg(list) };
    let mut out = Vec::with_capacity(pg_list.len());
    for def in pg_list.iter_ptr() {
        if def.is_null() {
            continue;
        }
        // SAFETY: `defname` is a NUL-terminated palloc'd string owned by
        // the catalog tuple's memory context; we copy into owned `String`s
        // immediately so we don't retain pointers into PG memory.
        let name = unsafe {
            CStr::from_ptr((*def).defname)
                .to_string_lossy()
                .into_owned()
        };
        let value_ptr = unsafe {
            // SAFETY: `def` is a non-null `*mut DefElem` from the surrounding
            // list iteration; `defGetString` returns a palloc'd NUL-terminated
            // C string (or null when the elem has no string value).
            pg_sys::defGetString(def)
        };
        let value = if value_ptr.is_null() {
            String::new()
        } else {
            // SAFETY: documented to be a NUL-terminated palloc'd C string.
            unsafe { CStr::from_ptr(value_ptr).to_string_lossy().into_owned() }
        };
        out.push((name, value));
    }
    out
}

/// Public wrapper exposed to the modify path so `update::build_plan` can
/// re-derive the blob list without duplicating the glob logic.
///
/// Note: the production modify path now consumes a scan-time list via
/// [`crate::fdw::modify::scan_handoff`]; this fallback is retained for
/// integration tests that drive `build_plan` without a real scan, and is
/// therefore compiled in only under `cfg(any(test, feature = "pg_test"))`.
#[cfg(any(test, feature = "pg_test"))]
pub(crate) fn expand_glob_for_modify(
    client: &AzureBlobClient,
    pattern: &str,
) -> FdwResult<Vec<(String, String)>> {
    expand_glob_with_etags(client, pattern)
}

/// Expand a filename pattern (with `*` / `?` wildcards) into a list of
/// `(blob_name, etag)` by listing the container with the longest no-wildcard
/// prefix, then post-filtering with the regex from `parse_glob`. For the
/// non-glob (single-blob) case, performs a HEAD to capture the etag.
#[cfg(any(test, feature = "pg_test"))]
fn expand_glob_with_etags(
    client: &AzureBlobClient,
    pattern: &str,
) -> FdwResult<Vec<(String, String)>> {
    // Single-blob fast path: no wildcards → one HEAD, no LIST.
    if !pattern.contains('*') && !pattern.contains('?') {
        let etag = runtime::block_on(client.head_etag(pattern))?;
        return Ok(vec![(pattern.to_string(), etag)]);
    }
    let g = crate::fdw::glob::parse_glob(pattern)?;
    let listed = runtime::block_on(client.list_with_prefix_etags(&g.prefix))?;
    let mut out: Vec<(String, String)> = listed
        .into_iter()
        .filter(|(name, _)| g.regex.is_match(name))
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

// ---------- seam: PlanSource (SP-2 replaces this) -------------------------

/// Discovers the list of `(blob_name, etag)` pairs to scan.
///
/// SP-0 introduced this seam so SP-2 (IMPORT FOREIGN SCHEMA + full glob)
/// could replace the implementation without touching the rest of `scan.rs`.
/// SP-2 ships two impls: [`GlobSource`] (full-glob LIST) and
/// [`HeadEtagSource`] (single-blob HEAD for non-glob references).
pub trait PlanSource: Send {
    fn list(&self, client: &AzureBlobClient) -> FdwResult<Vec<(String, String)>>;
}

pub struct GlobSource {
    pub glob: crate::fdw::glob::GlobPattern,
}

impl PlanSource for GlobSource {
    fn list(&self, client: &AzureBlobClient) -> FdwResult<Vec<(String, String)>> {
        let listed = runtime::block_on(client.list_with_prefix_etags(&self.glob.prefix))?;
        let mut out: Vec<(String, String)> = listed
            .into_iter()
            .filter(|(name, _)| self.glob.regex.is_match(name))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }
}

pub struct HeadEtagSource {
    pub name: String,
}

impl PlanSource for HeadEtagSource {
    fn list(&self, client: &AzureBlobClient) -> FdwResult<Vec<(String, String)>> {
        let etag = runtime::block_on(client.head_etag(&self.name))?;
        Ok(vec![(self.name.clone(), etag)])
    }
}

// ---------- seam: QualFilter (SP-1 replaces this) -------------------------

/// Builds a parquet [`RowFilter`] for the blob about to be scanned.
///
/// SP-0 introduces this seam so SP-1 (real qual + row-group pushdown) can
/// replace the implementation. Default impl in SP-0 is
/// [`PushedExprFilter`], which delegates to the existing
/// `crate::fdw::pushdown::build_row_filter` and preserves pre-SP-0 behavior.
pub(crate) trait QualFilter: Send {
    fn build_row_filter(
        &self,
        arrow_schema: &arrow::datatypes::Schema,
        parquet_schema: &parquet::schema::types::SchemaDescriptor,
    ) -> Option<parquet::arrow::arrow_reader::RowFilter>;

    /// Pre-stream row-group pruning. Default impl returns `None` (no pruning).
    /// Returning `Some(Vec<usize>)` is passed verbatim to
    /// `ParquetRecordBatchStreamBuilder::with_row_groups`; `Some(empty)` skips
    /// the blob.
    fn keep_row_groups(
        &self,
        _meta: &parquet::file::metadata::ParquetMetaData,
    ) -> Option<Vec<usize>> {
        None
    }
}

pub(crate) struct PushedExprFilter {
    pub(crate) exprs: Vec<PushedExpr>,
}

impl QualFilter for PushedExprFilter {
    fn build_row_filter(
        &self,
        arrow_schema: &arrow::datatypes::Schema,
        parquet_schema: &parquet::schema::types::SchemaDescriptor,
    ) -> Option<parquet::arrow::arrow_reader::RowFilter> {
        if self.exprs.is_empty() {
            return None;
        }
        build_row_filter(&self.exprs, arrow_schema, parquet_schema)
    }

    fn keep_row_groups(
        &self,
        meta: &parquet::file::metadata::ParquetMetaData,
    ) -> Option<Vec<usize>> {
        if self.exprs.is_empty() {
            return None;
        }
        // We don't have the arrow_schema here — but `prune_row_groups`
        // currently only needs it for type lookup; pass the parquet-derived
        // schema instead via the metadata's file_metadata().
        // INVARIANT: this assumes the foreign-table arrow schema column
        // indices match the parquet schema column indices position-for-
        // position. SP-2 (IMPORT FOREIGN SCHEMA) MUST either preserve this
        // invariant or thread the foreign-table arrow_schema through the
        // QualFilter trait. See SP-1 final review finding I2.
        // TODO(SP-2): thread the foreign-table arrow_schema if SP-2 allows
        // column reordering / subsetting.
        let arrow_schema = parquet::arrow::parquet_to_arrow_schema(
            meta.file_metadata().schema_descr(),
            meta.file_metadata().key_value_metadata(),
        )
        .ok()?;
        crate::fdw::pushdown::prune_row_groups(meta, &self.exprs, &arrow_schema)
    }
}

// ---------- seam: RangeProducer (SP-3 replaces this) ----------------------

/// Hands out the next `(blob_name, etag)` to open for scanning.
///
/// SP-0's [`SequentialRanges`] returns blobs in the order produced by
/// [`PlanSource::list`]. SP-3 (parallel scan + Hive partition) swaps in a
/// DSM-cursor-backed impl that hands work to parallel workers without
/// duplicating blobs across them.
pub(crate) trait RangeProducer: Send {
    fn next_blob(&mut self) -> Option<(String, String)>;
}

pub(crate) struct SequentialRanges {
    pub(crate) blobs: Vec<(String, String)>,
    pub(crate) cursor: usize,
}

impl RangeProducer for SequentialRanges {
    fn next_blob(&mut self) -> Option<(String, String)> {
        let out = self.blobs.get(self.cursor).cloned();
        if out.is_some() {
            self.cursor += 1;
        }
        out
    }
}

/// Advance the per-scan cursor and return the `(batch, row)` of the next
/// available row, or `Ok(None)` at end-of-scan.
fn next_row(state: &mut ScanState) -> FdwResult<Option<(RecordBatch, usize)>> {
    // SP-3c: when sorted mode is active, drive iteration from the K-way
    // merge. Sorted mode is SELECT-only (the parallel-safe gate keeps modify
    // statements sequential, and modify never sets sorted_stream), so the
    // rows have no meaningful per-blob ctid — PG never re-fetches a SELECT
    // row by ctid, so this is sound.
    if let Some(ref mut sm) = state.sorted_stream {
        return runtime::block_on(sm.next_row());
    }
    loop {
        // Fast path: still rows left in the current batch.
        if let Some(batch) = state.cur_batch.as_ref() {
            if state.cur_row < batch.num_rows() {
                let row = state.cur_row;
                state.cur_row += 1;
                // Cheap: RecordBatch clone bumps Arc refcounts only.
                return Ok(Some((batch.clone(), row)));
            }
            state.cur_batch = None;
            state.cur_row = 0;
        }

        // Pull the next batch off the current stream.
        if let Some(stream) = state.cur_stream.as_mut() {
            let next = runtime::block_on(stream.next());
            match next {
                Some(Ok(batch)) => {
                    state.cur_batch = Some(batch);
                    continue;
                }
                Some(Err(e)) => return Err(FdwError::from(e)),
                None => {
                    state.cur_stream = None;
                    state.cur_blob += 1;
                    continue;
                }
            }
        }

        // No stream — pull the next blob from the seam (SP-3 swaps this).
        let (blob, etag) = match state.range_producer.next_blob() {
            Some(p) => p,
            None => return Ok(None),
        };
        // Parse + cache partition values for this blob. On parse/cast
        // failure emit a NOTICE and skip the blob entirely (continue to
        // next blob via the loop).
        if !state.partition_keys_decl.is_empty() {
            let parsed = match crate::fdw::partition::partition_values_from_path(
                &blob,
                &state.partition_keys_decl,
            ) {
                Ok(p) => p,
                Err(e) => {
                    pgrx::notice!("skipping blob '{}': {}", blob, e);
                    continue;
                }
            };
            let mut datums: Vec<pg_sys::Datum> =
                Vec::with_capacity(state.partition_keys_decl.len());
            let mut cast_ok = true;
            for (key, ty) in &state.partition_keys_decl {
                let raw = parsed
                    .get(key)
                    .expect("partition_values_from_path validated presence");
                match crate::convert::arrow_to_pg::parse_text_to_datum(*ty, raw) {
                    Ok(d) => datums.push(d),
                    Err(e) => {
                        pgrx::notice!(
                            "skipping blob '{}': partition cast failed for '{}': {}",
                            blob,
                            key,
                            e
                        );
                        cast_ok = false;
                        break;
                    }
                }
            }
            if !cast_ok {
                continue;
            }
            state.partition_datums_for_current_blob = datums;
        }
        // First chunk of this source blob lands at the next free blob_id.
        state.cur_blob_base_id = state.blob_id_table.len() as u32;
        state.blob_id_table.push(crate::fdw::modify::BlobIdEntry {
            name: blob.clone(),
            chunk_base_row: 0,
            etag,
        });
        state.cur_row_in_blob = 0;
        let reader = state.client.open_blob(&blob);
        // When no pushdown is active, go through the shared `open_stream`
        // helper (also used by `apply_edits` and tests). When we have pushed
        // expressions, we must inline the builder so we can call
        // `with_row_filter` using the parquet/arrow schemas that only become
        // available after `ParquetRecordBatchStreamBuilder::new`.
        let stream = match state.qual_filter.as_ref() {
            None => {
                let opts = ParquetReadOptions {
                    projection: state.projection.clone(),
                    row_filter: None,
                };
                runtime::block_on(open_stream(reader, opts))?
            }
            Some(qf) => runtime::block_on(async {
                let mut b = ParquetRecordBatchStreamBuilder::new(reader).await?;
                if let Some(keep) = qf.keep_row_groups(b.metadata()) {
                    b = b.with_row_groups(keep);
                }
                if let Some(cols) = state.projection.clone() {
                    let pq_schema = b.parquet_schema().clone();
                    b = b.with_projection(ProjectionMask::roots(&pq_schema, cols));
                }
                let arrow_schema = b.schema().clone();
                let parquet_schema = b.parquet_schema().clone();
                if let Some(rf) = qf.build_row_filter(arrow_schema.as_ref(), &parquet_schema) {
                    b = b.with_row_filter(rf);
                }
                FdwResult::Ok(b.build()?)
            })?,
        };
        state.cur_stream = Some(stream);
    }
}

// Silence the unused-field warning on `table_opts`: kept on the state so a
// future task (e.g. PREPARE re-plan) can re-derive blobs without re-reading
// catalogs.
#[allow(dead_code)]
fn _table_opts_used(s: &ScanState) -> &TableOptions {
    &s.table_opts
}

/// Ensure `blob_id_table` has a `BlobIdEntry` for the chunk that owns
/// absolute row `cur_row_in_blob` within the current source blob (whose
/// first chunk lives at index `cur_blob_base_id`).
///
/// Called BEFORE stamping each row's ctid. Idempotent under re-entry on the
/// same row (it checks `len()` before pushing). The lazy "push on first
/// emit into a new chunk" discipline is what keeps a source blob ending at
/// exactly K*CHUNK_ROWS rows from stranding a phantom (K+1)-th chunk entry
/// in `blob_id_table` — the older "push after every emit on the boundary"
/// version did, which shifted every subsequent blob's `blob_id` and either
/// made ctids resolve to the wrong blob (silent rewrite of the wrong file)
/// or to an out-of-range index (spurious SQLSTATE 40001).
pub(crate) fn register_chunk_if_boundary(
    blob_id_table: &mut Vec<crate::fdw::modify::BlobIdEntry>,
    cur_blob_base_id: u32,
    cur_row_in_blob: u64,
) {
    if cur_row_in_blob == 0 {
        return;
    }
    if !cur_row_in_blob.is_multiple_of(crate::fdw::modify::rowid::CHUNK_ROWS) {
        return;
    }
    let base_idx = cur_blob_base_id as usize;
    let chunks_so_far = (cur_row_in_blob / crate::fdw::modify::rowid::CHUNK_ROWS) as usize;
    let expected_entries = base_idx + chunks_so_far + 1;
    if blob_id_table.len() >= expected_entries {
        return;
    }
    let name = blob_id_table[base_idx].name.clone();
    let etag = blob_id_table[base_idx].etag.clone();
    blob_id_table.push(crate::fdw::modify::BlobIdEntry {
        name,
        chunk_base_row: cur_row_in_blob,
        etag,
    });
}

#[cfg(test)]
mod tests {
    use super::register_chunk_if_boundary;
    use crate::fdw::modify::rowid::CHUNK_ROWS;
    use crate::fdw::modify::BlobIdEntry;

    fn entry(name: &str, chunk_base_row: u64, etag: &str) -> BlobIdEntry {
        BlobIdEntry {
            name: name.into(),
            chunk_base_row,
            etag: etag.into(),
        }
    }

    // The lazy-push discipline must be a no-op for rows BEFORE the first
    // chunk boundary. The c0 entry is registered by `next_row` when the blob
    // is opened, not by us.
    #[test]
    fn no_push_within_first_chunk() {
        let mut table = vec![entry("a.parquet", 0, "e")];
        for row in 0..CHUNK_ROWS {
            register_chunk_if_boundary(&mut table, 0, row);
            assert_eq!(table.len(), 1, "row {row} pushed prematurely");
        }
    }

    // First emit at exactly CHUNK_ROWS triggers the lazy push of c1.
    #[test]
    fn push_on_first_emit_into_second_chunk() {
        let mut table = vec![entry("a.parquet", 0, "e")];
        register_chunk_if_boundary(&mut table, 0, CHUNK_ROWS);
        assert_eq!(table.len(), 2);
        assert_eq!(table[1].chunk_base_row, CHUNK_ROWS);
        assert_eq!(table[1].name, "a.parquet");
        assert_eq!(table[1].etag, "e");
    }

    // REGRESSION: a blob ending at exactly K*CHUNK_ROWS rows must NOT
    // strand a phantom (K+1)-th chunk entry. With the old "push after each
    // emit on the boundary" scheme, the final emit at row K*CHUNK_ROWS-1
    // would speculatively push the c_K entry for a row that never comes;
    // `next_row` then opened the next source blob and set
    // `cur_blob_base_id = blob_id_table.len()` — already shifted by one.
    // Modify-side `build_plan` reconstructs the table from
    // `nrows.div_ceil(CHUNK_ROWS).max(1) = K`, mismatching the scan's K+1,
    // so every later blob's ctid resolves to the wrong entry (silent
    // corruption when the wrong entry is a real blob; SQLSTATE 40001 on the
    // way out otherwise).
    #[test]
    fn blob_ending_at_chunk_boundary_does_not_overpush() {
        // Simulate scan iteration over a blob with exactly CHUNK_ROWS rows.
        let mut table = vec![entry("a.parquet", 0, "ea")];
        for row in 0..CHUNK_ROWS {
            register_chunk_if_boundary(&mut table, 0, row);
        }
        // After the LAST row of the blob is emitted (row = CHUNK_ROWS - 1)
        // the iterator increments cur_row_in_blob to CHUNK_ROWS but does NOT
        // emit another row from this blob (stream exhausted) — so
        // `register_chunk_if_boundary` is never called for the phantom row.
        // We assert the table is the single c0 entry, NOT [c0, c1].
        assert_eq!(table.len(), 1, "phantom chunk entry was pushed: {table:?}");
    }

    // The exact bug scenario from the workflow finding: two-blob glob where
    // the first blob has exactly CHUNK_ROWS rows. After scanning blob A and
    // opening blob B, `cur_blob_base_id` must be 1 (NOT 2) so the modify
    // side's `blob_table = [a.c0, b.c0]` (len 2) lines up with the scan's
    // ctids.
    #[test]
    fn two_blobs_first_exactly_chunk_sized_stays_in_sync() {
        // Scan opens A → pushes [a.c0], cur_blob_base_id = 0.
        let mut table = vec![entry("a.parquet", 0, "ea")];
        for row in 0..CHUNK_ROWS {
            register_chunk_if_boundary(&mut table, 0, row);
        }
        assert_eq!(table.len(), 1, "blob A overpushed");

        // Stream of A ends; scan opens B and (via `next_row`) pushes b.c0,
        // taking `cur_blob_base_id = blob_id_table.len() = 1`. Simulate:
        table.push(entry("b.parquet", 0, "eb"));
        let cur_blob_base_id_for_b: u32 = 1;

        // Scan row 0 of B; no push needed.
        register_chunk_if_boundary(&mut table, cur_blob_base_id_for_b, 0);
        assert_eq!(table.len(), 2);
        // The ctid that B row 0 will be stamped with carries
        // blob_id = cur_blob_base_id_for_b + 0 = 1 — pointing at b.c0.
        // Modify side's build_plan derives the same len-2 blob_table from
        // HEAD nrows. We're in sync.
        assert_eq!(table[1].name, "b.parquet");
    }

    // Idempotent under repeated calls at the same row (e.g. a future code
    // path that retries before the cur_row_in_blob increment).
    #[test]
    fn idempotent_on_same_row() {
        let mut table = vec![entry("a.parquet", 0, "ea")];
        register_chunk_if_boundary(&mut table, 0, CHUNK_ROWS);
        register_chunk_if_boundary(&mut table, 0, CHUNK_ROWS);
        register_chunk_if_boundary(&mut table, 0, CHUNK_ROWS);
        assert_eq!(table.len(), 2);
    }
}
