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
    /// Pushed-down expressions walked from the ForeignScan's residual qual
    /// list at `BeginForeignScan` time. Re-published on `ReScan`. Used to
    /// build a parquet [`RowFilter`] per blob opened in `next_row`. Empty
    /// when `enable_pushdown=false` or when no quals are pushable; PG always
    /// re-evaluates the original quals above the scan so it is safe to drop
    /// this entirely on any error path.
    pushed_exprs: Vec<PushedExpr>,
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

/// `GetForeignPaths_function` — add a single un-parameterised path.
///
/// # Safety
///
/// `root` / `baserel` are valid planner pointers; `create_foreignscan_path`
/// is the documented entry point for FDWs to register a scan path.
pub unsafe extern "C-unwind" fn get_foreign_paths(
    root: *mut pg_sys::PlannerInfo,
    baserel: *mut pg_sys::RelOptInfo,
    _foreigntableid: pg_sys::Oid,
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
        #[cfg(any(feature = "pg14", feature = "pg15", feature = "pg16"))]
        let path = pg_sys::create_foreignscan_path(
            root,
            baserel,
            std::ptr::null_mut(), // target = default (baserel->reltarget)
            (*baserel).rows,
            0.0,                  // startup_cost
            (*baserel).rows,      // total_cost — placeholder
            std::ptr::null_mut(), // pathkeys
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
            std::ptr::null_mut(), // pathkeys
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
            std::ptr::null_mut(), // pathkeys
            std::ptr::null_mut(), // required_outer
            std::ptr::null_mut(), // fdw_outerpath
            std::ptr::null_mut(), // fdw_restrictinfo
            std::ptr::null_mut(), // fdw_private
        );
        pg_sys::add_path(baserel, path as *mut pg_sys::Path);
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
            // Lazy chunk registration: see `register_chunk_if_boundary`.
            register_chunk_if_boundary(
                &mut state.blob_id_table,
                state.cur_blob_base_id,
                state.cur_row_in_blob,
            );
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
                // SAFETY (still inside the same `unsafe` block as the
                // `tts_values`/`tts_isnull` writes above): `slot` is a live
                // `TupleTableSlot` owned by the executor for the lifetime
                // of this callback; `tts_tid` is a plain struct field that
                // FDW callbacks are expected to write per PG's executor
                // contract — `ExecForeignUpdate`/`ExecForeignDelete` will
                // decode it back into a `RowId`.
                let rid = crate::fdw::modify::rowid::RowId::from_absolute(
                    state.cur_blob_base_id,
                    state.cur_row_in_blob,
                );
                (*slot).tts_tid = rid.to_ctid();
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
    // SAFETY: fdw_state was set by begin_foreign_scan and not yet dropped.
    let state: &mut ScanState = unsafe { &mut *((*node).fdw_state as *mut ScanState) };
    state.cur_stream = None;
    state.cur_batch = None;
    state.cur_row = 0;
    state.cur_blob = 0;
    state.blob_id_table.clear();
    state.cur_blob_base_id = 0;
    state.cur_row_in_blob = 0;

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

/// Walk the foreign-table / server / user-mapping catalog entries, build
/// the Azure client, expand the glob, and populate a [`ScanState`].
///
/// # Safety
///
/// `node` is a valid `ForeignScanState`. `relid` is a valid foreign-table
/// OID — passed straight from PG.
unsafe fn build_state(
    node: *mut pg_sys::ForeignScanState,
    relid: pg_sys::Oid,
) -> FdwResult<ScanState> {
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
    let blobs = expand_glob_with_etags(&client, &table_opts.filename)?;

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

    // Now that every fallible step has succeeded, publish the scan-time
    // (name, etag) list so `begin_foreign_modify` can consume it instead of
    // re-listing. Keyed by relid. `end_foreign_scan` calls
    // `scan_handoff::discard` as a safety net for SELECT-without-UPDATE.
    crate::fdw::modify::scan_handoff::publish(relid, blobs.clone());

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

    Ok(ScanState {
        client,
        blobs,
        cur_blob: 0,
        cur_stream: None,
        cur_batch: None,
        cur_row: 0,
        pg_oids,
        projection,
        attno_map,
        pushed_exprs,
        table_opts,
        blob_id_table: Vec::new(),
        cur_blob_base_id: 0,
        cur_row_in_blob: 0,
    })
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

/// Expand a filename pattern (optional trailing `*` glob) into a list of
/// `(blob_name, etag)` by listing the container with the longest no-wildcard
/// prefix, then post-filtering with a tiny prefix/suffix matcher. For the
/// non-glob (single-blob) case, performs a HEAD to capture the etag.
fn expand_glob_with_etags(
    client: &AzureBlobClient,
    pattern: &str,
) -> FdwResult<Vec<(String, String)>> {
    if !pattern.contains('*') && !pattern.contains('?') {
        let etag = runtime::block_on(client.head_etag(pattern))?;
        return Ok(vec![(pattern.to_string(), etag)]);
    }
    let star = pattern.find('*').unwrap_or(pattern.len());
    let q = pattern.find('?').unwrap_or(pattern.len());
    let first_wild = star.min(q);
    let prefix = &pattern[..first_wild];

    let listed = runtime::block_on(client.list_with_prefix_etags(prefix))?;

    if pattern.contains('?') {
        return Err(FdwError::InvalidOption(
            "filename glob '?' is not supported in v1".into(),
        ));
    }
    let stars = pattern.matches('*').count();
    if stars > 1 {
        return Err(FdwError::InvalidOption(
            "filename glob may contain at most one '*' in v1".into(),
        ));
    }
    let suffix = &pattern[first_wild + 1..];

    let mut out: Vec<(String, String)> = listed
        .into_iter()
        .filter(|(name, _)| name.starts_with(prefix) && name.ends_with(suffix))
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0)); // deterministic scan order
    Ok(out)
}

/// Advance the per-scan cursor and return the `(batch, row)` of the next
/// available row, or `Ok(None)` at end-of-scan.
fn next_row(state: &mut ScanState) -> FdwResult<Option<(RecordBatch, usize)>> {
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

        // No stream — open the next blob.
        if state.cur_blob >= state.blobs.len() {
            return Ok(None);
        }
        let (blob, etag) = state.blobs[state.cur_blob].clone();
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
        let stream = if state.pushed_exprs.is_empty() {
            let opts = ParquetReadOptions {
                projection: state.projection.clone(),
                row_filter: None,
            };
            runtime::block_on(open_stream(reader, opts))?
        } else {
            runtime::block_on(async {
                let mut b = ParquetRecordBatchStreamBuilder::new(reader).await?;
                if let Some(cols) = state.projection.clone() {
                    let pq_schema = b.parquet_schema().clone();
                    b = b.with_projection(ProjectionMask::roots(&pq_schema, cols));
                }
                // Pushdown is advisory: if `build_row_filter` returns None
                // (no compilable predicates), continue without a filter —
                // PG will re-evaluate the original quals above the scan.
                let arrow_schema = b.schema().clone();
                let parquet_schema = b.parquet_schema().clone();
                if let Some(rf) =
                    build_row_filter(&state.pushed_exprs, arrow_schema.as_ref(), &parquet_schema)
                {
                    b = b.with_row_filter(rf);
                }
                FdwResult::Ok(b.build()?)
            })?
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
