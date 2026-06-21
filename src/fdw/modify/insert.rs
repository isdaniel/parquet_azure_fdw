#![deny(unsafe_op_in_unsafe_fn)]
//! Postgres FDW write-path callbacks (INSERT only in v1).
//!
//! This module is the FFI boundary for the modify side of the FDW. The four
//! callbacks here mirror pgrx's `*_function` signatures and will be installed
//! into the `FdwRoutine` by the (Task 14) handler glue:
//!
//! - [`plan_foreign_modify`] — planner hook; returns `NULL` (no private state
//!   needed for v1 INSERT).
//! - [`begin_foreign_modify`] — allocates a [`InsertState`] (Arrow builders,
//!   parquet writer, Azure client, target blob name) and stashes the boxed
//!   pointer in `ResultRelInfo.ri_FdwState`.
//! - [`exec_foreign_insert`] — decodes one tuple from the slot, appends to
//!   builders, flushes a `RecordBatch` to the parquet writer every
//!   [`BATCH_ROWS`] rows. Returns the input slot unchanged (RETURNING is
//!   echoed by Postgres from the input tuple).
//! - [`end_foreign_modify`] — finishes any pending batch, flushes the parquet
//!   footer, uploads the buffered file as a single block blob, drops state.
//!
//! ## Transaction semantics (v1, see spec §7.2)
//!
//! There is intentionally no `xact_callback` registered. The reasoning:
//!
//! - Postgres calls `end_foreign_modify` on both COMMIT and ABORT paths. We
//!   use `IsAbortedTransactionBlockState()` to detect the abort case and
//!   drop the buffer without uploading, per spec §7.2: a ROLLBACK before
//!   `EndForeignModify` discards the in-memory buffer and produces no blob.
//!   Builders and writer live only in process memory, so dropping the boxed
//!   state is sufficient cleanup; there is no external state to undo.
//! - On the COMMIT path the upload happens inside `end_foreign_modify`, which
//!   runs before transaction commit completes. If the upload fails we raise
//!   and the transaction aborts cleanly.
//! - A ROLLBACK that fires *after* `end_foreign_modify` succeeded (i.e. after
//!   the blob is already in Azure) will NOT delete the uploaded blob — this
//!   is documented in spec §7.2 as an accepted v1 limitation (treat
//!   foreign-table INSERT as best-effort atomic at the file granularity,
//!   not the transaction granularity).
//!
//! ## Unsafe / FFI carve-out
//!
//! `src/fdw/mod.rs` carries `#![deny(unsafe_code)]`; this submodule opts in
//! with `#![allow(unsafe_code)]` because the slot-decoding logic needs raw
//! `tts_values` / `tts_isnull` access. Every `unsafe { ... }` block is paired
//! with a `// SAFETY:` comment naming the pgrx / Postgres invariant it
//! relies on. `#![deny(unsafe_op_in_unsafe_fn)]` forces each unsafe op
//! inside an `unsafe fn` to be individually opted into.

use crate::azure::{build_credential, generate_blob_name, AzureBlobClient, AzureBlobWriter};
use crate::convert::pg_to_arrow::{pg_attrs_to_arrow_schema, RecordBatchBuilders};
use crate::error::{raise, FdwError, FdwResult};
use crate::fdw::options::{
    parse_server_options_from_slice, parse_table_options_from_slice,
    parse_user_mapping_options_from_slice, validate_combo, TableOptions,
};
use crate::parquet_io::writer::ParquetBatchWriter;
use crate::parquet_io::Compression;
use crate::runtime;

use arrow::datatypes::SchemaRef;
use pgrx::pg_sys;
use std::ffi::{c_int, c_void, CStr};

/// Buffered rows before flushing a `RecordBatch` to the parquet writer.
/// Matches the arrow default row-group / parquet page hinting and keeps peak
/// resident memory bounded for wide tables.
const BATCH_ROWS: usize = 8192;

/// Per-INSERT executor state, owned by PG via `Box::into_raw` /
/// `Box::from_raw` round-trip through `ResultRelInfo.ri_FdwState`.
pub struct InsertState {
    /// Column-aligned Arrow array builders for the in-progress batch.
    /// Re-allocated after each flush.
    builders: RecordBatchBuilders,
    /// Cached schema — needed both to re-create builders after each flush and
    /// to construct the per-batch `RecordBatch`.
    schema: SchemaRef,
    /// Per-attribute pg type OIDs in tupdesc order. Used by `append_slot`
    /// to dispatch to the right typed builder helper.
    pg_oids: Vec<pg_sys::Oid>,
    /// In-memory parquet writer. Buffers the whole file before upload.
    writer: ParquetBatchWriter,
    /// Container-scoped Azure client; cheap to clone.
    client: AzureBlobClient,
    /// Target blob name chosen at BEGIN time (timestamp + uuid).
    target_name: String,
    /// Compression codec (stored only for diagnostics; the writer already
    /// carries it).
    #[allow(dead_code)]
    compression: Compression,
    /// Rows in the in-progress (un-finalized) builder batch.
    rows_in_current_batch: usize,
}

// ---------- public callbacks ------------------------------------------------

/// `PlanForeignModify_function` — v1 stores no private planning state; the
/// per-relation options are re-read in `begin_foreign_modify`.
///
/// # Safety
///
/// PG passes valid planner pointers to a registered FDW callback.
pub unsafe extern "C-unwind" fn plan_foreign_modify(
    _root: *mut pg_sys::PlannerInfo,
    _plan: *mut pg_sys::ModifyTable,
    _result_relation: pg_sys::Index,
    _subplan_index: c_int,
) -> *mut pg_sys::List {
    std::ptr::null_mut()
}

/// `BeginForeignModify_function` — dispatches on `mtstate->operation`. For
/// INSERT, builds an [`InsertState`]; for UPDATE/DELETE, builds a
/// [`super::update::ModifyPlan`]. The constructed `FdwModifyState` is boxed
/// into `ri_FdwState` and reclaimed in `end_foreign_modify`.
///
/// # Safety
///
/// `mtstate` / `rinfo` are valid executor pointers; `rinfo->ri_RelationDesc`
/// is the live foreign relation being modified. We only mutate `ri_FdwState`,
/// which is reserved for FDW use.
pub unsafe extern "C-unwind" fn begin_foreign_modify(
    mtstate: *mut pg_sys::ModifyTableState,
    rinfo: *mut pg_sys::ResultRelInfo,
    _fdw_private: *mut pg_sys::List,
    _subplan_index: c_int,
    eflags: c_int,
) {
    // EXPLAIN-only: skip everything (no Azure client, no builders) — matches
    // the scan callback's EXPLAIN short-circuit.
    if (eflags as u32) & pg_sys::EXEC_FLAG_EXPLAIN_ONLY != 0 {
        return;
    }

    // SAFETY: PG guarantees `mtstate` is a live `ModifyTableState`; its
    // `operation` field is a `CmdType` enum that selects our state shape.
    let op = unsafe { (*mtstate).operation };

    let state = match op {
        pg_sys::CmdType::CMD_INSERT => {
            // SAFETY: PG guarantees `rinfo->ri_RelationDesc` is valid.
            let rel = unsafe { (*rinfo).ri_RelationDesc };
            // SAFETY: `rel` is a live Relation pointer; `rd_id` is its OID.
            let relid = unsafe { (*rel).rd_id };
            // SAFETY: `build_state` documents that `rel` must be a live
            // Relation; we just dereferenced it above.
            match unsafe { build_state(rel, relid) } {
                Ok(s) => super::FdwModifyState::Insert(Box::new(s)),
                Err(e) => raise(e),
            }
        }
        pg_sys::CmdType::CMD_UPDATE => {
            // SAFETY: `mtstate` and `rinfo` are valid executor pointers per
            // the FDW callback contract; `build_plan` documents the same.
            match unsafe { super::update::build_plan(mtstate, rinfo, false) } {
                Ok(p) => super::FdwModifyState::Update(Box::new(p)),
                Err(e) => raise(e),
            }
        }
        pg_sys::CmdType::CMD_DELETE => {
            // SAFETY: same as the UPDATE arm.
            match unsafe { super::update::build_plan(mtstate, rinfo, true) } {
                Ok(p) => super::FdwModifyState::Update(Box::new(p)),
                Err(e) => raise(e),
            }
        }
        other => raise(FdwError::SchemaMismatch(format!(
            "unsupported CmdType {other}"
        ))),
    };

    // Open the per-statement coordinator with a clone of the AzureBlobClient
    // held by the state. This is the hook that lets E-T4's `commit_plan`
    // register staging blobs / mark commits — without `open_statement`, those
    // calls silently no-op.
    let client = match &state {
        super::FdwModifyState::Insert(s) => s.client.clone(),
        super::FdwModifyState::Update(p) => p.client.clone(),
    };
    super::coordinator::open_statement(client);

    // SAFETY: stash the boxed state into the FDW-reserved slot. The matching
    // `Box::from_raw` lives in `end_foreign_modify` (via `super::take_state`).
    unsafe {
        (*rinfo).ri_FdwState = Box::into_raw(Box::new(state)) as *mut c_void;
    }
}

/// `ExecForeignInsert_function` — decode one tuple, append to builders, flush
/// a batch if `BATCH_ROWS` rows accumulated. Returns the input slot unchanged
/// (Postgres builds RETURNING from it).
///
/// # Safety
///
/// `rinfo->ri_FdwState` was populated by `begin_foreign_modify` and is alive
/// until `end_foreign_modify`. `slot` is the input tuple owned by the
/// executor for the duration of this call.
pub unsafe extern "C-unwind" fn exec_foreign_insert(
    _estate: *mut pg_sys::EState,
    rinfo: *mut pg_sys::ResultRelInfo,
    slot: *mut pg_sys::TupleTableSlot,
    _plan_slot: *mut pg_sys::TupleTableSlot,
) -> *mut pg_sys::TupleTableSlot {
    // SAFETY: see fn-level safety; `ri_FdwState` is non-null between begin
    // and end (we return early in begin when EXPLAIN-only, but PG also
    // refuses to call exec in EXPLAIN-only mode, so the pointer is live).
    let state_ptr = unsafe { (*rinfo).ri_FdwState as *mut super::FdwModifyState };
    if state_ptr.is_null() {
        raise(FdwError::SchemaMismatch(
            "exec_foreign_insert called without initialised ri_FdwState".into(),
        ));
    }
    // SAFETY: non-null per the check above; pointer was produced by
    // `Box::into_raw` in `begin_foreign_modify`.
    let state: &mut super::FdwModifyState = unsafe { &mut *state_ptr };
    let insert_state = match state {
        super::FdwModifyState::Insert(s) => s,
        _ => raise(FdwError::SchemaMismatch(
            "exec_foreign_insert on non-insert state".into(),
        )),
    };

    // Make sure every attr is materialized; virtual slots from RETURNING may
    // not have all columns realized otherwise.
    // SAFETY: pgrx exposes `slot_getallattrs` as a cshim; it requires only
    // that `slot` be a valid `TupleTableSlot`.
    unsafe {
        pg_sys::slot_getallattrs(slot);
    }

    // SAFETY: every attr was materialized via `slot_getallattrs` above;
    // `append_slot` requires only a valid slot and its `pg_oids` length.
    if let Err(e) = unsafe { append_slot(insert_state, slot) } {
        raise(e);
    }

    if insert_state.rows_in_current_batch >= BATCH_ROWS {
        if let Err(e) = flush_batch(insert_state) {
            raise(e);
        }
    }

    slot
}

/// `EndForeignModify_function` — for INSERT, flush+finalize+upload; for
/// UPDATE/DELETE, drive the commit_plan rewrite. Always reclaims the boxed
/// state (via the shared `super::take_state` helper) so memory is released
/// on both COMMIT and ABORT paths.
///
/// # Safety
///
/// Pairs with the `Box::into_raw` in `begin_foreign_modify`. PG calls
/// `end_foreign_modify` exactly once per `begin_foreign_modify`, on both
/// COMMIT and ABORT paths.
pub unsafe extern "C-unwind" fn end_foreign_modify(
    _estate: *mut pg_sys::EState,
    rinfo: *mut pg_sys::ResultRelInfo,
) {
    // SAFETY: `take_state` reclaims the box and nulls the slot. Returns None
    // for EXPLAIN-only (state never created) or double-call paths.
    let boxed = match unsafe { super::take_state(rinfo) } {
        Some(b) => b,
        None => return,
    };

    // Per spec §7.2: a ROLLBACK before `end_foreign_modify` discards the
    // in-memory buffer; no blob is created. Postgres still calls this
    // callback on the abort path, so we must check the transaction state
    // and skip the I/O when aborted. The boxed state drops at end of scope,
    // freeing builders/writer/Azure clients.
    // SAFETY: `IsAbortedTransactionBlockState` is a pure read of the
    // current transaction state; safe to call from any backend context.
    let aborted = unsafe { pg_sys::IsAbortedTransactionBlockState() };
    if aborted {
        return;
    }

    match *boxed {
        super::FdwModifyState::Insert(s) => {
            if let Err(e) = finalize_and_upload(*s) {
                raise(e);
            }
        }
        super::FdwModifyState::Update(p) => {
            if let Err(e) = super::update::commit_plan(*p) {
                raise(e);
            }
        }
    }

    // Successful end: drop the per-statement coordinator. The abort path above
    // returns early without calling this — the xact callback handles cleanup
    // of any registered staging blobs in that case.
    super::coordinator::close_statement_success();
}

// ---------- internals -------------------------------------------------------

/// Walk the foreign-table / server / user-mapping catalog entries, build the
/// Arrow schema + builders, open the parquet writer and Azure client, and
/// pick the target blob name.
///
/// # Safety
///
/// `rel` is a valid live `Relation` (caller asserts via the FDW callback
/// contract); `relid` is its OID.
unsafe fn build_state(rel: pg_sys::Relation, relid: pg_sys::Oid) -> FdwResult<InsertState> {
    // --- options -------------------------------------------------------
    // SAFETY: documented PG catalog accessors; results are palloc'd.
    let (server_opts, um_opts, table_opts) = unsafe { read_all_options(relid) }?;
    validate_combo(&server_opts, &um_opts)?;

    // --- credential + client ------------------------------------------
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

    // --- schema + per-attr OIDs ---------------------------------------
    // SAFETY: `rd_att` is a live TupleDesc. We use the version-portable
    // `crate::fdw::tupdesc_attr` accessor (pg18 dropped the `attrs` FAM and
    // replaced it with `compact_attrs`).
    let (attrs, pg_oids) = unsafe {
        let tupdesc = (*rel).rd_att;
        let natts = (*tupdesc).natts as usize;
        let mut attrs = Vec::with_capacity(natts);
        let mut oids = Vec::with_capacity(natts);
        for i in 0..natts {
            let att = crate::fdw::tupdesc_attr(tupdesc, i);
            if (*att).attisdropped {
                return Err(FdwError::SchemaMismatch(format!(
                    "dropped column at attnum {} not supported on write path",
                    (*att).attnum
                )));
            }
            attrs.push(*att);
            oids.push((*att).atttypid);
        }
        (attrs, oids)
    };

    let schema = pg_attrs_to_arrow_schema(&attrs)?;
    let builders = RecordBatchBuilders::new(schema.clone(), BATCH_ROWS)?;
    let writer = ParquetBatchWriter::new(schema.clone(), table_opts.compression)?;

    let target_name = pick_target_name(&table_opts);

    Ok(InsertState {
        builders,
        schema,
        pg_oids,
        writer,
        client,
        target_name,
        compression: table_opts.compression,
        rows_in_current_batch: 0,
    })
}

/// Derive the upload target blob name from the table option.
///
/// If the user gave a literal blob name (no `*`), reuse it verbatim — that's
/// the natural one-blob-per-table case. If they gave a glob, treat the
/// portion before `*` as a prefix and synthesize a `{prefix}{ts}-{uuid}.parquet`
/// name so each INSERT lands in a fresh blob (matches the read-side glob
/// expansion: `dir/*` → write to `dir/`).
fn pick_target_name(opts: &TableOptions) -> String {
    let fname = opts.filename.as_str();
    if let Some(star) = fname.find('*') {
        let prefix = &fname[..star];
        generate_blob_name(prefix)
    } else {
        fname.to_string()
    }
}

/// Walk `slot->tts_values` / `tts_isnull` and append one row to the per-column
/// Arrow builders. Dispatches on the cached pg type OIDs.
///
/// # Safety
///
/// `slot` is a valid `TupleTableSlot` whose `tts_values`/`tts_isnull` arrays
/// are sized to the tuple descriptor — i.e. at least `state.pg_oids.len()`
/// entries. Caller must have called `slot_getallattrs(slot)` so every datum
/// is materialized.
unsafe fn append_slot(state: &mut InsertState, slot: *mut pg_sys::TupleTableSlot) -> FdwResult<()> {
    // SAFETY: `tts_values` / `tts_isnull` are arrays sized to `tts_nvalid`
    // (≥ tupdesc->natts after slot_getallattrs). We index them by column
    // number within `state.pg_oids.len()`.
    let (values, isnulls) = unsafe { ((*slot).tts_values, (*slot).tts_isnull) };

    for (i, &oid) in state.pg_oids.iter().enumerate() {
        // SAFETY: `i < state.pg_oids.len() <= tts_nvalid` by construction
        // above, so the offset stays within the slot's value/null arrays.
        let is_null = unsafe { *isnulls.add(i) };
        // SAFETY: same bound as `is_null` above.
        let datum = unsafe { *values.add(i) };
        // SAFETY: `i` indexes both `state.pg_oids` and `state.builders`,
        // which were built in parallel; `oid` and (datum, is_null) describe
        // the same column.
        unsafe { append_one(&mut state.builders, i, oid, datum, is_null) }?;
    }
    state.rows_in_current_batch += 1;
    Ok(())
}

// ---------- version-portable Datum -> primitive helpers --------------------
//
// PG14 and PG15 do not FFI-export `DatumGetBool` / `DatumGetInt16` /
// `DatumGetInt32` / `DatumGetInt64` — they are `static inline` in `postgres.h`
// and pgrx omits them on those versions. The C inlines are pure bit-casts of
// the underlying `uintptr_t` (`Datum`) value, so we reproduce them in Rust
// against the public `Datum::value()` accessor. This is byte-identical to
// what the C inlines do on every PG version we support.
//
// `DatumGetFloat4` / `DatumGetFloat8` ARE FFI-exported on pg14+ (they need an
// f32/f64 reinterpret that the bindgen pipeline doesn't elide), so we keep
// calling them directly.

#[inline]
fn datum_to_bool(d: pg_sys::Datum) -> bool {
    (d.value() as u8) != 0
}

#[inline]
fn datum_to_i16(d: pg_sys::Datum) -> i16 {
    d.value() as i16
}

#[inline]
fn datum_to_i32(d: pg_sys::Datum) -> i32 {
    d.value() as i32
}

#[inline]
fn datum_to_i64(d: pg_sys::Datum) -> i64 {
    d.value() as i64
}

/// Decode a single (datum, isnull) pair and route to the typed builder helper
/// matching the PG type OID. Unsupported OIDs return `UnsupportedType` —
/// matches the v1 type matrix in `pg_oid_to_arrow_type`.
///
/// # Safety
///
/// `datum` is the raw datum from `tts_values` for type `oid`. We only call
/// the documented `DatumGet*` accessors and, for varlena types, the standard
/// `pg_detoast_datum` + slice-from-`text*` pattern.
pub(crate) unsafe fn append_one(
    builders: &mut RecordBatchBuilders,
    col: usize,
    oid: pg_sys::Oid,
    datum: pg_sys::Datum,
    is_null: bool,
) -> FdwResult<()> {
    if is_null {
        return builders.append_null(col);
    }

    match oid {
        x if x == pg_sys::BOOLOID => {
            // PG14/15 do not export `DatumGetBool` (it's a `static inline` in
            // postgres.h). On all versions a bool datum is the low byte of
            // the underlying machine word — bit-cast directly via the public
            // `Datum::value()` accessor to stay version-portable.
            let v = datum_to_bool(datum);
            builders.append_bool(col, Some(v))
        }
        x if x == pg_sys::INT2OID => {
            // See BOOL arm: pg14/15 don't export `DatumGetInt16`.
            let v = datum_to_i16(datum);
            builders.append_i16(col, Some(v))
        }
        x if x == pg_sys::INT4OID => {
            // See BOOL arm: pg14/15 don't export `DatumGetInt32`.
            let v = datum_to_i32(datum);
            builders.append_i32(col, Some(v))
        }
        x if x == pg_sys::INT8OID => {
            // See BOOL arm: pg14/15 don't export `DatumGetInt64`.
            let v = datum_to_i64(datum);
            builders.append_i64(col, Some(v))
        }
        x if x == pg_sys::FLOAT4OID => {
            // SAFETY: see BOOL arm.
            let v = unsafe { pg_sys::DatumGetFloat4(datum) };
            builders.append_f32(col, Some(v))
        }
        x if x == pg_sys::FLOAT8OID => {
            // SAFETY: see BOOL arm.
            let v = unsafe { pg_sys::DatumGetFloat8(datum) };
            builders.append_f64(col, Some(v))
        }
        x if x == pg_sys::TEXTOID || x == pg_sys::VARCHAROID => {
            // SAFETY: text/varchar datums are pointers to `text` (varlena).
            // We must detoast before reading length/payload because the value
            // may be compressed or external. `pg_detoast_datum` returns an
            // owned-by-current-memory-context detoasted copy (or the same
            // pointer if already inline + uncompressed).
            let s = unsafe { text_datum_to_str(datum) }?;
            builders.append_str(col, Some(&s))
        }
        other => Err(FdwError::UnsupportedType {
            pg_type: format!("oid={} (write path)", other.to_u32()),
            arrow_type: "<n/a>".to_string(),
        }),
    }
}

/// Convert a Postgres `text` Datum to an owned `String`.
///
/// # Safety
///
/// `datum` is a non-null `text*` datum produced by Postgres for the
/// surrounding tuple. We detoast (which is a documented no-op for inline
/// uncompressed values) and convert via `text_to_cstring`.
unsafe fn text_datum_to_str(datum: pg_sys::Datum) -> FdwResult<String> {
    // SAFETY: the datum's pointer payload is a live `varlena*`. PG guarantees
    // detoast returns a usable `text*` pointer in the current memory context.
    let s = unsafe {
        let detoasted = pg_sys::pg_detoast_datum(datum.cast_mut_ptr::<pg_sys::varlena>());
        // `text_to_cstring` allocates a NUL-terminated palloc'd C string
        // copy. We immediately clone into a Rust `String` so we don't hold
        // pointers into PG memory beyond this scope.
        let cstr_ptr = pg_sys::text_to_cstring(detoasted as *const pg_sys::text);
        let owned = CStr::from_ptr(cstr_ptr).to_string_lossy().into_owned();
        // Free the palloc'd cstring + the detoast copy (if it was a copy)
        // via pfree to keep peak memory in long INSERTs bounded.
        pg_sys::pfree(cstr_ptr.cast());
        if !std::ptr::eq(detoasted, datum.cast_mut_ptr::<pg_sys::varlena>()) {
            pg_sys::pfree(detoasted.cast());
        }
        owned
    };
    Ok(s)
}

/// Finalize the in-progress builder into a `RecordBatch`, write it to the
/// parquet writer, and reset the builders for the next chunk.
fn flush_batch(state: &mut InsertState) -> FdwResult<()> {
    if state.rows_in_current_batch == 0 {
        return Ok(());
    }
    // Swap out the builders so we can take ownership for `finish()` while
    // keeping `state` borrow-safe.
    let fresh = RecordBatchBuilders::new(state.schema.clone(), BATCH_ROWS)?;
    let old = std::mem::replace(&mut state.builders, fresh);
    let batch = old.finish()?;
    state.writer.write(&batch)?;
    // Cap the in-progress parquet accumulator at MAX_BLOB_BYTES. Without
    // this, a `COPY ... FROM` of a multi-GiB source grows the ArrowWriter's
    // internal Vec<u8> unbounded — `AzureBlobWriter::upload`'s MAX_BLOB_BYTES
    // check only fires AFTER the writer is finalised, by which point the
    // backend has already OOMed. We refuse early with a clear error so the
    // user can split the load. (Note: `bytes_written` is a lower bound;
    // unflushed row-group buffers may add a few MiB more, comfortably below
    // the MaxAllocSize headroom.)
    let written = state.writer.bytes_written() as u64;
    if written > crate::azure::MAX_BLOB_BYTES {
        return Err(crate::error::FdwError::Azure(format!(
            "INSERT/COPY accumulator: {written} bytes written exceeds \
             MAX_BLOB_BYTES={cap}; split the load into multiple statements \
             or smaller batches",
            cap = crate::azure::MAX_BLOB_BYTES
        )));
    }
    state.rows_in_current_batch = 0;
    Ok(())
}

/// Flush the tail batch, finalize the parquet footer, and upload the bytes.
fn finalize_and_upload(mut state: InsertState) -> FdwResult<()> {
    // Tail flush.
    if state.rows_in_current_batch > 0 {
        flush_batch(&mut state)?;
    }
    // If no rows were ever inserted, skip the upload entirely: writing a
    // zero-row parquet file is legal but probably not what the user wants
    // for an empty `INSERT ... SELECT ... WHERE false`.
    let InsertState {
        writer,
        client,
        target_name,
        rows_in_current_batch: _,
        builders: _,
        schema: _,
        pg_oids: _,
        compression: _,
    } = state;

    let bytes = writer.finish()?;
    // Always upload — even a header-only parquet file is a valid record of
    // the (possibly zero-row) write. This matches what a `COPY ... TO`
    // would do.
    let blob_writer = AzureBlobWriter::new(&client, &target_name);
    runtime::block_on(blob_writer.upload(bytes))?;
    Ok(())
}

// ---------- catalog plumbing (mirrors scan.rs) -----------------------------

/// Read SERVER / USER MAPPING / TABLE options off the catalog.
///
/// # Safety
///
/// `relid` is a live foreign-table OID. All returned pointers from
/// `GetForeignTable` / `GetForeignServer` / `GetUserMapping` are valid for
/// the duration of the current memory context.
pub(crate) unsafe fn read_all_options(
    relid: pg_sys::Oid,
) -> FdwResult<(
    crate::fdw::options::ServerOptions,
    crate::fdw::options::UserMappingOptions,
    TableOptions,
)> {
    // SAFETY: documented PG accessors; non-null on success.
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

    // SAFETY: `options` is either null or a valid `*List` of `DefElem*`.
    let server_kv = unsafe { pg_list_to_kv((*server).options) };
    // SAFETY: same — `(*table).options` is either null or a valid `*List`.
    let table_kv = unsafe { pg_list_to_kv((*table).options) };
    let um_kv = if um.is_null() {
        Vec::new()
    } else {
        // SAFETY: checked non-null.
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
/// `*mut DefElem`.
unsafe fn pg_list_to_kv(list: *mut pg_sys::List) -> Vec<(String, String)> {
    if list.is_null() {
        return Vec::new();
    }
    // SAFETY: documented requirement is a valid `*mut List`.
    let pg_list: pgrx::PgList<pg_sys::DefElem> = unsafe { pgrx::PgList::from_pg(list) };
    let mut out = Vec::with_capacity(pg_list.len());
    for def in pg_list.iter_ptr() {
        if def.is_null() {
            continue;
        }
        // SAFETY: `defname` is a NUL-terminated palloc'd C string; we copy
        // immediately into owned `String`s.
        let name = unsafe {
            CStr::from_ptr((*def).defname)
                .to_string_lossy()
                .into_owned()
        };
        let value_ptr = unsafe {
            // SAFETY: `def` is a non-null `*mut DefElem` from `pg_list_to_kv`'s
            // iteration; `defGetString` returns a palloc'd NUL-terminated
            // C string (or null if the elem has no string value).
            pg_sys::defGetString(def)
        };
        let value = if value_ptr.is_null() {
            String::new()
        } else {
            // SAFETY: documented NUL-terminated palloc'd C string.
            unsafe { CStr::from_ptr(value_ptr).to_string_lossy().into_owned() }
        };
        out.push((name, value));
    }
    out
}
