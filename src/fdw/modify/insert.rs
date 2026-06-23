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
    parse_user_mapping_options_from_slice, validate_combo, PgPartitionType, TableOptions,
};
use crate::fdw::partition::PartitionTupleKey;
use crate::parquet_io::writer::ParquetBatchWriter;
use crate::parquet_io::Compression;
use crate::runtime;

use arrow::datatypes::SchemaRef;
use pgrx::pg_sys;
use std::collections::HashMap;
use std::ffi::{c_int, c_void, CStr};

/// Buffered rows before flushing a `RecordBatch` to the parquet writer.
/// Matches the arrow default row-group / parquet page hinting and keeps peak
/// resident memory bounded for wide tables.
const BATCH_ROWS: usize = 8192;

/// Per-INSERT executor state, owned by PG via `Box::into_raw` /
/// `Box::from_raw` round-trip through `ResultRelInfo.ri_FdwState`.
pub struct InsertState {
    /// Storage-column Arrow schema. EXCLUDES partition columns (those are
    /// encoded in the blob path, not in the parquet data). For
    /// non-partitioned tables this is the full tupdesc schema.
    schema: SchemaRef,
    /// Per-attribute pg type OIDs in tupdesc order. Used by `append_slot`
    /// to dispatch to the right typed builder helper.
    pg_oids: Vec<pg_sys::Oid>,
    /// Container-scoped Azure client; cheap to clone.
    client: AzureBlobClient,
    /// Compression codec for newly-written blobs (also threaded into each
    /// per-group `ParquetBatchWriter` lazily on first append to that group).
    compression: Compression,
    /// 0-based attnums of the partition columns (cached for hot-path access
    /// in `append_slot`). Empty for non-partitioned tables.
    partition_attnums: Vec<usize>,
    /// Declared (name, type) for each partition column, in declaration
    /// order. Used to format per-group blob paths
    /// (`base_prefix/key=val/key=val/...`).
    partition_keys_decl: Vec<(String, PgPartitionType)>,
    /// Path prefix derived from the table option `filename` glob. Per-group
    /// blob names are built as `generate_blob_name(base_prefix/seg/seg/...)`.
    base_prefix: String,
    /// Target blob name for the empty-key (non-partitioned) fall-through.
    /// Preserved verbatim for the legacy "literal filename, no glob" case so
    /// non-partitioned tables keep their single-blob-per-statement semantics.
    ///
    /// Dead state when `partition_attnums` is non-empty: partitioned INSERTs
    /// always derive per-group blob names via `generate_blob_name` under
    /// `base_prefix/key=val/...` and never consult this field.
    legacy_single_target: Option<String>,
    /// Per-partition-tuple in-progress builders. For non-partitioned tables
    /// this map has exactly one entry keyed by the empty `PartitionTupleKey`.
    /// Entries are created lazily on first row routed to a group.
    builders: HashMap<PartitionTupleKey, RecordBatchBuilders>,
    /// Per-group accumulated parquet bytes — one writer per group, created
    /// lazily on the first BATCH_ROWS flush for that group.
    writers: HashMap<PartitionTupleKey, ParquetBatchWriter>,
    /// Per-group target blob name, fixed when the group is first observed.
    /// Ensures all rows for one tuple key land in the same blob.
    target_names: HashMap<PartitionTupleKey, String>,
    /// Per-group in-progress row counts (for BATCH_ROWS check + empty-group
    /// detection at finalize).
    rows_in_current_batch: HashMap<PartitionTupleKey, usize>,
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

    // Per-group BATCH_ROWS threshold check. We flush ANY group that has
    // reached the cap, not the whole state — keeps memory bounded under
    // skewed partition distributions (one heavy group + many tiny groups
    // shouldn't all flush together).
    let to_flush: Vec<PartitionTupleKey> = insert_state
        .rows_in_current_batch
        .iter()
        .filter_map(|(k, &n)| {
            if n >= BATCH_ROWS {
                Some(k.clone())
            } else {
                None
            }
        })
        .collect();
    for key in to_flush {
        if let Err(e) = flush_group(insert_state, &key) {
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

    // --- partition resolution ------------------------------------------
    // Mirrors `scan.rs::build_scan_state_core`: look up declared partition
    // column names in the tupdesc (case-insensitive). Empty when the table
    // is not partitioned.
    let partition_attnums: Vec<usize> = if table_opts.partition_columns.is_empty() {
        Vec::new()
    } else {
        // SAFETY: tupdesc live; `tupdesc_attr` is the version-portable
        // accessor; `attname` is a fixed-size NameData with a NUL-terminated
        // C string in-bounds — same shape as `scan.rs`.
        unsafe {
            let tupdesc = (*rel).rd_att;
            let natts = (*tupdesc).natts as usize;
            let mut out = Vec::with_capacity(table_opts.partition_columns.len());
            for name in &table_opts.partition_columns {
                let mut found: Option<usize> = None;
                for i in 0..natts {
                    let att = crate::fdw::tupdesc_attr(tupdesc, i);
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

    // Storage schema: drop partition columns from the Arrow schema so the
    // parquet blob only carries non-partition columns (Hive convention —
    // partition values are in the path, not the file).
    let storage_attrs: Vec<pg_sys::FormData_pg_attribute> = attrs
        .iter()
        .enumerate()
        .filter_map(|(i, a)| {
            if partition_attnums.contains(&i) {
                None
            } else {
                Some(*a)
            }
        })
        .collect();
    let schema = pg_attrs_to_arrow_schema(&storage_attrs)?;

    let (base_prefix, legacy_single_target) = pick_target_layout(&table_opts);

    Ok(InsertState {
        schema,
        pg_oids,
        client,
        compression: table_opts.compression,
        partition_attnums,
        partition_keys_decl: table_opts.partition_keys.clone(),
        base_prefix,
        legacy_single_target,
        builders: HashMap::new(),
        writers: HashMap::new(),
        target_names: HashMap::new(),
        rows_in_current_batch: HashMap::new(),
    })
}

/// Derive (base_prefix, legacy_single_target) from the table option.
///
/// - Glob (`dir/*` or `dir/prefix*`): `base_prefix = "dir"` (or "dir/prefix"
///   without the `*`), trimmed of trailing `/`. No legacy single-target —
///   every group lands in a fresh `generate_blob_name`-style file.
/// - Literal filename: `base_prefix = dirname(fname)`; we ALSO record the
///   literal name as `legacy_single_target` so non-partitioned tables keep
///   writing to a single blob (matches pre-SP-3b behavior). Partitioned
///   tables ignore the literal target — they always synthesize per-group
///   names under the dirname prefix.
fn pick_target_layout(opts: &TableOptions) -> (String, Option<String>) {
    let fname = opts.filename.as_str();
    if let Some(star) = fname.find('*') {
        let prefix = fname[..star].trim_end_matches('/').to_string();
        (prefix, None)
    } else {
        let dir = match fname.rfind('/') {
            Some(idx) => fname[..idx].to_string(),
            None => String::new(),
        };
        (dir, Some(fname.to_string()))
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

    // 1) Derive the partition tuple key from this row's partition-column
    //    datums. Empty key (single group) for non-partitioned tables.
    let key = if state.partition_attnums.is_empty() {
        PartitionTupleKey { values: Vec::new() }
    } else {
        let mut vals: Vec<String> = Vec::with_capacity(state.partition_attnums.len());
        for &attno in &state.partition_attnums {
            // SAFETY: `attno < natts ≤ tts_nvalid` after `slot_getallattrs`.
            let is_null = unsafe { *isnulls.add(attno) };
            if is_null {
                return Err(FdwError::SchemaMismatch(format!(
                    "partition column attno {attno} cannot be NULL"
                )));
            }
            // SAFETY: same bound as `is_null`.
            let datum = unsafe { *values.add(attno) };
            // SAFETY: `datum` is a valid datum of type `pg_oids[attno]`;
            // `datum_to_partition_string`'s contract matches `append_one`.
            let s =
                unsafe { crate::convert::datum_to_partition_string(datum, state.pg_oids[attno])? };
            vals.push(s);
        }
        PartitionTupleKey { values: vals }
    };

    // 2) Ensure builders, target name, and row count exist for this group.
    let schema = state.schema.clone();
    state.builders.entry(key.clone()).or_insert_with(|| {
        RecordBatchBuilders::new(schema.clone(), BATCH_ROWS)
            .expect("RecordBatchBuilders::new should not fail for a validated schema")
    });
    if !state.target_names.contains_key(&key) {
        let name = compute_target_name(state, &key);
        state.target_names.insert(key.clone(), name);
    }
    let builders = state
        .builders
        .get_mut(&key)
        .expect("inserted above by or_insert_with");

    // 3) Walk slot columns, appending non-partition cols to the storage
    //    builders in storage-schema order. Partition cols are skipped — they
    //    live in the blob path, not the parquet data.
    let mut storage_col = 0usize;
    for (i, &oid) in state.pg_oids.iter().enumerate() {
        if state.partition_attnums.contains(&i) {
            continue;
        }
        // SAFETY: `i < state.pg_oids.len() ≤ tts_nvalid`.
        let is_null = unsafe { *isnulls.add(i) };
        // SAFETY: same bound.
        let datum = unsafe { *values.add(i) };
        // SAFETY: `(oid, datum, is_null)` describe one storage column;
        // `storage_col` indexes the storage-schema builders.
        unsafe { append_one(builders, storage_col, oid, datum, is_null) }?;
        storage_col += 1;
    }

    *state.rows_in_current_batch.entry(key).or_insert(0) += 1;
    Ok(())
}

/// Pick the target blob name for `key`. For non-partitioned tables with a
/// literal `filename` option, reuse the literal name (`legacy_single_target`).
/// Otherwise synthesize `generate_blob_name(base_prefix[/seg=val/...])`.
fn compute_target_name(state: &InsertState, key: &PartitionTupleKey) -> String {
    if key.values.is_empty() {
        if let Some(literal) = &state.legacy_single_target {
            return literal.clone();
        }
        return generate_blob_name(&state.base_prefix);
    }
    let mut path = state.base_prefix.clone();
    for (i, value) in key.values.iter().enumerate() {
        if !path.is_empty() && !path.ends_with('/') {
            path.push('/');
        }
        let name = state
            .partition_keys_decl
            .get(i)
            .map(|(n, _)| n.as_str())
            .unwrap_or("p");
        path.push_str(name);
        path.push('=');
        path.push_str(value);
    }
    generate_blob_name(&path)
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

/// Flush one group's in-progress builder into its per-group ParquetBatchWriter
/// (creating the writer lazily on first flush). Bounded by MAX_BLOB_BYTES on
/// the per-group accumulator.
fn flush_group(state: &mut InsertState, key: &PartitionTupleKey) -> FdwResult<()> {
    let n = state.rows_in_current_batch.get(key).copied().unwrap_or(0);
    if n == 0 {
        return Ok(());
    }
    let fresh = RecordBatchBuilders::new(state.schema.clone(), BATCH_ROWS)?;
    let old = state
        .builders
        .insert(key.clone(), fresh)
        .expect("group's builders present (created in append_slot)");
    let batch = old.finish()?;

    // Get-or-create the per-group parquet writer (lazy: empty groups never
    // allocate a writer).
    let compression = state.compression;
    let schema = state.schema.clone();
    if !state.writers.contains_key(key) {
        state
            .writers
            .insert(key.clone(), ParquetBatchWriter::new(schema, compression)?);
    }
    let writer = state.writers.get_mut(key).expect("inserted above");
    writer.write(&batch)?;

    // Same MAX_BLOB_BYTES guard as the pre-SP-3b single-writer path, applied
    // per group. With partition routing a runaway COPY could OOM ANY one
    // group; the check here fires once per group reaches the cap.
    let written = writer.bytes_written() as u64;
    if written > crate::azure::MAX_BLOB_BYTES {
        return Err(crate::error::FdwError::Azure(format!(
            "INSERT/COPY accumulator: group rendered {written} bytes — exceeds \
             MAX_BLOB_BYTES={cap}; split the load into multiple statements \
             or smaller batches (per-partition groups are bounded \
             independently)",
            cap = crate::azure::MAX_BLOB_BYTES
        )));
    }
    state.rows_in_current_batch.insert(key.clone(), 0);
    Ok(())
}

/// Test-only entry that drives the per-group finalize + upload path without
/// going through the executor / slot decoder. For each non-empty
/// `(key, [RecordBatch])` it constructs a per-group `ParquetBatchWriter`,
/// writes the batches, picks the per-group target blob name via the same
/// `compute_target_name` helper the production path uses, and uploads.
/// Returns the `(key, blob_name)` pairs that were actually uploaded so the
/// caller can assert per-tuple routing.
///
/// Used by `tests/partition_write.rs` to assert:
/// - multi-tuple routing produces one blob per distinct tuple key, at the
///   expected `base_prefix/key=val/.../{ts}-{uuid}.parquet` paths;
/// - empty groups (zero rows for a key) produce NO blob.
#[cfg(any(test, feature = "pg_test"))]
pub fn finalize_and_upload_for_test(
    schema: SchemaRef,
    compression: Compression,
    client: AzureBlobClient,
    base_prefix: String,
    legacy_single_target: Option<String>,
    partition_keys_decl: Vec<(String, PgPartitionType)>,
    groups: HashMap<PartitionTupleKey, Vec<arrow::array::RecordBatch>>,
) -> FdwResult<Vec<(PartitionTupleKey, String)>> {
    // Stub state purely for `compute_target_name`'s layout fields.
    let stub = InsertState {
        schema: schema.clone(),
        pg_oids: Vec::new(),
        client: client.clone(),
        compression,
        partition_attnums: Vec::new(),
        partition_keys_decl,
        base_prefix,
        legacy_single_target,
        builders: HashMap::new(),
        writers: HashMap::new(),
        target_names: HashMap::new(),
        rows_in_current_batch: HashMap::new(),
    };

    // Collect non-empty groups in sorted key order so the upload order (and
    // hence test-observable timestamps) is deterministic.
    let mut sorted: Vec<(PartitionTupleKey, Vec<arrow::array::RecordBatch>)> = groups
        .into_iter()
        .filter(|(_, batches)| batches.iter().any(|b| b.num_rows() > 0))
        .collect();
    sorted.sort_by(|a, b| a.0.values.cmp(&b.0.values));

    let mut uploaded: Vec<(PartitionTupleKey, String)> = Vec::with_capacity(sorted.len());
    for (key, batches) in sorted {
        let mut writer = ParquetBatchWriter::new(schema.clone(), compression)?;
        for b in &batches {
            writer.write(b)?;
        }
        let bytes = writer.finish()?;
        let target_name = compute_target_name(&stub, &key);
        let blob_writer = AzureBlobWriter::new(&client, &target_name);
        runtime::block_on(blob_writer.upload(bytes))?;
        uploaded.push((key, target_name));
    }
    Ok(uploaded)
}

/// Flush every group's tail batch, finalize each group's parquet writer, and
/// upload one blob per non-empty group. Groups that never had a row appended
/// (and were therefore never inserted into the maps) produce no upload —
/// this is the "empty group" case from the brief.
fn finalize_and_upload(mut state: InsertState) -> FdwResult<()> {
    // Tail flush for every group with rows still in builders.
    let keys: Vec<PartitionTupleKey> = state.builders.keys().cloned().collect();
    for key in &keys {
        if state.rows_in_current_batch.get(key).copied().unwrap_or(0) > 0 {
            flush_group(&mut state, key)?;
        }
    }

    // Drain by-value so we can `finish()` each writer (consumes self) and
    // upload. Sort by partition values so multi-group tests are stable.
    let InsertState {
        client,
        writers,
        target_names,
        ..
    } = state;
    let mut entries: Vec<(PartitionTupleKey, ParquetBatchWriter)> = writers.into_iter().collect();
    entries.sort_by(|a, b| a.0.values.cmp(&b.0.values));

    for (key, writer) in entries {
        let target_name = target_names.get(&key).cloned().ok_or_else(|| {
            FdwError::SchemaMismatch(format!(
                "no target_name registered for partition tuple key {:?}",
                key.values
            ))
        })?;
        let bytes = writer.finish()?;
        let blob_writer = AzureBlobWriter::new(&client, &target_name);
        runtime::block_on(blob_writer.upload(bytes))?;
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_target_layout_glob_strips_star_and_trailing_slash() {
        let opts = TableOptions {
            container: "c".into(),
            filename: "events/year=2026/*".into(),
            compression: Compression::Snappy,
            parallel_workers: None,
            partition_columns: vec!["year".into()],
            partition_keys: vec![("year".into(), PgPartitionType::Int4)],
            sorted: vec![],
            files_in_order: false,
        };
        let (prefix, literal) = pick_target_layout(&opts);
        assert_eq!(prefix, "events/year=2026");
        assert!(literal.is_none());
    }

    #[test]
    fn pick_target_layout_literal_preserves_full_name() {
        let opts = TableOptions {
            container: "c".into(),
            filename: "dir/one.parquet".into(),
            compression: Compression::Snappy,
            parallel_workers: None,
            partition_columns: vec![],
            partition_keys: vec![],
            sorted: vec![],
            files_in_order: false,
        };
        let (prefix, literal) = pick_target_layout(&opts);
        assert_eq!(prefix, "dir");
        assert_eq!(literal.as_deref(), Some("dir/one.parquet"));
    }

    fn make_stub_state(
        base_prefix: &str,
        legacy: Option<&str>,
        decls: Vec<(String, PgPartitionType)>,
    ) -> InsertState {
        use arrow::datatypes::{DataType, Field, Schema};
        let schema: SchemaRef =
            std::sync::Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, true)]));
        // SasUrl client is parsed-only; we never touch the network in this test.
        let client = crate::azure::AzureBlobClient::new(
            "fake.invalid",
            "fakeaccount",
            crate::azure::Credential::SasUrl {
                container_url: "http://127.0.0.1:1/c?sv=2024-11-04&sig=stub".into(),
            },
            "c",
        )
        .expect("client constructs from a parseable SAS URL");
        InsertState {
            schema,
            pg_oids: Vec::new(),
            client,
            compression: Compression::Snappy,
            partition_attnums: Vec::new(),
            partition_keys_decl: decls,
            base_prefix: base_prefix.to_string(),
            legacy_single_target: legacy.map(|s| s.to_string()),
            builders: HashMap::new(),
            writers: HashMap::new(),
            target_names: HashMap::new(),
            rows_in_current_batch: HashMap::new(),
        }
    }

    #[test]
    fn compute_target_name_partitioned_emits_key_eq_val_segments() {
        let state = make_stub_state(
            "events",
            None,
            vec![
                ("year".into(), PgPartitionType::Int4),
                ("region".into(), PgPartitionType::Text),
            ],
        );
        let key = PartitionTupleKey {
            values: vec!["2026".into(), "us".into()],
        };
        let name = compute_target_name(&state, &key);
        assert!(
            name.starts_with("events/year=2026/region=us/"),
            "got {name}"
        );
        assert!(name.ends_with(".parquet"));
    }

    #[test]
    fn compute_target_name_empty_key_reuses_literal_when_present() {
        let state = make_stub_state("dir", Some("dir/one.parquet"), Vec::new());
        let key = PartitionTupleKey { values: Vec::new() };
        let name = compute_target_name(&state, &key);
        assert_eq!(name, "dir/one.parquet");
    }

    #[test]
    fn compute_target_name_empty_key_glob_synthesizes_under_prefix() {
        let state = make_stub_state("dir", None, Vec::new());
        let key = PartitionTupleKey { values: Vec::new() };
        let name = compute_target_name(&state, &key);
        assert!(name.starts_with("dir/"), "got {name}");
        assert!(name.ends_with(".parquet"));
    }
}
