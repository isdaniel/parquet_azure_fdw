#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]
//! UPDATE / DELETE callbacks for the Parquet-on-Azure FDW.
//!
//! See design spec §3.2: `ExecForeignUpdate` / `ExecForeignDelete` are pure
//! accumulators — they record edits into a [`ModifyPlan`], and
//! `EndForeignModify` drives one download + rewrite + upload round per
//! affected blob via [`commit_plan`].
//!
//! ## Flow
//!
//! 1. `add_foreign_update_targets` injects a resjunk `ctid` column so the
//!    executor delivers the synthetic ctid we stamped during scan into each
//!    modify slot.
//! 2. `begin_foreign_modify` constructs the plan: re-runs glob expansion,
//!    HEADs each blob to learn its row count, builds a `BlobIdEntry` chunk
//!    table that mirrors the scan's, and stashes an empty `edits` map.
//! 3. Per-row `exec_foreign_{update,delete}` decode the resjunk `ctid`
//!    junk-attribute back into a `RowId`, resolve `(base_blob_id, abs_row)`,
//!    and append to `edits[base_blob_id]` (a `BlobEdits` of roaring deletes
//!    plus per-row updates).
//! 4. `commit_plan` (called from `end_foreign_modify`) GETs each touched
//!    blob with an `If-Match` etag, decodes batches, runs the pure
//!    [`apply_edits`] kernel, then either PUTs the new bytes or DELETEs the
//!    blob (when no rows remain) — both with the captured etag.
//!
//! ## Unsafe / FFI carve-out
//!
//! This file carries `#![allow(unsafe_code)]` plus
//! `#![deny(unsafe_op_in_unsafe_fn)]`; every `unsafe { ... }` block is
//! paired with a `// SAFETY:` comment.

use crate::azure::AzureBlobClient;
use crate::convert::pg_to_arrow::{pg_attrs_to_arrow_schema, RecordBatchBuilders};
use crate::error::{raise, FdwError, FdwResult};
use crate::fdw::modify::kernel::{apply_edits, BlobEdits, RowOverride};
use crate::fdw::modify::rowid::{RowId, CHUNK_ROWS};
use crate::fdw::modify::BlobIdEntry;
use crate::parquet_io::reader::{open_stream_from_bytes, ParquetReadOptions};
use crate::parquet_io::writer::ParquetBatchWriter;
use crate::parquet_io::Compression;
use crate::runtime;

use arrow::array::{ArrayRef, RecordBatch};
use arrow::datatypes::SchemaRef;
use futures::StreamExt;
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use pgrx::pg_sys;
use std::collections::HashMap;

/// Plan accumulated by `ExecForeignUpdate` / `ExecForeignDelete` and drained
/// by `EndForeignModify` via [`commit_plan`].
pub struct ModifyPlan {
    /// Per-(blob, 65_536-row chunk) entries. Mirrors the scan side: each
    /// source blob occupies a contiguous run starting at its base index, and
    /// `chunk_base_row` records the first absolute row each chunk represents.
    pub blob_table: Vec<BlobIdEntry>,
    /// Map from **base** blob_id (the source blob's first chunk index) to its
    /// accumulated edits.
    pub edits: HashMap<u32, BlobEdits>,
    /// Arrow schema reflecting the foreign relation's tuple descriptor.
    pub schema: SchemaRef,
    /// Per-attribute PG type OIDs in tupdesc order.
    pub pg_oids: Vec<pg_sys::Oid>,
    /// 0-based column indices that the UPDATE statement actually SETs. Empty
    /// for DELETE.
    pub update_attnums: Vec<usize>,
    /// Container-scoped Azure client (cheap to clone).
    pub client: AzureBlobClient,
    /// Compression codec to use for the rewritten parquet files.
    pub compression: Compression,
    /// `true` for DELETE, `false` for UPDATE. Recorded for assertions /
    /// diagnostics; the kernel itself doesn't need to know.
    #[allow(dead_code)]
    pub is_delete: bool,
    /// AttrNumber of the resjunk `ctid` column injected by
    /// [`add_foreign_update_targets`]. Located once in `build_plan` via
    /// `ExecFindJunkAttributeInTlist` over the outer subplan's targetlist;
    /// read every row in `exec_foreign_{update,delete}` via
    /// `ExecGetJunkAttribute`. This is the only path the executor uses to
    /// deliver our scan-stamped synthetic ctid to the modify callbacks — the
    /// plan slot's `tts_tid` field is only filled for real heap tables and
    /// would otherwise read as the all-ones `InvalidItemPointer` sentinel.
    pub ctid_attno: pg_sys::AttrNumber,
    /// Running total of edits (deletes + updates) recorded across all
    /// affected blobs. Bumped by `record_delete` / `record_update` and
    /// checked against [`MAX_EDITS_PER_STATEMENT`] so a runaway UPDATE
    /// (e.g. `UPDATE t SET v = v WHERE true` over a 100M-row foreign rel)
    /// is refused before its `RowOverride { values: Vec<Option<ArrayRef>> }`
    /// allocations OOM the backend. The accumulator is per-statement and
    /// only drained in `commit_plan`, so there is no other backstop.
    #[allow(dead_code)]
    pub edit_count: usize,
}

/// Per-statement cap on the number of UPDATE/DELETE edits the plan may
/// accumulate before `commit_plan` runs. Each recorded edit costs:
///   * a u64/u32 key in the per-blob HashMap / RoaringBitmap;
///   * for UPDATE: `vec![None; ncols]` (16 bytes/col) plus one single-row
///     Arrow `ArrayRef` per actually-SET column (~100 bytes header + value
///     bytes; the metadata alone dominates for sparse SETs);
///   * for DELETE: a few bits inside the RoaringBitmap (compact).
///
/// With ncols = 100 and worst-case update accounting (~2 KiB/row of
/// accumulator overhead), a cap of 10M edits bounds the accumulator at
/// roughly 20 GiB — already too much for a single statement but well above
/// any reasonable workload. Users with larger UPDATE/DELETE scopes should
/// split the work; future v2 will move the kernel to per-batch streaming
/// and the cap can rise (or vanish) then.
pub const MAX_EDITS_PER_STATEMENT: usize = 10_000_000;

// ---------- AddForeignUpdateTargets ----------------------------------------

/// `AddForeignUpdateTargets_function`. Inject a resjunk `ctid` column
/// matching `SelfItemPointerAttributeNumber` (-1) so the executor delivers
/// the synthetic ctid we stamped during scan into each modify slot.
///
/// # Safety
///
/// PG passes valid planner pointers; the parse tree's `targetList` is alive
/// for the duration of the planner callback.
pub unsafe extern "C-unwind" fn add_foreign_update_targets(
    root: *mut pg_sys::PlannerInfo,
    rtindex: pg_sys::Index,
    _target_rte: *mut pg_sys::RangeTblEntry,
    _target_relation: pg_sys::Relation,
) {
    // SAFETY: PG-supplied pointers are valid for the duration of the
    // callback. We only call documented planner helpers (`makeVar`,
    // `add_row_identity_var`).
    //
    // Use `add_row_identity_var` (the post-pg14 API) instead of manually
    // appending to `parse->targetList`. The latter "works" for UPDATE on
    // pg14 because a Result projection above the ForeignScan carries the
    // resjunk column through, but for DELETE the planner collapses to the
    // bare ForeignScan whose targetlist gets the resjunk dropped — leading
    // to an empty plan_slot at the ExecForeignDelete callback. The
    // row-identity API exists precisely to register an extra column the
    // planner is *required* to keep visible on the modify subplan output,
    // and `postgres_fdw` switched to it for the same reason.
    unsafe {
        let attr = pg_sys::SelfItemPointerAttributeNumber as pg_sys::AttrNumber;
        // Build a Var referencing the system ctid column of the foreign
        // rel. varlevelsup=0, collation invalid (ctid is not collatable),
        // typmod -1. `rtindex` is typed as u32 on pg14 and i32 on pg15+;
        // try_into handles both.
        #[allow(clippy::useless_conversion)]
        let var = pg_sys::makeVar(
            rtindex.try_into().unwrap_or(0),
            attr,
            pg_sys::TIDOID,
            -1,
            pg_sys::InvalidOid,
            0,
        );
        let cname = std::ffi::CString::new("ctid").expect("static literal contains no NUL bytes");
        pg_sys::add_row_identity_var(root, var, rtindex, cname.as_ptr());
    }
}

// ---------- ModifyPlan construction ----------------------------------------

/// Build a [`ModifyPlan`] from the current `ModifyTableState`. Called from
/// `begin_foreign_modify` when `operation != CMD_INSERT`.
///
/// # Safety
///
/// `mtstate` and `rinfo` are valid executor pointers; the relation and the
/// subplan are alive for the call.
pub unsafe fn build_plan(
    mtstate: *mut pg_sys::ModifyTableState,
    rinfo: *mut pg_sys::ResultRelInfo,
    is_delete: bool,
) -> FdwResult<ModifyPlan> {
    // SAFETY: caller invariants for the FDW callback contract.
    let rel = unsafe { (*rinfo).ri_RelationDesc };
    // SAFETY: live relation pointer.
    let relid = unsafe { (*rel).rd_id };

    // --- options + credentials + client ---------------------------------
    // SAFETY: documented PG catalog accessors via the (now pub(crate))
    // helper shared with the INSERT path.
    let (server_opts, um_opts, table_opts) =
        unsafe { crate::fdw::modify::insert::read_all_options(relid) }?;
    crate::fdw::options::validate_combo(&server_opts, &um_opts)?;

    let cred = crate::azure::build_credential(
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

    // --- glob expansion + per-blob chunk table --------------------------
    // PRIMARY: consume the scan-time blob list (with etags captured at
    // BeginForeignScan) so blob_id indexes match the ctids the executor will
    // deliver back to us AND the etags pin the scan's snapshot.
    //
    // The fallback to a fresh re-list is **only compiled into test builds**
    // (`cfg(any(test, feature = "pg_test"))`). It exists for unit-test
    // harnesses that drive `build_plan` without a real scan. Production
    // (release) builds raise `SchemaMismatch` if the handoff is missing,
    // making it physically impossible to silently bypass the lost-update
    // guard — see CLAUDE.md → "Architectural invariants".
    let blobs: Vec<(String, String)> = match crate::fdw::modify::scan_handoff::take_unique(relid)? {
        Some(v) => v,
        None => {
            #[cfg(any(test, feature = "pg_test"))]
            {
                crate::fdw::scan::expand_glob_for_modify(&client, &table_opts.filename)?
            }
            #[cfg(not(any(test, feature = "pg_test")))]
            {
                return Err(FdwError::SchemaMismatch(format!(
                    "no scan-time blob list published for relid {}; \
                     UPDATE/DELETE requires a preceding ForeignScan that \
                     publishes etags via scan_handoff",
                    relid.to_u32()
                )));
            }
        }
    };

    let mut blob_table: Vec<BlobIdEntry> = Vec::with_capacity(blobs.len());
    for (blob_name, etag) in &blobs {
        let reader = client.open_blob(blob_name);
        let nrows = runtime::block_on(blob_row_count(reader))?;
        // Reject blobs we cannot represent in the u32 row-index space the
        // delete bitmap / RowId encoding require. RoaringBitmap is u32; the
        // RowId encoding caps offsets at u32::MAX. We surface this once
        // here instead of silently truncating later in `record_*`.
        u32::try_from(nrows).map_err(|_| {
            FdwError::SchemaMismatch(format!(
                "blob {blob_name} has {nrows} rows; UPDATE/DELETE in v1 supports up to {} rows per blob",
                u32::MAX
            ))
        })?;
        let chunks = nrows.div_ceil(CHUNK_ROWS).max(1);
        for chunk in 0..chunks {
            blob_table.push(BlobIdEntry {
                name: blob_name.clone(),
                chunk_base_row: chunk * CHUNK_ROWS,
                etag: etag.clone(),
            });
        }
    }

    // --- schema + attribute OIDs ----------------------------------------
    // SAFETY: tupdesc is live for the relation; `tupdesc_attr` is the
    // version-portable accessor (pg18 reshaped the layout).
    let (attrs, pg_oids) = unsafe {
        let tupdesc = (*rel).rd_att;
        let natts = (*tupdesc).natts as usize;
        let mut attrs = Vec::with_capacity(natts);
        let mut oids = Vec::with_capacity(natts);
        for i in 0..natts {
            let att = crate::fdw::tupdesc_attr(tupdesc, i);
            if (*att).attisdropped {
                return Err(FdwError::SchemaMismatch(format!(
                    "dropped column at attnum {}",
                    (*att).attnum
                )));
            }
            attrs.push(*att);
            oids.push((*att).atttypid);
        }
        (attrs, oids)
    };
    let schema = pg_attrs_to_arrow_schema(&attrs)?;

    // --- UPDATE-target columns ------------------------------------------
    // For UPDATE we need to know which columns the statement actually SETs.
    // The cross-version shim in `crate::fdw::update_cols_for_subplan` reads
    // `mtstate->mt_updateColnosLists` for our subplan and returns the
    // 0-based attnums; DELETE never has SET columns.
    //
    // INVARIANT: these attnums are used directly as arrow-column indices in
    // `exec_foreign_update` and `apply_edits`. That mapping is correct only
    // because the schema-build loop above (`for i in 0..natts`) rejects any
    // dropped column up front — if you ever relax that rejection, you MUST
    // also remap `update_attnums` through the live-column projection here,
    // or the writes silently land in the wrong arrow column.
    let update_attnums = if is_delete {
        Vec::new()
    } else {
        // SAFETY: executor-supplied pointers per the build_plan contract;
        // `rinfo` lives inside `mtstate->resultRelInfo`.
        unsafe { crate::fdw::update_cols_for_subplan(mtstate, rinfo) }
    };

    // --- resjunk ctid AttrNumber ----------------------------------------
    // The executor delivers the synthetic ctid we stamped during scan as a
    // resjunk column on the modify subplan's plan slot. On foreign tables
    // the plan slot's `tts_tid` field is NOT populated (that's a heap-AM
    // contract), so we must fetch the ctid datum by AttrNumber via
    // `ExecGetJunkAttribute`. Mirrors `postgres_fdw`'s create_foreign_modify.
    //
    // We locate the attno by name from the subplan's targetlist:
    //   * For UPDATE on pg14, the subplan above the ForeignScan is a Result
    //     node whose `plan.targetlist` carries the NEW expressions plus our
    //     resjunk "ctid" TLE.
    //   * For DELETE on pg14, the planner may collapse the subplan down to
    //     the ForeignScan itself with `plan.targetlist` NIL (it intends the
    //     scan to emit a "physical" tuple). In that case we fall back to
    //     walking the plan_slot's tupdesc by attname at the per-row callback.
    //     Here we record `InvalidAttrNumber` as a sentinel.
    //
    // SAFETY: `mtstate->ps.lefttree` is the outer subplan's `PlanState` (the
    // `outerPlanState(mtstate)` macro in C). Its `.plan` is the subplan's
    // `Plan` node. All four pointers are executor-owned and live for the
    // duration of `begin_foreign_modify`.
    let ctid_attno = unsafe {
        let outer_ps = (*mtstate).ps.lefttree;
        if outer_ps.is_null() {
            return Err(FdwError::SchemaMismatch(
                "ModifyTableState has no outer PlanState (subplan)".into(),
            ));
        }
        let subplan = (*outer_ps).plan;
        if subplan.is_null() {
            return Err(FdwError::SchemaMismatch(
                "outer PlanState has no Plan node".into(),
            ));
        }
        let tlist = (*subplan).targetlist;
        if tlist.is_null() {
            // pg14 DELETE collapse case — see comment above. Defer lookup.
            pg_sys::InvalidAttrNumber as pg_sys::AttrNumber
        } else {
            let cname =
                std::ffi::CString::new("ctid").expect("static literal contains no NUL bytes");
            pg_sys::ExecFindJunkAttributeInTlist(tlist, cname.as_ptr())
        }
    };
    // `ctid_attno` may legitimately be `InvalidAttrNumber` when the pg14
    // DELETE planner collapsed the subplan to a ForeignScan with NIL
    // `plan.targetlist`. In that case we resolve the attno per-row inside
    // `read_ctid_via_junk` by walking the plan_slot's tupdesc by attname.

    Ok(ModifyPlan {
        blob_table,
        edits: HashMap::new(),
        schema,
        pg_oids,
        update_attnums,
        client,
        compression: table_opts.compression,
        is_delete,
        ctid_attno,
        edit_count: 0,
    })
}

/// Open just enough of a parquet blob to read its footer and return the row
/// count. The async file reader fetches the footer range only, so this is
/// effectively one or two HTTP range-GETs per blob.
async fn blob_row_count(reader: crate::azure::AzureBlobReader) -> FdwResult<u64> {
    let builder = ParquetRecordBatchStreamBuilder::new(reader).await?;
    let nrows = builder.metadata().file_metadata().num_rows();
    Ok(nrows.max(0) as u64)
}

// ---------- ExecForeignUpdate / ExecForeignDelete --------------------------

/// `ExecForeignDelete_function` — pure accumulator. Fetches the row's ctid
/// from the resjunk `ctid` junk attribute on `plan_slot` (wired up by
/// `add_foreign_update_targets` and copied through the executor's
/// `JunkFilter`), maps to `(base_blob_id, abs_row)`, and records the row
/// index in `edits[base_blob_id].deletes`.
///
/// # Safety
///
/// `rinfo->ri_FdwState` was populated by `begin_foreign_modify`. `plan_slot`
/// is owned by the executor for this call and its `tts_tid` carries the
/// synthetic ctid we stamped during scan.
pub unsafe extern "C-unwind" fn exec_foreign_delete(
    _estate: *mut pg_sys::EState,
    rinfo: *mut pg_sys::ResultRelInfo,
    slot: *mut pg_sys::TupleTableSlot,
    plan_slot: *mut pg_sys::TupleTableSlot,
) -> *mut pg_sys::TupleTableSlot {
    // SAFETY: see fn-level safety; pointer was produced by `Box::into_raw`
    // in `begin_foreign_modify`.
    let state_ptr = unsafe { (*rinfo).ri_FdwState as *mut crate::fdw::modify::FdwModifyState };
    if state_ptr.is_null() {
        raise(FdwError::SchemaMismatch(
            "exec_foreign_delete called without initialised ri_FdwState".into(),
        ));
    }
    // SAFETY: non-null per the check above.
    let state: &mut crate::fdw::modify::FdwModifyState = unsafe { &mut *state_ptr };
    let plan = match state {
        crate::fdw::modify::FdwModifyState::Update(p) => p,
        _ => raise(FdwError::SchemaMismatch(
            "exec_foreign_delete on non-update state".into(),
        )),
    };
    // SAFETY: materialize all attributes on the plan slot so junk attrs are
    // populated in `tts_values` before `ExecGetJunkAttribute` reads them.
    unsafe {
        pg_sys::slot_getallattrs(plan_slot);
    }
    // SAFETY: `plan_slot` is owned by the executor; `plan.ctid_attno` was
    // located in `build_plan` via `ExecFindJunkAttributeInTlist`.
    let rid = match unsafe { read_ctid_via_junk(plan_slot, plan.ctid_attno) } {
        Ok(r) => r,
        Err(e) => raise(e),
    };
    if let Err(e) = record_delete(plan, rid) {
        raise(e);
    }
    slot
}

/// `ExecForeignUpdate_function` — pure accumulator. Fetches the row's ctid
/// from the resjunk `ctid` junk attribute on `plan_slot` and the new column
/// values for `update_attnums` from the regular `slot` columns, encodes each
/// as a single-row arrow array, and records into `edits[base_id].updates`.
///
/// # Safety
///
/// `rinfo->ri_FdwState` was populated by `begin_foreign_modify`. `plan_slot`
/// is owned by the executor for this call.
pub unsafe extern "C-unwind" fn exec_foreign_update(
    _estate: *mut pg_sys::EState,
    rinfo: *mut pg_sys::ResultRelInfo,
    slot: *mut pg_sys::TupleTableSlot,
    plan_slot: *mut pg_sys::TupleTableSlot,
) -> *mut pg_sys::TupleTableSlot {
    // SAFETY: contract identical to `exec_foreign_delete`.
    let state_ptr = unsafe { (*rinfo).ri_FdwState as *mut crate::fdw::modify::FdwModifyState };
    if state_ptr.is_null() {
        raise(FdwError::SchemaMismatch(
            "exec_foreign_update called without initialised ri_FdwState".into(),
        ));
    }
    // SAFETY: non-null per the check above.
    let state: &mut crate::fdw::modify::FdwModifyState = unsafe { &mut *state_ptr };
    let plan = match state {
        crate::fdw::modify::FdwModifyState::Update(p) => p,
        _ => raise(FdwError::SchemaMismatch(
            "exec_foreign_update on non-update state".into(),
        )),
    };

    // SAFETY: materialize all attributes before reading tts_values.
    // The `slot` arg (NOT `plan_slot`) carries the NEW values projected into
    // the target relation's tupdesc; per-column reads index by the target
    // rel's 0-based attnum, which is exactly what `update_attnums` holds.
    // `plan_slot` instead carries the raw subplan output (whose layout is
    // SET-expression order + appended resjunk junk columns) and is used
    // here to fetch the resjunk ctid via `ExecGetJunkAttribute` below; that
    // accessor reads `tts_values[attno-1]`, so we materialize plan_slot too.
    unsafe {
        pg_sys::slot_getallattrs(slot);
        pg_sys::slot_getallattrs(plan_slot);
    }
    // SAFETY: `plan_slot` is owned by the executor; `plan.ctid_attno` was
    // located in `build_plan` via `ExecFindJunkAttributeInTlist`.
    let rid = match unsafe { read_ctid_via_junk(plan_slot, plan.ctid_attno) } {
        Ok(r) => r,
        Err(e) => raise(e),
    };
    // Build a RowOverride: one single-row arrow array per column actually
    // updated, with None placeholders for columns left untouched.
    let ncols = plan.schema.fields().len();
    let mut values: Vec<Option<ArrayRef>> = vec![None; ncols];
    for &col in &plan.update_attnums {
        if col >= ncols {
            raise(FdwError::SchemaMismatch(format!(
                "update_attnums col {col} >= tupdesc natts {ncols}"
            )));
        }
        // SAFETY: `tts_values` / `tts_isnull` are arrays sized to natts.
        let (datum, is_null) = unsafe {
            (
                *((*slot).tts_values.add(col)),
                *((*slot).tts_isnull.add(col)),
            )
        };
        // SAFETY: `(datum, oid)` is the same pair the INSERT path hands to
        // `append_one`; type routing is identical.
        let one = match unsafe {
            datum_to_single_row_array(plan.pg_oids[col], datum, is_null, &plan.schema, col)
        } {
            Ok(a) => a,
            Err(e) => raise(e),
        };
        values[col] = Some(one);
    }
    if let Err(e) = record_update(plan, rid, RowOverride { values }) {
        raise(e);
    }
    slot
}

/// Read the synthetic ctid out of the plan slot via its resjunk `ctid`
/// column. `ctid_attno` is the 1-based AttrNumber stashed in `ModifyPlan`
/// during `build_plan`; when the planner left the subplan's targetlist NIL
/// (pg14 DELETE collapse), we fall back to a tupdesc walk to find the
/// column named "ctid".
///
/// # Safety
///
/// `slot` is a live `TupleTableSlot` owned by the executor for this call.
unsafe fn read_ctid_via_junk(
    slot: *mut pg_sys::TupleTableSlot,
    ctid_attno: pg_sys::AttrNumber,
) -> FdwResult<RowId> {
    // Resolve the attno, walking the tupdesc by name if the planner left it
    // invalid in build_plan.
    let attno = if ctid_attno == pg_sys::InvalidAttrNumber as pg_sys::AttrNumber {
        // SAFETY: `tts_tupleDescriptor` is non-null on a live slot; we read
        // each `pg_attribute.attname` via the version-portable helper
        // `crate::fdw::tupdesc_attr` and compare the leading bytes against
        // the NUL-terminated literal "ctid".
        let found = unsafe {
            let td = (*slot).tts_tupleDescriptor;
            let natts = (*td).natts as usize;
            let mut found: pg_sys::AttrNumber = pg_sys::InvalidAttrNumber as pg_sys::AttrNumber;
            for i in 0..natts {
                let att = crate::fdw::tupdesc_attr(td, i);
                if (*att).attisdropped {
                    continue;
                }
                let name_ptr = (*att).attname.data.as_ptr() as *const u8;
                let mut matches = true;
                for (j, &b) in b"ctid\0".iter().enumerate() {
                    if *name_ptr.add(j) != b {
                        matches = false;
                        break;
                    }
                }
                if matches {
                    found = (i as i32 + 1) as pg_sys::AttrNumber;
                    break;
                }
            }
            found
        };
        if found == pg_sys::InvalidAttrNumber as pg_sys::AttrNumber {
            return Err(FdwError::SchemaMismatch(
                "could not find resjunk ctid column on plan slot tupdesc".into(),
            ));
        }
        found
    } else {
        ctid_attno
    };
    let mut is_null: bool = false;
    // SAFETY: `ExecGetJunkAttribute` is the documented accessor for junk
    // attributes; it reads tts_values[attno-1] under the hood and yields the
    // raw Datum. Passing a stack `&mut bool` is the standard pattern.
    let datum = unsafe { pg_sys::ExecGetJunkAttribute(slot, attno, &mut is_null as *mut bool) };
    if is_null {
        return Err(FdwError::SchemaMismatch(
            "resjunk ctid is NULL on UPDATE/DELETE plan slot".into(),
        ));
    }
    // SAFETY: a TIDOID Datum is a pointer to a palloc'd `ItemPointerData`
    // whose lifetime spans this callback (the executor's per-tuple slot
    // owns it). Copy by value out before returning to detach from PG memory.
    let ip_ptr = datum.cast_mut_ptr::<pg_sys::ItemPointerData>();
    if ip_ptr.is_null() {
        return Err(FdwError::SchemaMismatch(
            "resjunk ctid datum unexpectedly null".into(),
        ));
    }
    // SAFETY: `ip_ptr` is non-null (checked above) and points to a valid
    // `ItemPointerData` owned by the executor's tuple slot; we copy by value
    // before returning so the caller does not retain the borrow.
    let ip: pg_sys::ItemPointerData = unsafe { *ip_ptr };
    Ok(RowId::from_ctid(ip))
}

// ---------- record_{delete,update} -----------------------------------------

fn record_delete(plan: &mut ModifyPlan, rid: RowId) -> FdwResult<()> {
    check_edit_budget(plan)?;
    let (base_id, abs_row) = resolve_to_source(plan, rid)?;
    let entry = plan.edits.entry(base_id).or_default();
    // abs_row fits in u32 because build_plan rejected blobs with > u32::MAX
    // rows.
    let abs32 = abs_row as u32;
    entry.deletes.insert(abs32);
    plan.edit_count = plan.edit_count.saturating_add(1);
    Ok(())
}

fn record_update(plan: &mut ModifyPlan, rid: RowId, ovr: RowOverride) -> FdwResult<()> {
    check_edit_budget(plan)?;
    let (base_id, abs_row) = resolve_to_source(plan, rid)?;
    let entry = plan.edits.entry(base_id).or_default();
    // abs_row fits in u32 because build_plan rejected blobs with > u32::MAX
    // rows. The updates map keys on u64 to keep the kernel index space
    // symmetric with arrow row offsets.
    entry.updates.insert(abs_row, ovr);
    plan.edit_count = plan.edit_count.saturating_add(1);
    Ok(())
}

/// Refuse to accept another edit if the per-statement accumulator cap would
/// be exceeded. The check is "fail just before recording" rather than "fail
/// just after" so the rejected row is the one that triggers the error — the
/// accumulator's contents are exactly MAX_EDITS_PER_STATEMENT when we bail,
/// which keeps the error message and the accounting consistent.
fn check_edit_budget(plan: &ModifyPlan) -> FdwResult<()> {
    if plan.edit_count >= MAX_EDITS_PER_STATEMENT {
        return Err(FdwError::SchemaMismatch(format!(
            "UPDATE/DELETE accumulator reached MAX_EDITS_PER_STATEMENT={cap}; \
             split the statement (the kernel materialises one `RowOverride` per \
             row before commit, so an uncapped multi-million-row UPDATE OOMs the \
             backend before commit_plan runs)",
            cap = MAX_EDITS_PER_STATEMENT
        )));
    }
    Ok(())
}

/// Walk back from the ctid's `blob_id` (a *chunk* id) to the *base* blob_id
/// of the source blob and the absolute row within that source blob. Errors
/// rather than panics if the ctid's `blob_id` is out of range — the executor
/// could deliver a stale or out-of-band ctid (e.g. across a ReScan) and we
/// don't want to abort with a Postgres panic across the FFI boundary.
fn resolve_to_source(plan: &ModifyPlan, rid: RowId) -> FdwResult<(u32, u64)> {
    let idx = rid.blob_id as usize;
    let entry = plan.blob_table.get(idx).ok_or_else(|| {
        FdwError::SchemaMismatch(format!(
            "ctid blob_id {} out of range (blob_table len {})",
            rid.blob_id,
            plan.blob_table.len()
        ))
    })?;
    let name = entry.name.clone();
    let mut base = rid.blob_id as i64;
    while base > 0 {
        let prev = plan
            .blob_table
            .get((base - 1) as usize)
            .ok_or_else(|| FdwError::SchemaMismatch("blob_table prev index underflow".into()))?;
        if prev.name != name {
            break;
        }
        base -= 1;
    }
    let base = base as u32;
    let chunk_base_row = entry.chunk_base_row;
    Ok((base, chunk_base_row + rid.offset as u64))
}

/// Decode `(oid, datum)` into a single-row Arrow `ArrayRef` by routing
/// through the same `append_one` dispatch table the INSERT path uses.
///
/// # Safety
///
/// `datum` is the raw datum from the plan slot's `tts_values` for type
/// `oid`. We construct a single-column, single-row batch and call into the
/// crate-local `append_one`, which only touches PG memory through
/// documented `DatumGet*` patterns.
unsafe fn datum_to_single_row_array(
    oid: pg_sys::Oid,
    datum: pg_sys::Datum,
    is_null: bool,
    schema: &SchemaRef,
    col: usize,
) -> FdwResult<ArrayRef> {
    let one_field = schema.field(col).clone();
    let one_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![one_field]));
    let mut b = RecordBatchBuilders::new(one_schema.clone(), 1)?;
    // SAFETY: same routing as `insert::append_slot` would do for a single
    // column. `append_one` reads `datum` according to `oid` only.
    unsafe { crate::fdw::modify::insert::append_one(&mut b, 0, oid, datum, is_null) }?;
    let rb = b.finish()?;
    Ok(rb.column(0).clone())
}

// ---------- commit_plan ----------------------------------------------------

/// Drive the rewrite for every dirty blob. Called from `EndForeignModify`
/// on the COMMIT path.
///
/// Per design spec §3.4 ("best-effort"): processes blobs in sorted base_id
/// order; the first I/O error short-circuits via `?` and aborts the
/// statement — subsequent blobs are not touched.
pub fn commit_plan(plan: ModifyPlan) -> FdwResult<()> {
    use crate::fdw::modify::coordinator::{make_staging_name, with_active};

    // NOTE: E-T7 will install `open_statement` / `close_statement_success`
    // at `begin_foreign_modify` / `end_foreign_modify`. Until then,
    // `with_active(...)` here returns None and the
    // `register_staging` / `mark_committed` book-keeping is a no-op — the
    // two-phase ordering itself (stage-all, then swap-all) is what's
    // load-bearing for atomicity; the coordinator only enables the
    // xact-abort sweep, which is wired by E-T7.

    let mut keys: Vec<u32> = plan.edits.keys().copied().collect();
    keys.sort_unstable();

    // PHASE 1 — STAGE: for each blob, GET-with-etag, apply_edits, PUT to a
    // `*.tmp.<uuid>.parquet` via If-None-Match. If any blob fails, the xact
    // callback sweeps registered staging names on abort.
    struct Staged {
        original_name: String,
        scan_etag: String,
        staging_name: String,
        new_bytes_for_swap: bytes::Bytes,
        // true → after-edit result is 0 rows; swap means DELETE original
        // instead of PUT (no staging blob written for this entry).
        empty_delete: bool,
    }
    let mut staged: Vec<Staged> = Vec::with_capacity(keys.len());

    for base_id in &keys {
        let entry = plan.blob_table.get(*base_id as usize).ok_or_else(|| {
            FdwError::SchemaMismatch(format!(
                "edits reference blob_id {base_id} outside blob_table (len {})",
                plan.blob_table.len()
            ))
        })?;
        let name = entry.name.clone();
        let scan_etag = entry.etag.clone();
        let edits = plan.edits.get(base_id).expect("present");

        // 1) GET blob bytes with `If-Match: scan_etag`. 412 → SQLSTATE 40001
        //    (the blob changed since the SELECT snapshot — surface, don't
        //    silently overwrite with stale ctids).
        let body = runtime::block_on(plan.client.get_body_if_match(&name, &scan_etag))?;

        // 2) Decode into RecordBatches. Bound the DECODED size too —
        //    MAX_BLOB_BYTES on the encoded body isn't sufficient because
        //    parquet routinely compresses 5x-20x for dictionary-encoded or
        //    repetitive columns, and apply_edits' peak RSS scales with
        //    decoded size.
        let mut s = runtime::block_on(open_stream_from_bytes(body, ParquetReadOptions::default()))?;
        let mut batches: Vec<RecordBatch> = Vec::new();
        let mut decoded_bytes: u64 = 0;
        while let Some(batch) = runtime::block_on(s.next()) {
            let batch = batch?;
            decoded_bytes = decoded_bytes.saturating_add(batch.get_array_memory_size() as u64);
            if decoded_bytes > crate::azure::MAX_BLOB_BYTES {
                return Err(FdwError::Azure(format!(
                    "blob '{name}' decodes to >{cap} bytes of arrow memory \
                     (parquet compression unfolds in RAM during \
                     UPDATE/DELETE rewrite); refusing to materialise",
                    cap = crate::azure::MAX_BLOB_BYTES
                )));
            }
            batches.push(batch);
        }

        // 3) Apply edits via the pure kernel.
        let out = apply_edits(batches, plan.schema.clone(), edits)?;
        let total_rows: usize = out.iter().map(|b| b.num_rows()).sum();

        if total_rows == 0 {
            // No staging blob — Phase 2 will DELETE the original under
            // If-Match: scan_etag.
            staged.push(Staged {
                original_name: name,
                scan_etag,
                staging_name: String::new(),
                new_bytes_for_swap: bytes::Bytes::new(),
                empty_delete: true,
            });
            continue;
        }

        // 4) Encode new parquet bytes and stage under a `*.tmp.<uuid>` name
        //    via If-None-Match (so we never overwrite an existing blob).
        let mut writer = ParquetBatchWriter::new(plan.schema.clone(), plan.compression)?;
        for b in &out {
            writer.write(b)?;
        }
        let bytes = writer.finish()?;

        let staging_name = make_staging_name(&name);
        // Register BEFORE the write — if the write itself fails partway,
        // the half-uploaded blob (or fully uploaded blob that we then
        // failed to record) is still tracked and the xact-abort sweep can
        // delete it.
        with_active(|c, _| c.register_staging(staging_name.clone()));
        runtime::block_on(plan.client.put_if_none_match(&staging_name, bytes.clone()))?;

        staged.push(Staged {
            original_name: name,
            scan_etag,
            staging_name,
            new_bytes_for_swap: bytes,
            empty_delete: false,
        });
    }

    // PHASE 2 — SWAP: for each staged item, PUT the new bytes onto the
    // original name with If-Match: scan_etag (or DELETE the original for
    // empty_delete), then drop the staging blob. A 412 here aborts the
    // statement; the xact callback cleans up the remaining staging blobs.
    for s in &staged {
        if s.empty_delete {
            runtime::block_on(plan.client.delete_if_match(&s.original_name, &s.scan_etag))?;
            continue;
        }
        runtime::block_on(plan.client.put_if_match(
            &s.original_name,
            s.new_bytes_for_swap.clone(),
            &s.scan_etag,
        ))?;
        // Original successfully swapped — drop the staging blob and clear
        // it from the in-flight set so xact-abort doesn't try to re-delete
        // it. delete_unconditional is best-effort here; if it fails the
        // blob still has a `*.tmp.*` name and remains a known orphan, but
        // the user data is committed.
        let _ = runtime::block_on(plan.client.delete_unconditional(&s.staging_name));
        with_active(|c, _| c.mark_committed(&s.staging_name));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::azure::Credential;
    use arrow::datatypes::{DataType, Field, Schema};

    fn make_test_client() -> AzureBlobClient {
        // SAS-URL variant only parses the URL; no network access happens
        // until a method is called. record_delete / record_update never
        // touch the client, so a parseable placeholder is sufficient.
        AzureBlobClient::new(
            "fake.invalid",
            "fakeaccount",
            Credential::SasUrl {
                container_url: "http://127.0.0.1:1/c?sv=2024-11-04&sig=stub".into(),
            },
            "c",
        )
        .expect("test client constructs from a parseable SAS URL")
    }

    fn make_plan_with_one_blob() -> ModifyPlan {
        let schema: SchemaRef = std::sync::Arc::new(Schema::new(vec![
            Field::new("i", DataType::Int32, true),
            Field::new("s", DataType::Utf8, true),
        ]));
        ModifyPlan {
            blob_table: vec![BlobIdEntry {
                name: "x.parquet".into(),
                chunk_base_row: 0,
                etag: "e".into(),
            }],
            edits: HashMap::new(),
            schema,
            pg_oids: vec![],
            update_attnums: vec![],
            client: make_test_client(),
            compression: crate::parquet_io::Compression::Snappy,
            is_delete: true,
            ctid_attno: 0,
            edit_count: 0,
        }
    }

    // The accumulator cap fires BEFORE the record is inserted, so the
    // accumulator's contents on error are exactly MAX_EDITS_PER_STATEMENT —
    // not MAX+1. That keeps the error message ("reached MAX...") and the
    // accounting consistent and means a caller observing the error has a
    // stable picture of the rejected row.
    #[test]
    fn record_delete_rejects_beyond_cap() {
        let mut plan = make_plan_with_one_blob();
        plan.edit_count = MAX_EDITS_PER_STATEMENT;
        // RowId::from_absolute(0, 0) — blob 0, abs row 0; resolves successfully
        // against the single-entry blob_table.
        let rid = crate::fdw::modify::rowid::RowId::from_absolute(0, 0);
        let err = record_delete(&mut plan, rid).expect_err("at cap, must reject");
        match err {
            FdwError::SchemaMismatch(msg) => {
                assert!(msg.contains("MAX_EDITS_PER_STATEMENT"), "{msg}");
            }
            other => panic!("expected SchemaMismatch, got {other:?}"),
        }
        // Cap-fail did not bump the counter past MAX.
        assert_eq!(plan.edit_count, MAX_EDITS_PER_STATEMENT);
        // The deletes bitmap was NOT touched.
        assert!(plan
            .edits
            .get(&0)
            .map(|e| e.deletes.is_empty())
            .unwrap_or(true));
    }

    #[test]
    fn record_update_rejects_beyond_cap() {
        let mut plan = make_plan_with_one_blob();
        plan.edit_count = MAX_EDITS_PER_STATEMENT;
        let rid = crate::fdw::modify::rowid::RowId::from_absolute(0, 0);
        let ovr = RowOverride {
            values: vec![None, None],
        };
        let err = record_update(&mut plan, rid, ovr).expect_err("at cap, must reject");
        assert!(matches!(err, FdwError::SchemaMismatch(_)));
        assert_eq!(plan.edit_count, MAX_EDITS_PER_STATEMENT);
    }

    #[test]
    fn record_delete_increments_count() {
        let mut plan = make_plan_with_one_blob();
        assert_eq!(plan.edit_count, 0);
        for i in 0..5 {
            let rid = crate::fdw::modify::rowid::RowId::from_absolute(0, i);
            record_delete(&mut plan, rid).expect("under cap");
        }
        assert_eq!(plan.edit_count, 5);
        assert_eq!(plan.edits.get(&0).expect("entry").deletes.len(), 5);
    }
}
