#![deny(unsafe_code)]
//! Foreign-data-wrapper glue: option parsing, scan/modify callbacks,
//! and pushdown translation. The `scan` submodule is the FFI boundary to
//! Postgres' executor and is the only place where `unsafe` is permitted
//! (it carries `#![deny(unsafe_op_in_unsafe_fn)]` so each unsafe op is
//! still individually opted into).

#[allow(unsafe_code)]
pub mod modify;
pub mod options;
pub mod pg_op_oids;
pub mod pushdown;
#[allow(unsafe_code)]
pub mod pushdown_walk;
#[allow(unsafe_code)]
pub mod scan;

use pgrx::{pg_sys, AllocatedByRust, PgBox};

/// Allocate an `FdwRoutine` node and wire up every callback we implement.
///
/// PostgreSQL's planner/executor calls into this struct via function pointers,
/// so each field is `Some(...)` for behaviour we support and `None` (the
/// `alloc_node` default) for operations we don't (UPDATE/DELETE, EXPLAIN,
/// foreign joins, batch insert, etc.).
#[allow(unsafe_code)]
pub fn build_routine() -> PgBox<pg_sys::FdwRoutine, AllocatedByRust> {
    let mut r = unsafe {
        // SAFETY: `alloc_node` palloc's a zero-initialized FdwRoutine node
        // tagged `T_FdwRoutine`; the resulting `PgBox` owns it for the
        // remainder of this call and is returned to the caller intact.
        PgBox::<pg_sys::FdwRoutine, AllocatedByRust>::alloc_node(pg_sys::NodeTag::T_FdwRoutine)
    };

    // Scan-side callbacks.
    r.GetForeignRelSize = Some(scan::get_foreign_rel_size);
    r.GetForeignPaths = Some(scan::get_foreign_paths);
    r.GetForeignPlan = Some(scan::get_foreign_plan);
    r.BeginForeignScan = Some(scan::begin_foreign_scan);
    r.IterateForeignScan = Some(scan::iterate_foreign_scan);
    r.ReScanForeignScan = Some(scan::re_scan_foreign_scan);
    r.EndForeignScan = Some(scan::end_foreign_scan);

    r.PlanForeignModify = Some(modify::plan_foreign_modify);
    r.BeginForeignModify = Some(modify::begin_foreign_modify);
    r.ExecForeignInsert = Some(modify::exec_foreign_insert);
    r.AddForeignUpdateTargets = Some(modify::update::add_foreign_update_targets);
    r.ExecForeignUpdate = Some(modify::update::exec_foreign_update);
    r.ExecForeignDelete = Some(modify::update::exec_foreign_delete);
    r.EndForeignModify = Some(modify::end_foreign_modify);

    r
}

/// Version-portable accessor for a `TupleDescData`'s i'th
/// `FormData_pg_attribute`.
///
/// On pg14..pg17 the descriptor carries a flexible-array `attrs` field of
/// `FormData_pg_attribute` we index directly. On pg18 the layout changed:
/// `attrs` was replaced with `compact_attrs` (a `CompactAttribute` FAM) and
/// PG added an FFI-exported `TupleDescAttr(tupdesc, i)` accessor that returns
/// the full `FormData_pg_attribute*` reconstructed from the cache slot. We
/// call that on pg18 to stay forward-compatible with the new on-disk shape.
///
/// # Safety
///
/// `td` must point to a live `TupleDescData` and `i` must be `< td->natts`.
/// The returned pointer is valid for the lifetime of the tuple descriptor.
#[allow(unsafe_code)]
pub(crate) unsafe fn tupdesc_attr(
    td: *mut pg_sys::TupleDescData,
    i: usize,
) -> *mut pg_sys::FormData_pg_attribute {
    #[cfg(any(feature = "pg14", feature = "pg15", feature = "pg16", feature = "pg17"))]
    {
        // SAFETY: caller asserts `td` is live and `i < natts`. `attrs` is a
        // flexible-array of `FormData_pg_attribute` immediately after the
        // header; `.as_ptr().add(i)` is the same arithmetic the C
        // `TupleDescAttr` macro performs.
        unsafe { (*td).attrs.as_ptr().add(i) as *mut pg_sys::FormData_pg_attribute }
    }
    #[cfg(feature = "pg18")]
    {
        // SAFETY: pgrx exposes the C `TupleDescAttr` accessor as an FFI
        // symbol on pg18. Caller asserts `td` valid and `i < natts`; PG
        // returns a pointer into a stable per-tupdesc cache slot that lives
        // as long as the descriptor itself.
        unsafe { pg_sys::TupleDescAttr(td, i as core::ffi::c_int) }
    }
}

/// 0-based column indices that are actually being SET in this UPDATE, for
/// the subplan that drives `rinfo`. Empty for DELETE.
///
/// PG's `updateColnosLists` is a `List` of `List<AttrNumber>`. Its location
/// differs across versions:
/// - pg14..pg17: on the `ModifyTable` *plan* node, as `updateColnosLists`,
///   reached via `mtstate->ps.plan`.
/// - pg18: lifted onto `ModifyTableState` as `mt_updateColnosLists`.
///
/// The shape (outer list indexed by subplan, inner list of `AttrNumber`) is
/// identical across all five versions. Subplan index is
/// `rinfo - mtstate->resultRelInfo`.
///
/// # Safety
///
/// `mtstate` and `rinfo` are valid executor pointers; `rinfo` lives within
/// the `mtstate->resultRelInfo` array.
#[allow(unsafe_code)]
pub(crate) unsafe fn update_cols_for_subplan(
    mtstate: *mut pg_sys::ModifyTableState,
    rinfo: *mut pg_sys::ResultRelInfo,
) -> Vec<usize> {
    // SAFETY: caller's contract — both pointers live in the executor's
    // ModifyTableState; PG guarantees `rinfo` is inside the contiguous
    // `resultRelInfo` array, so `offset_from` is well-defined.
    unsafe {
        let op = (*mtstate).operation;
        if op != pg_sys::CmdType::CMD_UPDATE {
            return Vec::new();
        }
        // Determine subplan index by pointer arithmetic against resultRelInfo.
        let base = (*mtstate).resultRelInfo;
        let sub = rinfo.offset_from(base);
        if sub < 0 {
            return Vec::new();
        }
        // Locate the outer List of per-subplan AttrNumber lists. pg14..pg17
        // keep it on the ModifyTable plan node; pg18 lifted it onto the
        // executor state.
        #[cfg(any(feature = "pg14", feature = "pg15", feature = "pg16", feature = "pg17"))]
        let lists: *mut pg_sys::List = {
            // SAFETY: a ModifyTableState's `ps.plan` always points at a
            // ModifyTable plan node when the state is a ModifyTableState.
            let plan = (*mtstate).ps.plan as *mut pg_sys::ModifyTable;
            if plan.is_null() {
                std::ptr::null_mut()
            } else {
                (*plan).updateColnosLists
            }
        };
        #[cfg(feature = "pg18")]
        let lists: *mut pg_sys::List = (*mtstate).mt_updateColnosLists;

        if lists.is_null() {
            return Vec::new();
        }
        let cell = pg_sys::list_nth(lists, sub as i32) as *mut pg_sys::List;
        if cell.is_null() {
            return Vec::new();
        }
        let pg_list: pgrx::PgList<std::ffi::c_void> = pgrx::PgList::from_pg(cell);
        let mut out = Vec::with_capacity(pg_list.len());
        // The inner list cells carry `int` (AttrNumber widened); pgrx exposes
        // `list_nth_int` for AttrNumber-typed integer lists.
        for i in 0..pg_list.len() {
            let attnum = pg_sys::list_nth_int(cell, i as i32);
            // attnum is 1-based; drop system columns (< 1) which cannot
            // appear in an UPDATE SET clause, and convert to 0-based.
            if attnum >= 1 {
                out.push((attnum - 1) as usize);
            }
        }
        out
    }
}
