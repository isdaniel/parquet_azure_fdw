#![deny(unsafe_op_in_unsafe_fn)]
//! FDW write-path callbacks (INSERT + UPDATE/DELETE).
//!
//! The submodule split is purely organizational — the FFI surface is
//! re-exported here so external callers (fdw::mod, lib.rs) keep working.
//!
//! Two state shapes share `ResultRelInfo.ri_FdwState`:
//!
//! - [`insert::InsertState`] for INSERT — Arrow builders + ParquetBatchWriter,
//!   one fresh blob per statement.
//! - [`update::ModifyPlan`] for UPDATE/DELETE — per-blob `BlobEdits`
//!   accumulator, drained in `end_foreign_modify` via
//!   [`update::commit_plan`].
//!
//! [`begin_foreign_modify`](insert::begin_foreign_modify) inspects
//! `mtstate->operation` to decide which variant to construct.

pub mod coordinator;
pub(crate) mod insert;
pub mod kernel;
pub mod rowid;
pub mod scan_handoff;
pub mod update;

pub use insert::{
    begin_foreign_modify, end_foreign_modify, exec_foreign_insert, plan_foreign_modify,
};

use pgrx::pg_sys;

/// One entry per (blob, 65_536-row chunk) pair surfaced to the scan/modify
/// path. The chunk machinery (`chunk_base_row`) lets `RowId::from_ctid` map
/// back to the source blob's absolute row.
///
/// `etag` is captured at scan-time (LIST or HEAD response) and propagated to
/// `commit_plan`, where the GET and PUT use it as an `If-Match` precondition.
/// This is the v1 lost-update guard: any concurrent writer that mutates the
/// blob between SELECT and UPDATE/DELETE will trip the GET's precondition
/// and we surface SQLSTATE 40001.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlobIdEntry {
    pub name: String,
    pub chunk_base_row: u64,
    pub etag: String,
}

/// Enum stash for `ResultRelInfo.ri_FdwState`. The variant is chosen by
/// `begin_foreign_modify` based on `mtstate->operation`. Both variants are
/// boxed to keep the discriminated-union size compact and to avoid the
/// `clippy::large_enum_variant` warning — `InsertState` carries Arrow
/// builders + a parquet writer, `ModifyPlan` carries a hashmap of edits;
/// boxing also keeps the enum cheap to move between match arms.
pub enum FdwModifyState {
    Insert(Box<insert::InsertState>),
    Update(Box<update::ModifyPlan>),
}

/// Take ownership of the boxed state stashed in `ri_FdwState`, leaving the
/// slot nulled so any accidental re-entry crashes loudly.
///
/// # Safety
///
/// `rinfo` must be a live `ResultRelInfo` whose `ri_FdwState` was either
/// null (EXPLAIN-only short-circuit) or populated via
/// `Box::into_raw(Box::new(FdwModifyState::...))`. Caller arranges for
/// exactly one matching reclamation per `Box::into_raw`.
pub(crate) unsafe fn take_state(rinfo: *mut pg_sys::ResultRelInfo) -> Option<Box<FdwModifyState>> {
    // SAFETY: see fn-level safety.
    unsafe {
        let p = (*rinfo).ri_FdwState as *mut FdwModifyState;
        if p.is_null() {
            return None;
        }
        (*rinfo).ri_FdwState = std::ptr::null_mut();
        Some(Box::from_raw(p))
    }
}
