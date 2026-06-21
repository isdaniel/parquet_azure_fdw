#![deny(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]
//! Statement-scoped coordinator that tracks in-flight staging blobs so a
//! PG xact-abort can sweep them. Pure data + helpers — the FFI hook (PG
//! `XactCallback`) lives in mod.rs which is the FDW unsafe carve-out.

use std::collections::HashSet;
use uuid::Uuid;

/// Tracks the set of staging blobs the current statement has created but
/// has not yet committed (renamed-into-place via put_if_match). On xact
/// abort the unfinished set is the cleanup target.
#[derive(Default)]
pub struct StatementCoordinator {
    /// Staging names registered for create; removed when the statement
    /// successfully swaps + deletes them.
    in_flight: HashSet<String>,
}

impl StatementCoordinator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_staging(&mut self, name: String) {
        self.in_flight.insert(name);
    }

    pub fn mark_committed(&mut self, name: &str) {
        self.in_flight.remove(name);
    }

    pub fn pending_staging(&self) -> impl Iterator<Item = &str> {
        self.in_flight.iter().map(String::as_str)
    }

    pub fn is_empty(&self) -> bool {
        self.in_flight.is_empty()
    }
}

/// Build a staging name from the original blob name: strip the `.parquet`
/// suffix if present, append `.tmp.<uuid_v4>.parquet`.
///
/// uuid v4 is 128 bits of entropy — collision-safe for any realistic
/// statement size; a server-side If-None-Match still catches the
/// astronomical case.
pub fn make_staging_name(original: &str) -> String {
    let stem = original.strip_suffix(".parquet").unwrap_or(original);
    format!("{stem}.tmp.{}.parquet", Uuid::new_v4().simple())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_mark_committed_round_trip() {
        let mut c = StatementCoordinator::new();
        c.register_staging("a.tmp.1.parquet".into());
        c.register_staging("b.tmp.2.parquet".into());
        assert_eq!(c.pending_staging().count(), 2);
        c.mark_committed("a.tmp.1.parquet");
        let pending: Vec<&str> = c.pending_staging().collect();
        assert_eq!(pending, vec!["b.tmp.2.parquet"]);
        c.mark_committed("b.tmp.2.parquet");
        assert!(c.is_empty());
    }

    #[test]
    fn staging_name_contains_tmp_infix_and_preserves_ext() {
        let s = make_staging_name("data/year=2025/file.parquet");
        assert!(s.starts_with("data/year=2025/file.tmp."), "got {s}");
        assert!(s.ends_with(".parquet"), "got {s}");
    }

    #[test]
    fn staging_name_without_parquet_ext_still_produces_tmp() {
        let s = make_staging_name("rawfile");
        assert!(s.starts_with("rawfile.tmp."), "got {s}");
        assert!(s.ends_with(".parquet"));
    }
}

// ---------------------------------------------------------------------------
// Per-backend active coordinator + PG XactCallback registration.
//
// PG backends are single-threaded, so a thread_local! cell suffices for the
// "currently-open modify statement". The xact callback (in the `xact`
// submodule — the only carve-out from this file's forbid(unsafe_code)) takes
// the slot at COMMIT/ABORT and, on abort, best-effort-deletes any staging
// blobs still in flight so a failed statement doesn't leak tmp blobs.
// ---------------------------------------------------------------------------

use crate::azure::AzureBlobClient;
use std::cell::RefCell;

thread_local! {
    static ACTIVE: RefCell<Option<Active>> = const { RefCell::new(None) };
}

struct Active {
    client: AzureBlobClient,
    coord: StatementCoordinator,
}

/// Install the coordinator at the start of a foreign-modify statement.
/// Replaces any prior slot (defensively logs — shouldn't happen in practice).
pub fn open_statement(client: AzureBlobClient) {
    ACTIVE.with(|slot| {
        let mut s = slot.borrow_mut();
        if s.is_some() {
            eprintln!("parquet_azure_fdw: replaced existing StatementCoordinator (likely a bug)");
        }
        *s = Some(Active {
            client,
            coord: StatementCoordinator::new(),
        });
    });
}

/// Drop the coordinator at the end of a successful statement
/// (`EndForeignModify`). Any remaining in-flight staging blobs are orphans —
/// log a warning but don't fail the statement.
pub fn close_statement_success() {
    ACTIVE.with(|slot| {
        if let Some(active) = slot.borrow_mut().take() {
            if !active.coord.is_empty() {
                let pending: Vec<String> =
                    active.coord.pending_staging().map(String::from).collect();
                pgrx::warning!(
                    "parquet_azure_fdw: {} staging blob(s) left at statement end \
                     (orphans, not cleaned): {:?}",
                    pending.len(),
                    pending
                );
            }
        }
    });
}

/// Borrow the active coordinator + client mutably. Returns `None` if no
/// statement is currently open.
pub fn with_active<R>(
    f: impl FnOnce(&mut StatementCoordinator, &AzureBlobClient) -> R,
) -> Option<R> {
    ACTIVE.with(|slot| {
        slot.borrow_mut()
            .as_mut()
            .map(|a| f(&mut a.coord, &a.client))
    })
}

/// Register the PG xact callback exactly once per backend. Safe to call
/// repeatedly (a `Once` gates the actual registration). Wired from
/// `_PG_init`.
pub fn install_xact_callback_once() {
    xact::install_once();
}

// ---- PG XactCallback registration ----------------------------------------
//
// The callback is the only FFI surface this file owns. We carve it out of
// the file-level forbid(unsafe_code) with an inner `#[allow(unsafe_code)]`
// submodule (mirroring fdw/modify/mod.rs's deny(unsafe_op_in_unsafe_fn)
// posture). Every unsafe op below has an explicit SAFETY comment.

#[allow(unsafe_code)]
mod xact {
    #![deny(unsafe_op_in_unsafe_fn)]

    use super::ACTIVE;
    use pgrx::pg_sys;
    use std::ffi::c_void;
    use std::ptr;
    use std::sync::Once;

    static INSTALLED: Once = Once::new();

    pub fn install_once() {
        INSTALLED.call_once(|| {
            // SAFETY: pg_sys::RegisterXactCallback is a stable PG C API
            // since PG10. We pass a 'static extern "C-unwind" fn and a null
            // arg. Called from _PG_init under PG's startup serialization;
            // the Once additionally guarantees one registration per backend.
            unsafe {
                pg_sys::RegisterXactCallback(Some(xact_cb), ptr::null_mut());
            }
        });
    }

    /// PG xact callback. Acts only on COMMIT/ABORT (top-level + parallel
    /// variants). MUST NOT raise — that would crash the backend. All cleanup
    /// is best-effort and any error is logged via `pgrx::warning!`.
    unsafe extern "C-unwind" fn xact_cb(event: pg_sys::XactEvent::Type, _arg: *mut c_void) {
        let is_abort = event == pg_sys::XactEvent::XACT_EVENT_ABORT
            || event == pg_sys::XactEvent::XACT_EVENT_PARALLEL_ABORT;
        let is_commit = event == pg_sys::XactEvent::XACT_EVENT_COMMIT
            || event == pg_sys::XactEvent::XACT_EVENT_PARALLEL_COMMIT;
        if !(is_abort || is_commit) {
            return;
        }
        // Wrap the body in catch_unwind so a stray panic from within the
        // cleanup path can't unwind into PG's xact machinery. Logging the
        // panic message itself uses eprintln (pgrx logging may already be
        // torn down in this phase).
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ACTIVE.with(|slot| {
                let active = match slot.borrow_mut().take() {
                    Some(a) => a,
                    None => return,
                };
                if !is_abort || active.coord.is_empty() {
                    return;
                }
                let pending: Vec<String> =
                    active.coord.pending_staging().map(String::from).collect();
                let client = active.client.clone();
                for name in &pending {
                    // runtime::block_on lazily inits a per-backend
                    // current-thread tokio runtime — fine to call from this
                    // xact callback (we're on the backend thread).
                    let res = crate::runtime::block_on(client.delete_unconditional(name));
                    if let Err(e) = res {
                        pgrx::warning!(
                            "parquet_azure_fdw xact-abort: failed to delete staging '{}': {}",
                            name,
                            e
                        );
                    }
                }
            });
        }));
    }
}
