#![forbid(unsafe_code)]
//! Backend-local handoff from the scan path to the modify path.
//!
//! ## Why
//!
//! For UPDATE/DELETE, the scan side stamps each row with a synthetic ctid
//! whose `blob_id` is a positional index into the scan's blob listing. The
//! modify side later resolves `ctid.blob_id → blob name + etag` to drive the
//! GET/PUT round-trip in `commit_plan`. If the modify side re-runs its own
//! listing, the indexes can drift (concurrent inserts that sort before
//! existing names shift everything) — a different blob could end up at the
//! same `blob_id`, silently corrupting data.
//!
//! ## How
//!
//! `begin_foreign_scan` calls [`publish`] with the (name, etag) pairs from its
//! list/HEAD response, keyed by the foreign-table OID. `begin_foreign_modify`
//! calls [`take`] for the same OID and reuses that exact list, capturing the
//! etag-at-SELECT-time into each `BlobIdEntry`. `commit_plan` then GETs and
//! PUTs with `If-Match: <that etag>`, so any concurrent writer between
//! SELECT and UPDATE is caught.
//!
//! Postgres backends are single-threaded, so a thread-local stack per OID is
//! sufficient. We stack rather than overwrite to tolerate the (rare) nested
//! case of a foreign-table referenced by multiple subplans in one statement.
//! `end_foreign_scan` cleans up any unconsumed entry to keep state tidy.

use pgrx::pg_sys;
use std::cell::RefCell;
use std::collections::HashMap;

/// Stacked list of `(blob_name, etag)` published by a scan and consumed by
/// the matching modify. LIFO so a nested-subplan scan-then-modify pair lines
/// up correctly.
type HandoffStack = Vec<Vec<(String, String)>>;

thread_local! {
    static HANDOFF: RefCell<HashMap<u32, HandoffStack>> = RefCell::new(HashMap::new());
}

/// Publish a scan-time blob list keyed by `relid`. LIFO: a matching `take`
/// will pop the most-recent entry first.
pub fn publish(relid: pg_sys::Oid, blobs: Vec<(String, String)>) {
    HANDOFF.with(|h| {
        h.borrow_mut()
            .entry(relid.to_u32())
            .or_default()
            .push(blobs);
    });
}

/// Consume the most-recent entry for `relid`. Returns `None` if no entry was
/// published (e.g. modify driven without a prior scan — unit tests, or a
/// future code path that bypasses the scan side).
pub fn take(relid: pg_sys::Oid) -> Option<Vec<(String, String)>> {
    HANDOFF.with(|h| {
        let mut map = h.borrow_mut();
        let entry = map.get_mut(&relid.to_u32())?;
        let v = entry.pop();
        if entry.is_empty() {
            map.remove(&relid.to_u32());
        }
        v
    })
}

/// How many entries are currently stacked for `relid`. Used by `build_plan`
/// to detect the self-referential UPDATE case (`UPDATE t ... WHERE col IN
/// (SELECT col FROM t WHERE ...)`) where two `ForeignScan` nodes over the
/// same relid coexist in one plan, both publish, and we can't safely guess
/// which one drives the modify subtree. See [`take_unique`].
pub fn stacked_count(relid: pg_sys::Oid) -> usize {
    HANDOFF.with(|h| {
        h.borrow()
            .get(&relid.to_u32())
            .map(|e| e.len())
            .unwrap_or(0)
    })
}

/// Like [`take`] but refuses the take when more than one entry is currently
/// stacked for `relid`. This is the consumer the UPDATE/DELETE modify path
/// uses: with two scans of the same foreign table in one statement (e.g.
/// `UPDATE t SET v=1 WHERE v IN (SELECT v FROM t WHERE k>0)`), a plain LIFO
/// `take` would return whichever scan was initialised last, which is NOT
/// guaranteed to be the modify-driving scan — the resulting `blob_table`
/// indexes wouldn't match the ctids the driving scan stamps, silently
/// corrupting writes (the driving scan's `blob_id` would resolve to a
/// different blob in the modify's `blob_table`, and that blob's etag is
/// also from the wrong snapshot so the lost-update guard does not catch it).
///
/// Returns:
///   * `Ok(Some(list))` — exactly one entry was stacked; consumed.
///   * `Ok(None)` — no entry stacked (preserved as a separate state so the
///     test-only fallback to a fresh listing in `build_plan` still works).
///   * `Err(SchemaMismatch)` — more than one entry; refuse to guess.
pub fn take_unique(
    relid: pg_sys::Oid,
) -> Result<Option<Vec<(String, String)>>, crate::error::FdwError> {
    HANDOFF.with(|h| {
        let mut map = h.borrow_mut();
        let entry = match map.get_mut(&relid.to_u32()) {
            Some(e) => e,
            None => return Ok(None),
        };
        if entry.len() > 1 {
            // Drain the stack so a subsequent retry in the same backend gets
            // a clean slate (each retry will re-publish from its own scan).
            let n = entry.len();
            entry.clear();
            map.remove(&relid.to_u32());
            return Err(crate::error::FdwError::SchemaMismatch(format!(
                "ambiguous scan_handoff for relid {}: {} concurrent scans of the same \
                 foreign table in one statement (e.g. a self-referential UPDATE). \
                 v1 does not yet match the modify subtree to its driving scan; \
                 rewrite the query so the source rows come from a non-foreign \
                 source (CTE, materialised subquery) or use two distinct foreign \
                 tables over the same data",
                relid.to_u32(),
                n
            )));
        }
        let v = entry.pop();
        if entry.is_empty() {
            map.remove(&relid.to_u32());
        }
        Ok(v)
    })
}

/// Drop any unconsumed entry for `relid`. Called from `end_foreign_scan` so
/// abandoned publishes don't leak across statements within the backend.
pub fn discard(relid: pg_sys::Oid) {
    HANDOFF.with(|h| {
        let mut map = h.borrow_mut();
        if let Some(entry) = map.get_mut(&relid.to_u32()) {
            entry.pop();
            if entry.is_empty() {
                map.remove(&relid.to_u32());
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oid(n: u32) -> pg_sys::Oid {
        pg_sys::Oid::from(n)
    }

    #[test]
    fn publish_then_take_returns_same() {
        let v = vec![("a.parquet".into(), "e1".into())];
        publish(oid(101), v.clone());
        assert_eq!(take(oid(101)), Some(v));
        assert_eq!(take(oid(101)), None);
    }

    #[test]
    fn take_is_lifo() {
        publish(oid(102), vec![("first".into(), "e1".into())]);
        publish(oid(102), vec![("second".into(), "e2".into())]);
        assert_eq!(take(oid(102)).unwrap()[0], ("second".into(), "e2".into()));
        assert_eq!(take(oid(102)).unwrap()[0], ("first".into(), "e1".into()));
    }

    #[test]
    fn discard_drops_top() {
        publish(oid(103), vec![("x".into(), "e".into())]);
        discard(oid(103));
        assert_eq!(take(oid(103)), None);
    }

    #[test]
    fn different_oids_isolated() {
        publish(oid(201), vec![("a".into(), "e1".into())]);
        publish(oid(202), vec![("b".into(), "e2".into())]);
        assert_eq!(take(oid(202)).unwrap()[0].0, "b");
        assert_eq!(take(oid(201)).unwrap()[0].0, "a");
    }

    #[test]
    fn take_unique_empty_returns_none() {
        assert!(matches!(take_unique(oid(300)), Ok(None)));
    }

    #[test]
    fn take_unique_single_entry_returns_it() {
        let v = vec![("only.parquet".into(), "e1".into())];
        publish(oid(301), v.clone());
        assert_eq!(take_unique(oid(301)).unwrap(), Some(v));
        // Drained.
        assert!(matches!(take_unique(oid(301)), Ok(None)));
    }

    // Regression: two scans of the same relid in one statement (self-referential
    // UPDATE) would silently corrupt writes because plain `take` returns whichever
    // scan was initialised last — not necessarily the modify-driving scan. The
    // safe v1 behaviour is to refuse the take and surface a clear error.
    #[test]
    fn take_unique_multiple_entries_errors_and_drains() {
        publish(oid(302), vec![("driver".into(), "e_driver".into())]);
        publish(oid(302), vec![("inner".into(), "e_inner".into())]);
        let err = take_unique(oid(302)).expect_err("multiple entries must error");
        match err {
            crate::error::FdwError::SchemaMismatch(msg) => {
                assert!(msg.contains("ambiguous scan_handoff"), "{msg}");
                assert!(msg.contains("self-referential"), "{msg}");
            }
            other => panic!("expected SchemaMismatch, got {other:?}"),
        }
        // Stack must be drained so a future retry starts clean.
        assert_eq!(stacked_count(oid(302)), 0);
    }

    #[test]
    fn stacked_count_reflects_pushes_and_pops() {
        assert_eq!(stacked_count(oid(400)), 0);
        publish(oid(400), vec![]);
        publish(oid(400), vec![]);
        assert_eq!(stacked_count(oid(400)), 2);
        let _ = take(oid(400));
        assert_eq!(stacked_count(oid(400)), 1);
        let _ = take(oid(400));
        assert_eq!(stacked_count(oid(400)), 0);
    }
}
