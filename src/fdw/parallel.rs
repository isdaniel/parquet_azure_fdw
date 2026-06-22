#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]
//! Parallel-scan FDW support: DSM-backed atomic cursor over the blob list,
//! a `RangeProducer` impl (`ParallelRanges`) that reads from it, and the
//! five PG parallel-FDW callbacks (`IsForeignScanParallelSafe`,
//! `EstimateDSMForeignScan`, `InitializeDSMForeignScan`,
//! `ReInitializeDSMForeignScan`, `InitializeWorkerForeignScan`).
//!
//! See `docs/superpowers/specs/2026-06-22-sp3a-parallel-scan-design.md`.

use crate::error::{FdwError, FdwResult};
use crate::fdw::scan::ScanState;
use pgrx::pg_sys;
use std::ffi::{c_void, CStr};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

/// DSM segment header. Followed inline by `n_blobs` entries, each:
///   [u32 name_len_le][u32 etag_len_le][name bytes][etag bytes]
#[repr(C)]
pub(crate) struct ParallelDsmStateHeader {
    pub(crate) cursor: AtomicUsize,
    pub(crate) n_blobs: usize,
    pub(crate) total_payload_bytes: usize,
}

/// Compute total DSM bytes needed for the given blob list.
pub(crate) fn dsm_size_for(blobs: &[(String, String)]) -> usize {
    let payload: usize = blobs.iter().map(|(n, e)| 8 + n.len() + e.len()).sum();
    std::mem::size_of::<ParallelDsmStateHeader>() + payload
}

/// Write the header + packed entries into `dst`. Caller guarantees
/// `dst_len == dsm_size_for(blobs)` and `dst` is properly aligned for the
/// header.
///
/// # Safety
/// `dst` must be a writable region of at least `dst_len` bytes, aligned to
/// `align_of::<ParallelDsmStateHeader>()`. `dst_len` MUST equal
/// `dsm_size_for(blobs)`; otherwise this writes past the buffer.
pub(crate) unsafe fn dsm_serialize_blobs(dst: *mut u8, dst_len: usize, blobs: &[(String, String)]) {
    debug_assert_eq!(dst_len, dsm_size_for(blobs));
    let payload_bytes = dst_len - std::mem::size_of::<ParallelDsmStateHeader>();
    // SAFETY: caller guarantees alignment + size of `dst`.
    let header_ptr = dst as *mut ParallelDsmStateHeader;
    unsafe {
        std::ptr::write(
            header_ptr,
            ParallelDsmStateHeader {
                cursor: AtomicUsize::new(0),
                n_blobs: blobs.len(),
                total_payload_bytes: payload_bytes,
            },
        );
    }
    // SAFETY: byte-offset past the header into the payload region.
    let mut cur = unsafe { dst.add(std::mem::size_of::<ParallelDsmStateHeader>()) };
    for (name, etag) in blobs {
        let nl = name.len() as u32;
        let el = etag.len() as u32;
        // SAFETY: each iteration writes (8 + nl + el) bytes; total over the
        // loop equals `payload_bytes` (verified by dsm_size_for math).
        unsafe {
            std::ptr::copy_nonoverlapping(nl.to_le_bytes().as_ptr(), cur, 4);
            cur = cur.add(4);
            std::ptr::copy_nonoverlapping(el.to_le_bytes().as_ptr(), cur, 4);
            cur = cur.add(4);
            std::ptr::copy_nonoverlapping(name.as_ptr(), cur, name.len());
            cur = cur.add(name.len());
            std::ptr::copy_nonoverlapping(etag.as_ptr(), cur, etag.len());
            cur = cur.add(etag.len());
        }
    }
}

/// Deserialize the packed entries from a previously-serialized region.
/// Returns the header pointer (so workers can do `cursor.fetch_add`) and
/// the owned `Vec<(String, String)>` of blob entries.
pub(crate) type DsmDeserialized = (NonNull<ParallelDsmStateHeader>, Vec<(String, String)>);

/// Deserialize the packed entries from a previously-serialized region.
/// Returns the header pointer (so workers can do `cursor.fetch_add`) and
/// the owned `Vec<(String, String)>` of blob entries.
///
/// # Safety
/// `src` must point at a region previously written by `dsm_serialize_blobs`
/// with the same blob list. Header alignment requirements as for
/// `dsm_serialize_blobs`.
pub(crate) unsafe fn dsm_deserialize_blobs(src: *const u8) -> FdwResult<DsmDeserialized> {
    // SAFETY: caller's contract — region was written by our serializer.
    let header_ptr = src as *const ParallelDsmStateHeader;
    let n_blobs = unsafe { (*header_ptr).n_blobs };
    let mut out = Vec::with_capacity(n_blobs);
    // SAFETY: same as above; payload starts immediately after header.
    let mut cur = unsafe { src.add(std::mem::size_of::<ParallelDsmStateHeader>()) };
    for _ in 0..n_blobs {
        let mut nl_buf = [0u8; 4];
        let mut el_buf = [0u8; 4];
        // SAFETY: bytes were written by serializer in matching layout.
        unsafe {
            std::ptr::copy_nonoverlapping(cur, nl_buf.as_mut_ptr(), 4);
            cur = cur.add(4);
            std::ptr::copy_nonoverlapping(cur, el_buf.as_mut_ptr(), 4);
            cur = cur.add(4);
        }
        let nl = u32::from_le_bytes(nl_buf) as usize;
        let el = u32::from_le_bytes(el_buf) as usize;
        let mut name_bytes = vec![0u8; nl];
        let mut etag_bytes = vec![0u8; el];
        // SAFETY: bytes still in our payload region.
        unsafe {
            std::ptr::copy_nonoverlapping(cur, name_bytes.as_mut_ptr(), nl);
            cur = cur.add(nl);
            std::ptr::copy_nonoverlapping(cur, etag_bytes.as_mut_ptr(), el);
            cur = cur.add(el);
        }
        // Names and etags from Azure are guaranteed valid UTF-8; defensively
        // route a corrupted DSM payload through FdwError rather than panicking
        // across the `extern "C-unwind"` FFI boundary.
        let name = String::from_utf8(name_bytes).map_err(|e| {
            FdwError::SchemaMismatch(format!("DSM payload contained non-UTF-8 blob name: {e}"))
        })?;
        let etag = String::from_utf8(etag_bytes).map_err(|e| {
            FdwError::SchemaMismatch(format!("DSM payload contained non-UTF-8 etag: {e}"))
        })?;
        out.push((name, etag));
    }
    let nn = NonNull::new(header_ptr as *mut ParallelDsmStateHeader)
        .ok_or_else(|| FdwError::SchemaMismatch("DSM header pointer was null".into()))?;
    Ok((nn, out))
}

/// `RangeProducer` impl backed by a shared DSM atomic cursor. Each worker
/// holds its own deserialized blob list (cheap; per-worker copy) and shares
/// only the cursor with sibling workers.
///
/// SAFETY: the raw `NonNull<ParallelDsmStateHeader>` wraps a pointer into
/// PG's DSM segment, which is alive for the duration of the scan and is
/// safely sharable across worker threads (the `AtomicUsize` inside the
/// header provides the only inter-thread synchronization point).
pub(crate) struct ParallelRanges {
    pub(crate) dsm: NonNull<ParallelDsmStateHeader>,
    pub(crate) blobs: Vec<(String, String)>,
}

// SAFETY: `dsm` is a `NonNull<ParallelDsmStateHeader>` whose target is the
// PG DSM segment — alive for the duration of the scan, with the only shared
// mutable state being its `AtomicUsize` cursor (atomic by construction).
// `blobs` is `Vec<(String, String)>` which is already `Send`.
unsafe impl Send for ParallelRanges {}

impl crate::fdw::scan::RangeProducer for ParallelRanges {
    fn next_blob(&mut self) -> Option<(String, String)> {
        // SAFETY: see struct doc — the dsm pointer is valid for the scan.
        let idx = unsafe { (*self.dsm.as_ptr()).cursor.fetch_add(1, Ordering::Relaxed) };
        self.blobs.get(idx).cloned()
    }
}

// ---------- FFI callbacks --------------------------------------------------

/// `IsForeignScanParallelSafe_function` — gate parallel scan to SELECT and
/// to tables whose `parallel_workers` option is not `Some(0)`.
///
/// # Safety
/// PG-supplied pointers are live for the callback.
pub unsafe extern "C-unwind" fn is_foreign_scan_parallel_safe(
    root: *mut pg_sys::PlannerInfo,
    _rel: *mut pg_sys::RelOptInfo,
    rte: *mut pg_sys::RangeTblEntry,
) -> bool {
    // SAFETY: PG-supplied pointers.
    let cmd = unsafe {
        if root.is_null() || (*root).parse.is_null() {
            return false;
        }
        (*(*root).parse).commandType
    };
    if cmd != pg_sys::CmdType::CMD_SELECT {
        return false;
    }
    if rte.is_null() {
        return false;
    }
    // SAFETY: rte->relid is the foreign-table OID; pg_sys catalog accessors
    // are safe to call with a valid OID.
    let relid = unsafe { (*rte).relid };
    if matches!(unsafe { read_parallel_workers_opt(relid) }, Some(0)) {
        return false;
    }
    // SP-3c: sorted mode runs a single-coordinator K-way heap merge that
    // cannot be split across workers. Force sequential execution.
    if unsafe { read_sorted_opt(relid) } {
        return false;
    }
    true
}

/// Read the `sorted` table option directly off the catalog. Returns `true`
/// iff the option is present and non-empty (mirrors `read_parallel_workers_opt`;
/// the planner callback runs before `BeginForeignScan`, so we can't borrow the
/// parsed value off `ScanState`).
pub(crate) unsafe fn read_sorted_opt(relid: pg_sys::Oid) -> bool {
    // SAFETY: documented PG catalog accessor.
    let table = unsafe { pg_sys::GetForeignTable(relid) };
    if table.is_null() {
        return false;
    }
    // SAFETY: `(*table).options` is either null or a valid `*List of DefElem*`.
    let opts: *mut pg_sys::List = unsafe { (*table).options };
    if opts.is_null() {
        return false;
    }
    // SAFETY: `from_pg` documented requirement is a valid `*mut List`.
    let pg_list: pgrx::PgList<pg_sys::DefElem> = unsafe { pgrx::PgList::from_pg(opts) };
    for def in pg_list.iter_ptr() {
        if def.is_null() {
            continue;
        }
        // SAFETY: DefElem.defname is a palloc'd NUL-terminated C string.
        let name = unsafe { CStr::from_ptr((*def).defname).to_string_lossy() };
        if name == "sorted" {
            // SAFETY: defGetString returns a palloc'd C string or null.
            let v_ptr = unsafe { pg_sys::defGetString(def) };
            if v_ptr.is_null() {
                return false;
            }
            // SAFETY: documented to be NUL-terminated.
            let v = unsafe { CStr::from_ptr(v_ptr).to_string_lossy() };
            return !v.trim().is_empty();
        }
    }
    false
}

/// Read the `parallel_workers` table option directly off the catalog
/// (the planner callback runs before `BeginForeignScan`, so we can't
/// borrow the parsed value off `ScanState`).
pub(crate) unsafe fn read_parallel_workers_opt(relid: pg_sys::Oid) -> Option<i32> {
    // SAFETY: documented PG catalog accessor.
    let table = unsafe { pg_sys::GetForeignTable(relid) };
    if table.is_null() {
        return None;
    }
    // SAFETY: `(*table).options` is either null or a valid `*List of DefElem*`.
    let opts: *mut pg_sys::List = unsafe { (*table).options };
    if opts.is_null() {
        return None;
    }
    // SAFETY: `from_pg` documented requirement is a valid `*mut List`.
    let pg_list: pgrx::PgList<pg_sys::DefElem> = unsafe { pgrx::PgList::from_pg(opts) };
    for def in pg_list.iter_ptr() {
        if def.is_null() {
            continue;
        }
        // SAFETY: DefElem.defname is a palloc'd NUL-terminated C string.
        let name = unsafe { CStr::from_ptr((*def).defname).to_string_lossy() };
        if name == "parallel_workers" {
            // SAFETY: defGetString returns a palloc'd C string or null.
            let v_ptr = unsafe { pg_sys::defGetString(def) };
            if v_ptr.is_null() {
                return None;
            }
            // SAFETY: documented to be NUL-terminated.
            let v = unsafe { CStr::from_ptr(v_ptr).to_string_lossy() };
            return v.parse::<i32>().ok();
        }
    }
    None
}

/// `EstimateDSMForeignScan_function` — return DSM bytes needed.
///
/// # Safety
/// `node` is a live `ForeignScanState` whose `fdw_state` was populated by
/// `begin_foreign_scan`.
pub unsafe extern "C-unwind" fn estimate_dsm_foreign_scan(
    node: *mut pg_sys::ForeignScanState,
    _pcxt: *mut pg_sys::ParallelContext,
) -> pg_sys::Size {
    // SAFETY: see fn doc.
    let state: &ScanState = unsafe { &*((*node).fdw_state as *const ScanState) };
    dsm_size_for(state.blobs_for_dsm()) as pg_sys::Size
}

/// `InitializeDSMForeignScan_function` — leader fills the DSM segment.
///
/// # Safety
/// `coordinate` points to `estimate_dsm_foreign_scan`'s returned byte count
/// of writable memory.
pub unsafe extern "C-unwind" fn initialize_dsm_foreign_scan(
    node: *mut pg_sys::ForeignScanState,
    _pcxt: *mut pg_sys::ParallelContext,
    coordinate: *mut c_void,
) {
    // SAFETY: see fn doc.
    let state: &ScanState = unsafe { &*((*node).fdw_state as *const ScanState) };
    let blobs = state.blobs_for_dsm();
    let size = dsm_size_for(blobs);
    // SAFETY: caller-provided buffer of exactly `size` bytes.
    unsafe {
        dsm_serialize_blobs(coordinate as *mut u8, size, blobs);
    }
}

/// `ReInitializeDSMForeignScan_function` — reset the cursor for a rescan.
///
/// # Safety
/// `coordinate` points at the SAME region a prior `initialize_dsm_foreign_scan`
/// populated.
pub unsafe extern "C-unwind" fn re_initialize_dsm_foreign_scan(
    _node: *mut pg_sys::ForeignScanState,
    _pcxt: *mut pg_sys::ParallelContext,
    coordinate: *mut c_void,
) {
    // SAFETY: see fn doc — region was written by initialize_dsm_foreign_scan.
    let header = coordinate as *mut ParallelDsmStateHeader;
    unsafe {
        // Release ordering ensures the cursor reset is visible to workers
        // before they `fetch_add`. PG synchronizes worker (re-)launch via
        // shm barriers in practice, but tightening this is hygienic and
        // costs nothing on x86_64.
        (*header).cursor.store(0, Ordering::Release);
    }
}

/// `InitializeWorkerForeignScan_function` — worker attaches to DSM and
/// constructs its own `ScanState` with a [`ParallelRanges`] `RangeProducer`.
///
/// # Safety
/// `coordinate` points at the leader-populated DSM region.
pub unsafe extern "C-unwind" fn initialize_worker_foreign_scan(
    node: *mut pg_sys::ForeignScanState,
    _toc: *mut pg_sys::shm_toc,
    coordinate: *mut c_void,
) {
    // SAFETY: see fn doc.
    let (hdr, blobs) = match unsafe { dsm_deserialize_blobs(coordinate as *const u8) } {
        Ok(v) => v,
        Err(e) => crate::error::raise(e),
    };
    let producer: Box<dyn crate::fdw::scan::RangeProducer> =
        Box::new(ParallelRanges { dsm: hdr, blobs });
    // SAFETY: `node->ss.ss_currentRelation` is alive for the worker-init
    // callback per PG executor contract.
    let rel = unsafe { (*node).ss.ss_currentRelation };
    if rel.is_null() {
        crate::error::raise(crate::error::FdwError::SchemaMismatch(
            "parallel worker init: ss_currentRelation is null".into(),
        ));
    }
    // SAFETY: relation pointer is live; `rd_id` is its OID.
    let relid = unsafe { (*rel).rd_id };
    // SAFETY: build the worker's ScanState with the injected ParallelRanges.
    let state =
        match unsafe { crate::fdw::scan::build_state_for_parallel_worker(node, relid, producer) } {
            Ok(s) => s,
            Err(e) => crate::error::raise(e),
        };
    // SAFETY: store the boxed state into the executor's per-scan slot
    // (same contract as begin_foreign_scan).
    unsafe {
        (*node).fdw_state = Box::into_raw(Box::new(state)) as *mut c_void;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_blobs() -> Vec<(String, String)> {
        vec![
            ("data/a.parquet".to_string(), "etag-a".to_string()),
            ("data/b.parquet".to_string(), "etag-b".to_string()),
            ("data/c.parquet".to_string(), "etag-c".to_string()),
            ("data/d.parquet".to_string(), "etag-d".to_string()),
            ("data/e.parquet".to_string(), "etag-e".to_string()),
        ]
    }

    #[test]
    fn dsm_size_matches_payload() {
        let blobs = sample_blobs();
        let header = std::mem::size_of::<ParallelDsmStateHeader>();
        let payload: usize = blobs.iter().map(|(n, e)| 8 + n.len() + e.len()).sum();
        assert_eq!(dsm_size_for(&blobs), header + payload);
    }

    #[test]
    fn round_trip_preserves_blobs() {
        let blobs = sample_blobs();
        let size = dsm_size_for(&blobs);
        // Use a Box<[u8]> so the buffer is well-aligned and stays alive.
        // align_of::<ParallelDsmStateHeader>() is the strictest alignment in
        // play; on x86_64 a Vec<u8> happens to be 8-aligned, but assert it
        // to be safe.
        let mut buf: Vec<u8> = vec![0u8; size];
        assert!(
            (buf.as_ptr() as usize).is_multiple_of(std::mem::align_of::<ParallelDsmStateHeader>())
        );
        unsafe {
            dsm_serialize_blobs(buf.as_mut_ptr(), size, &blobs);
        }
        let (_hdr, got) = unsafe { dsm_deserialize_blobs(buf.as_ptr()) }.unwrap();
        assert_eq!(got, blobs);
    }

    #[test]
    fn parallel_ranges_yields_each_blob_once() {
        use crate::fdw::scan::RangeProducer;
        let blobs = sample_blobs(); // 5 blobs
        let size = dsm_size_for(&blobs);
        let mut buf: Vec<u8> = vec![0u8; size];
        unsafe {
            dsm_serialize_blobs(buf.as_mut_ptr(), size, &blobs);
        }
        // Two workers attached to the same DSM.
        let (hdr, blobs_local_a) = unsafe { dsm_deserialize_blobs(buf.as_ptr()) }.unwrap();
        let (_, blobs_local_b) = unsafe { dsm_deserialize_blobs(buf.as_ptr()) }.unwrap();
        let mut w_a = ParallelRanges {
            dsm: hdr,
            blobs: blobs_local_a,
        };
        let mut w_b = ParallelRanges {
            dsm: hdr,
            blobs: blobs_local_b,
        };
        // Drive both alternately to exhaustion.
        let mut got: Vec<String> = Vec::new();
        loop {
            match (w_a.next_blob(), w_b.next_blob()) {
                (None, None) => break,
                (a, b) => {
                    if let Some((n, _)) = a {
                        got.push(n);
                    }
                    if let Some((n, _)) = b {
                        got.push(n);
                    }
                }
            }
        }
        got.sort();
        let mut expected: Vec<String> = blobs.iter().map(|(n, _)| n.clone()).collect();
        expected.sort();
        assert_eq!(
            got, expected,
            "every blob must be yielded exactly once across workers"
        );
    }

    #[test]
    fn parallel_ranges_concurrent_workers_are_exclusive() {
        use crate::fdw::scan::RangeProducer;
        use std::sync::{Arc, Mutex};
        let blobs: Vec<(String, String)> = (0..100)
            .map(|i| (format!("b/{i}.parquet"), format!("e{i}")))
            .collect();
        let size = dsm_size_for(&blobs);
        let mut buf: Vec<u8> = vec![0u8; size];
        unsafe {
            dsm_serialize_blobs(buf.as_mut_ptr(), size, &blobs);
        }
        // Wrap the buffer in Arc<Mutex<>> only to keep it alive in the
        // spawned threads — we never lock it; pointers into it are stable
        // for the buffer's lifetime.
        let buf_arc = Arc::new(buf);
        let collected: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let buf_clone = Arc::clone(&buf_arc);
            let col = Arc::clone(&collected);
            handles.push(std::thread::spawn(move || {
                let (hdr, blobs_local) =
                    unsafe { dsm_deserialize_blobs(buf_clone.as_ptr()) }.unwrap();
                let mut w = ParallelRanges {
                    dsm: hdr,
                    blobs: blobs_local,
                };
                while let Some((n, _)) = w.next_blob() {
                    col.lock().unwrap().push(n);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let mut got = collected.lock().unwrap().clone();
        got.sort();
        let mut expected: Vec<String> = blobs.iter().map(|(n, _)| n.clone()).collect();
        expected.sort();
        assert_eq!(got.len(), 100, "exactly 100 blobs, no duplicates");
        assert_eq!(got, expected);
    }

    #[test]
    fn parallel_ranges_single_blob_five_workers() {
        use crate::fdw::scan::RangeProducer;
        let blobs = vec![("only.parquet".to_string(), "e".to_string())];
        let size = dsm_size_for(&blobs);
        let mut buf: Vec<u8> = vec![0u8; size];
        unsafe {
            dsm_serialize_blobs(buf.as_mut_ptr(), size, &blobs);
        }
        let mut hit_count = 0;
        for _ in 0..5 {
            let (hdr, b) = unsafe { dsm_deserialize_blobs(buf.as_ptr()) }.unwrap();
            let mut w = ParallelRanges { dsm: hdr, blobs: b };
            for _ in 0..10 {
                if w.next_blob().is_some() {
                    hit_count += 1;
                }
            }
        }
        assert_eq!(
            hit_count, 1,
            "exactly one worker should claim the single blob"
        );
    }

    #[test]
    fn empty_blob_list_round_trip() {
        let blobs: Vec<(String, String)> = vec![];
        let size = dsm_size_for(&blobs);
        let mut buf: Vec<u8> = vec![0u8; size];
        unsafe {
            dsm_serialize_blobs(buf.as_mut_ptr(), size, &blobs);
        }
        let (hdr, got) = unsafe { dsm_deserialize_blobs(buf.as_ptr()) }.unwrap();
        assert!(got.is_empty());
        unsafe {
            assert_eq!((*hdr.as_ptr()).n_blobs, 0);
        }
    }
}
