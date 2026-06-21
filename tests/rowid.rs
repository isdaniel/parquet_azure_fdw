use parquet_azure_fdw::fdw::modify::rowid::{RowId, CHUNK_ROWS};
use pgrx::pg_sys;

fn ip(block: u32, off: u16) -> pg_sys::ItemPointerData {
    // PG's `ItemPointerSet` is a C macro not exposed by pgrx 0.18.1's pg14
    // bindings, so we transcribe it here field-by-field — same arithmetic
    // the production code in `rowid::to_ctid` uses, written independently to
    // catch transcription mistakes via the round-trip assertion below.
    // SAFETY: `ItemPointerData` is a POD `#[repr(C)]` struct of two
    // `u16`s and one `BlockIdData` (also two `u16`s); the all-zero bit
    // pattern is a valid initialized value.
    let mut t: pg_sys::ItemPointerData = unsafe { std::mem::zeroed() };
    t.ip_blkid.bi_hi = (block >> 16) as u16;
    t.ip_blkid.bi_lo = (block & 0xffff) as u16;
    t.ip_posid = off;
    t
}

#[test]
fn round_trip_boundaries() {
    for &(b, o) in &[(0u32, 0u16), (0, 65_535), (u32::MAX, 0), (u32::MAX, 65_535)] {
        let r = RowId {
            blob_id: b,
            offset: o,
        };
        let t = r.to_ctid();
        let back = RowId::from_ctid(t);
        assert_eq!(r.blob_id, back.blob_id, "blob_id mismatch for ({b},{o})");
        assert_eq!(r.offset, back.offset, "offset mismatch for ({b},{o})");
        // And we agree with PG's own constructor:
        let direct = ip(b, o);
        let from_direct = RowId::from_ctid(direct);
        assert_eq!(from_direct.blob_id, b);
        assert_eq!(from_direct.offset, o);
    }
}

#[test]
fn chunk_math_one_blob() {
    // A 70_000-row blob with base blob_id = 7 occupies blob_ids 7 and 8.
    // row 0     -> (7, 0)
    // row 65535 -> (7, 65535)
    // row 65536 -> (8, 0)
    // row 70000 -> (8, 4464)
    for &(abs_row, exp_id, exp_off) in &[
        (0u64, 7u32, 0u16),
        (CHUNK_ROWS - 1, 7, 65_535),
        (CHUNK_ROWS, 8, 0),
        (70_000, 8, (70_000 - CHUNK_ROWS) as u16),
    ] {
        let r = RowId::from_absolute(7, abs_row);
        assert_eq!(r.blob_id, exp_id);
        assert_eq!(r.offset, exp_off);
        assert_eq!(r.absolute_row_within_blob(7), abs_row);
    }
}

#[test]
fn scan_synthetic_ctid_matches_rowid_encoding() {
    // Lock in the contract surface that `scan::iterate_foreign_scan` stamps
    // into `slot->tts_tid`: for every (blob_base, abs_row) the scan node
    // emits, decoding the resulting ctid must yield the same `RowId` and
    // the original `abs_row` must be recoverable via
    // `absolute_row_within_blob`. Lives here (not `tests/runtime.rs`)
    // because it only exercises pure-Rust `RowId` helpers — no PG harness
    // needed.
    let cases = [
        (0u32, 0u64),
        (0, CHUNK_ROWS - 1),
        (0, CHUNK_ROWS),
        (5, 70_000),
    ];
    for &(base, abs) in &cases {
        let r = RowId::from_absolute(base, abs);
        let t = r.to_ctid();
        let back = RowId::from_ctid(t);
        assert_eq!(r, back);
        assert_eq!(back.absolute_row_within_blob(base), abs);
    }
}
