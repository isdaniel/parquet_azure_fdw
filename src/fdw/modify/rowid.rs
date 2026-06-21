#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]
//! Synthetic ctid <-> (blob_id, row_offset) encoding.
//!
//! See design spec §3.1. We pack PG's `ItemPointerData` such that:
//!   * BlockNumber (high 32 bits) = blob_id   (index into ModifyPlan.blob_table)
//!   * OffsetNumber (low 16 bits) = row_offset within this 65_536-row chunk
//!
//! Blobs larger than `CHUNK_ROWS` rows occupy multiple consecutive blob_ids;
//! the modify path's `BlobIdEntry` table records each chunk's `chunk_base_row`
//! so `RowId` can recover the absolute row within the source blob.
//!
//! FFI quirk: PG's `ItemPointerSet` / `ItemPointerGetBlockNumber` /
//! `ItemPointerGetOffsetNumber` are C macros (not real functions), and pgrx
//! 0.18.1's pg14 bindings do not expose them as FFI symbols. We therefore
//! transcribe the macros' field arithmetic directly against `ItemPointerData`
//! / `BlockIdData`, which are `#[repr(C, packed(2))]` and `#[repr(C)]` so the
//! layout matches Postgres byte-for-byte.

use pgrx::pg_sys;

/// Rows per chunk. Equals `OffsetNumber::MAX + 1`.
pub const CHUNK_ROWS: u64 = 65_536;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowId {
    pub blob_id: u32,
    pub offset: u16,
}

impl RowId {
    /// Build a `RowId` for the `abs_row`-th row of a source blob whose first
    /// chunk has `blob_base_id`. Rolls a new `blob_id` every `CHUNK_ROWS`.
    pub fn from_absolute(blob_base_id: u32, abs_row: u64) -> Self {
        let chunk = (abs_row / CHUNK_ROWS) as u32;
        let offset = (abs_row % CHUNK_ROWS) as u16;
        RowId {
            blob_id: blob_base_id.saturating_add(chunk),
            offset,
        }
    }

    pub fn to_ctid(self) -> pg_sys::ItemPointerData {
        // SAFETY: `ItemPointerData` is `#[repr(C, packed(2))]` with two
        // POD fields (`BlockIdData` + `OffsetNumber`); zeroing is a valid
        // bit-pattern for both.
        let mut t: pg_sys::ItemPointerData = unsafe { std::mem::zeroed() };
        // Replicates PG's `ItemPointerSet` macro (BlockIdSet + ip_posid =).
        t.ip_blkid.bi_hi = (self.blob_id >> 16) as u16;
        t.ip_blkid.bi_lo = (self.blob_id & 0xffff) as u16;
        t.ip_posid = self.offset;
        t
    }

    pub fn from_ctid(ctid: pg_sys::ItemPointerData) -> Self {
        // Replicates PG's `ItemPointerGetBlockNumber` /
        // `ItemPointerGetOffsetNumber` macros. Field access only; no FFI.
        let blob_id = ((ctid.ip_blkid.bi_hi as u32) << 16) | (ctid.ip_blkid.bi_lo as u32);
        let offset = ctid.ip_posid;
        RowId { blob_id, offset }
    }

    pub fn chunk_index_within_blob(self, blob_base_id: u32) -> u32 {
        self.blob_id - blob_base_id
    }

    pub fn absolute_row_within_blob(self, blob_base_id: u32) -> u64 {
        (self.chunk_index_within_blob(blob_base_id) as u64) * CHUNK_ROWS + self.offset as u64
    }
}
