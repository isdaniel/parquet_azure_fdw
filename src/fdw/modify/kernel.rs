#![forbid(unsafe_code)]
//! Pure rewrite kernel for the UPDATE/DELETE path. No FFI, no I/O — given a
//! sequence of input `RecordBatch`es and a `BlobEdits` map, produce the
//! output `RecordBatch`es with rows deleted or column values overridden.
//!
//! The kernel operates on **absolute row index within the source blob**, so
//! the caller (modify/update.rs) is responsible for converting `(blob_id,
//! offset)` ctids into absolute rows via `BlobIdEntry::chunk_base_row`.

use crate::error::{FdwError, FdwResult};
use arrow::array::{ArrayRef, RecordBatch};
use arrow::compute::{concat_batches, filter_record_batch};
use arrow::datatypes::SchemaRef;
use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct BlobEdits {
    pub deletes: roaring::RoaringBitmap,
    pub updates: HashMap<u64, RowOverride>,
}

#[derive(Debug, Clone)]
pub struct RowOverride {
    /// `values[i]` is a single-row Arrow array for column `i` (Some) or None
    /// to leave the original value untouched.
    pub values: Vec<Option<ArrayRef>>,
}

pub fn apply_edits(
    input: Vec<RecordBatch>,
    schema: SchemaRef,
    edits: &BlobEdits,
) -> FdwResult<Vec<RecordBatch>> {
    // Apply overrides column-by-column, then drop rows that appear in
    // `deletes`. Going column-by-column lets us reuse arrow's typed
    // primitives (concat + take) without unsafe row-wise reconstruction.

    // 1) Concatenate input batches to a single batch so absolute row indices
    // are stable. With a u64 row index space and typical blob sizes ≤ a few
    // million rows, this is fine; for very large blobs the kernel could
    // operate per-batch with a running offset, but v1 favours simplicity.
    let merged = concat_batches(&schema, input.iter()).map_err(FdwError::Arrow)?;
    let nrows = merged.num_rows() as u64;

    // 2) Apply column overrides.
    let mut columns: Vec<ArrayRef> = merged.columns().to_vec();
    if !edits.updates.is_empty() {
        // Pre-validate row indices once.
        for &row in edits.updates.keys() {
            if row >= nrows {
                return Err(FdwError::SchemaMismatch(format!(
                    "update row {row} >= blob row count {nrows}"
                )));
            }
        }
        let mut sorted_rows: Vec<u64> = edits.updates.keys().copied().collect();
        sorted_rows.sort_unstable();

        for (col_idx, col) in columns.iter_mut().enumerate() {
            // Quick check: does any override touch this column?
            let any = sorted_rows.iter().any(|r| {
                edits
                    .updates
                    .get(r)
                    .and_then(|o| o.values.get(col_idx).and_then(|v| v.as_ref()))
                    .is_some()
            });
            if !any {
                continue;
            }
            // Build a new column by walking row-by-row through the existing
            // array and the override map. Use arrow's typed concat trick:
            // slice the array into N+1 chunks and splice the single-row
            // override arrays in between. Conceptually O(nrows) per affected
            // column, which is fine since overrides are sparse.
            let mut pieces: Vec<ArrayRef> = Vec::with_capacity(sorted_rows.len() * 2 + 1);
            let mut cursor: u64 = 0;
            for &row in &sorted_rows {
                if row > cursor {
                    let slice = col.slice(cursor as usize, (row - cursor) as usize);
                    pieces.push(slice);
                }
                let ovr = edits.updates.get(&row).expect("present");
                match ovr.values.get(col_idx).and_then(|v| v.as_ref()) {
                    Some(replacement) => {
                        if replacement.len() != 1 {
                            return Err(FdwError::SchemaMismatch(format!(
                                "override array for col {col_idx} row {row} has \
                                 len={}, expected 1",
                                replacement.len()
                            )));
                        }
                        pieces.push(replacement.clone());
                    }
                    None => {
                        let slice = col.slice(row as usize, 1);
                        pieces.push(slice);
                    }
                }
                cursor = row + 1;
            }
            if cursor < nrows {
                let tail = col.slice(cursor as usize, (nrows - cursor) as usize);
                pieces.push(tail);
            }
            let refs: Vec<&dyn arrow::array::Array> = pieces.iter().map(|p| p.as_ref()).collect();
            *col = arrow::compute::concat(&refs).map_err(FdwError::Arrow)?;
        }
    }
    let updated = RecordBatch::try_new(schema.clone(), columns).map_err(FdwError::Arrow)?;

    // 3) Apply deletes via a packed BooleanArray mask.
    //
    // The mask is built from a `BooleanBufferBuilder` (packed: 1 bit per
    // row, 8 rows per byte), not a `Vec<bool>` (8 bits per row). For a
    // worst-case u32::MAX-row blob the packed form is ~512 MiB instead of
    // 4 GiB — well within the `MAX_BLOB_BYTES`-derived envelope. We iterate
    // the roaring bitmap (sorted, sparse) rather than probing per-row, so
    // the work is proportional to `deletes.len()`, not `nrows`.
    if edits.deletes.is_empty() {
        return Ok(vec![updated]);
    }
    let nrows_usize: usize = nrows
        .try_into()
        .map_err(|_| FdwError::SchemaMismatch(format!("blob row count {nrows} exceeds usize")))?;
    let mut builder = arrow::array::BooleanBufferBuilder::new(nrows_usize);
    builder.append_n(nrows_usize, true);
    for row in edits.deletes.iter() {
        let r = row as usize;
        if r < nrows_usize {
            builder.set_bit(r, false);
        }
    }
    let mask = arrow::array::BooleanArray::new(builder.finish(), None);
    let filtered = filter_record_batch(&updated, &mask).map_err(FdwError::Arrow)?;
    Ok(vec![filtered])
}
