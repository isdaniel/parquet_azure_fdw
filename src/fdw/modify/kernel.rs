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
use arrow::compute::filter_record_batch;
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

/// Transform ONE source batch in the UPDATE/DELETE rewrite: apply column
/// overrides for absolute rows in `[base_offset, base_offset + n)`, then
/// drop deleted rows in that same range. Pure; no I/O.
///
/// `base_offset` is the absolute row index of `batch`'s first row within
/// the source blob. Edits whose absolute row falls outside this batch's
/// range are ignored here (a later batch owns them).
pub fn apply_edits_batch(
    batch: &RecordBatch,
    base_offset: u64,
    schema: &SchemaRef,
    edits: &BlobEdits,
) -> FdwResult<RecordBatch> {
    let n = batch.num_rows() as u64;
    let end = base_offset + n; // exclusive

    // 1) Column overrides for absolute rows in [base_offset, end).
    let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
    if !edits.updates.is_empty() {
        // Local sorted list of (absolute_row) hits within this batch.
        let mut hits: Vec<u64> = edits
            .updates
            .keys()
            .copied()
            .filter(|&k| k >= base_offset && k < end)
            .collect();
        if !hits.is_empty() {
            hits.sort_unstable();
            for (col_idx, col) in columns.iter_mut().enumerate() {
                // Does any in-range override touch this column?
                let any = hits.iter().any(|abs| {
                    edits
                        .updates
                        .get(abs)
                        .and_then(|o| o.values.get(col_idx).and_then(|v| v.as_ref()))
                        .is_some()
                });
                if !any {
                    continue;
                }
                // Splice single-row override arrays in between slices of the
                // existing column, indexed by LOCAL row = abs - base_offset.
                let mut pieces: Vec<ArrayRef> = Vec::with_capacity(hits.len() * 2 + 1);
                let mut cursor: u64 = 0;
                for &abs in &hits {
                    let local = abs - base_offset;
                    if local > cursor {
                        pieces.push(col.slice(cursor as usize, (local - cursor) as usize));
                    }
                    let ovr = edits.updates.get(&abs).expect("present");
                    match ovr.values.get(col_idx).and_then(|v| v.as_ref()) {
                        Some(replacement) => {
                            if replacement.len() != 1 {
                                return Err(FdwError::SchemaMismatch(format!(
                                    "override array for col {col_idx} abs row {abs} has \
                                     len={}, expected 1",
                                    replacement.len()
                                )));
                            }
                            pieces.push(replacement.clone());
                        }
                        None => pieces.push(col.slice(local as usize, 1)),
                    }
                    cursor = local + 1;
                }
                if cursor < n {
                    pieces.push(col.slice(cursor as usize, (n - cursor) as usize));
                }
                let refs: Vec<&dyn arrow::array::Array> =
                    pieces.iter().map(|p| p.as_ref()).collect();
                *col = arrow::compute::concat(&refs).map_err(FdwError::Arrow)?;
            }
        }
    }
    let updated = RecordBatch::try_new(schema.clone(), columns).map_err(FdwError::Arrow)?;

    // 2) Deletes: a packed per-batch mask. Iterate the roaring bitmap's
    // range for this batch (sparse, sorted) rather than probing per row.
    //
    // `BlobEdits.deletes` is a u32-keyed `RoaringBitmap`; blobs are capped at
    // `u32::MAX` rows up front in `build_plan`, so both `base_offset` and
    // `end` fit u32 (clamp `end` defensively at the u32 boundary).
    let n_usize = batch.num_rows();
    let lo = base_offset as u32;
    let hi = end.min(u32::MAX as u64) as u32;
    let has_deletes_in_range = edits.deletes.range(lo..hi).next().is_some();
    if !has_deletes_in_range {
        return Ok(updated);
    }
    let mut builder = arrow::array::BooleanBufferBuilder::new(n_usize);
    builder.append_n(n_usize, true);
    for abs in edits.deletes.range(lo..hi) {
        let local = (abs as u64 - base_offset) as usize;
        builder.set_bit(local, false);
    }
    let mask = arrow::array::BooleanArray::new(builder.finish(), None);
    filter_record_batch(&updated, &mask).map_err(FdwError::Arrow)
}

/// Streaming wrapper: apply edits batch-by-batch with a running absolute
/// offset. No whole-blob concatenation. Kept so existing pure tests and any
/// non-streaming caller retain the `Vec<RecordBatch>` signature.
pub fn apply_edits(
    input: Vec<RecordBatch>,
    schema: SchemaRef,
    edits: &BlobEdits,
) -> FdwResult<Vec<RecordBatch>> {
    let mut out = Vec::with_capacity(input.len());
    let mut offset: u64 = 0;
    for batch in &input {
        out.push(apply_edits_batch(batch, offset, &schema, edits)?);
        offset += batch.num_rows() as u64;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn schema_two() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn batch(schema: &SchemaRef, ids: &[i64], names: &[&str]) -> RecordBatch {
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef,
                Arc::new(StringArray::from(names.to_vec())) as ArrayRef,
            ],
        )
        .unwrap()
    }

    fn ids_of(b: &RecordBatch) -> Vec<i64> {
        let c = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        (0..b.num_rows()).map(|i| c.value(i)).collect()
    }

    fn names_of(b: &RecordBatch) -> Vec<String> {
        let c = b.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        (0..b.num_rows()).map(|i| c.value(i).to_string()).collect()
    }

    #[test]
    fn batch_no_edits_is_identity() {
        let s = schema_two();
        let b = batch(&s, &[1, 2, 3], &["a", "b", "c"]);
        let edits = BlobEdits::default();
        let out = apply_edits_batch(&b, 0, &s, &edits).unwrap();
        assert_eq!(ids_of(&out), vec![1, 2, 3]);
        assert_eq!(names_of(&out), vec!["a", "b", "c"]);
    }

    #[test]
    fn batch_delete_uses_absolute_offset() {
        // Second batch in a blob: base_offset = 3. Delete absolute row 4
        // (local row 1 of this batch).
        let s = schema_two();
        let b = batch(&s, &[10, 11, 12], &["x", "y", "z"]);
        let mut edits = BlobEdits::default();
        edits.deletes.insert(4); // absolute → local index 1
        let out = apply_edits_batch(&b, 3, &s, &edits).unwrap();
        assert_eq!(ids_of(&out), vec![10, 12]);
        assert_eq!(names_of(&out), vec!["x", "z"]);
    }

    #[test]
    fn batch_update_uses_absolute_offset() {
        // base_offset = 3, override absolute row 5 (local index 2), name col.
        let s = schema_two();
        let b = batch(&s, &[10, 11, 12], &["x", "y", "z"]);
        let mut edits = BlobEdits::default();
        edits.updates.insert(
            5,
            RowOverride {
                values: vec![
                    None,
                    Some(Arc::new(StringArray::from(vec!["Z"])) as ArrayRef),
                ],
            },
        );
        let out = apply_edits_batch(&b, 3, &s, &edits).unwrap();
        assert_eq!(ids_of(&out), vec![10, 11, 12]);
        assert_eq!(names_of(&out), vec!["x", "y", "Z"]);
    }

    #[test]
    fn batch_ignores_edits_outside_its_range() {
        // base_offset = 0, n = 3, so this batch owns absolute rows 0..3.
        // An edit at absolute row 7 must NOT touch this batch.
        let s = schema_two();
        let b = batch(&s, &[1, 2, 3], &["a", "b", "c"]);
        let mut edits = BlobEdits::default();
        edits.deletes.insert(7);
        edits.updates.insert(
            9,
            RowOverride {
                values: vec![Some(Arc::new(Int64Array::from(vec![99])) as ArrayRef), None],
            },
        );
        let out = apply_edits_batch(&b, 0, &s, &edits).unwrap();
        assert_eq!(ids_of(&out), vec![1, 2, 3]);
        assert_eq!(names_of(&out), vec!["a", "b", "c"]);
    }

    #[test]
    fn batch_all_rows_deleted_yields_zero_rows() {
        let s = schema_two();
        let b = batch(&s, &[1, 2], &["a", "b"]);
        let mut edits = BlobEdits::default();
        edits.deletes.insert(0);
        edits.deletes.insert(1);
        let out = apply_edits_batch(&b, 0, &s, &edits).unwrap();
        assert_eq!(out.num_rows(), 0);
        assert_eq!(out.schema(), s); // schema preserved even at 0 rows
    }

    #[test]
    fn wrapper_streams_multiple_batches_with_running_offset() {
        // Two batches; delete absolute row 1 (batch 0 local 1) and update
        // absolute row 3 (batch 1 local 1). Wrapper must thread the offset.
        let s = schema_two();
        let b0 = batch(&s, &[1, 2], &["a", "b"]);
        let b1 = batch(&s, &[3, 4], &["c", "d"]);
        let mut edits = BlobEdits::default();
        edits.deletes.insert(1);
        edits.updates.insert(
            3,
            RowOverride {
                values: vec![
                    None,
                    Some(Arc::new(StringArray::from(vec!["D"])) as ArrayRef),
                ],
            },
        );
        let out = apply_edits(vec![b0, b1], s.clone(), &edits).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(ids_of(&out[0]), vec![1]); // row 2 (id=2) deleted
        assert_eq!(names_of(&out[0]), vec!["a"]);
        assert_eq!(ids_of(&out[1]), vec![3, 4]);
        assert_eq!(names_of(&out[1]), vec!["c", "D"]); // id=4 row's name overridden
    }
}
