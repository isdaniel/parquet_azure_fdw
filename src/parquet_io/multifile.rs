#![forbid(unsafe_code)]
//! K-way merge across N pre-sorted parquet streams. See SP-3c spec for
//! correctness rules (ASC + NULLS LAST, opt-in via files_in_order, etc.).

use crate::error::{FdwError, FdwResult};
use arrow::array::{Array, RecordBatch};
use arrow::datatypes::DataType;
use futures::StreamExt;
use parquet::arrow::async_reader::ParquetRecordBatchStream;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

/// Sort-key value with NULLS LAST semantics.
#[derive(Debug, Clone, PartialEq)]
pub enum SortKeyValue {
    Null,
    I64(i64),
    F64(f64),
    Utf8(String),
    Date32(i32),
    TimestampMicros(i64),
}

impl Eq for SortKeyValue {}

impl Ord for SortKeyValue {
    fn cmp(&self, other: &Self) -> Ordering {
        // NULLS LAST: Null is greater than any non-null.
        match (self, other) {
            (SortKeyValue::Null, SortKeyValue::Null) => Ordering::Equal,
            (SortKeyValue::Null, _) => Ordering::Greater,
            (_, SortKeyValue::Null) => Ordering::Less,
            (SortKeyValue::I64(a), SortKeyValue::I64(b)) => a.cmp(b),
            (SortKeyValue::F64(a), SortKeyValue::F64(b)) => {
                a.partial_cmp(b).unwrap_or(Ordering::Equal)
            }
            (SortKeyValue::Utf8(a), SortKeyValue::Utf8(b)) => a.cmp(b),
            (SortKeyValue::Date32(a), SortKeyValue::Date32(b)) => a.cmp(b),
            (SortKeyValue::TimestampMicros(a), SortKeyValue::TimestampMicros(b)) => a.cmp(b),
            // Mixed-type comparison: treat as Equal to keep heap monotonic
            // (caller must enforce schema agreement).
            _ => Ordering::Equal,
        }
    }
}

impl PartialOrd for SortKeyValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Extract a `SortKeyValue` from a column at a given row index. Supports the
/// SP-3c v1 type set (rejects timestamptz, decimals — same as SP-1).
pub fn extract_sort_key(col: &dyn Array, row: usize) -> FdwResult<SortKeyValue> {
    use arrow::array::*;
    if col.is_null(row) {
        return Ok(SortKeyValue::Null);
    }
    Ok(match col.data_type() {
        DataType::Int16 => SortKeyValue::I64(
            col.as_any()
                .downcast_ref::<Int16Array>()
                .unwrap()
                .value(row) as i64,
        ),
        DataType::Int32 => SortKeyValue::I64(
            col.as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(row) as i64,
        ),
        DataType::Int64 => SortKeyValue::I64(
            col.as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(row),
        ),
        DataType::Float32 => SortKeyValue::F64(
            col.as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .value(row) as f64,
        ),
        DataType::Float64 => SortKeyValue::F64(
            col.as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(row),
        ),
        DataType::Utf8 => SortKeyValue::Utf8(
            col.as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(row)
                .to_string(),
        ),
        DataType::Date32 => SortKeyValue::Date32(
            col.as_any()
                .downcast_ref::<Date32Array>()
                .unwrap()
                .value(row),
        ),
        DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None) => {
            SortKeyValue::TimestampMicros(
                col.as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .unwrap()
                    .value(row),
            )
        }
        other => {
            return Err(FdwError::SchemaMismatch(format!(
                "sort col type {other:?} not supported in v1"
            )))
        }
    })
}

/// One heap entry = (key tuple, source index, batch holding the row, row offset).
struct HeapEntry {
    key: Vec<SortKeyValue>,
    source_idx: usize,
    batch: RecordBatch,
    row_in_batch: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl Eq for HeapEntry {}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// K-way merge across N parquet streams. Generic over the reader type so
/// callers can use both `AzureBlobReader` (production) and `Cursor<Bytes>`
/// (in-process tests).
pub struct MultiFileSortedStream<R>
where
    R: parquet::arrow::async_reader::AsyncFileReader + Unpin + Send + 'static,
{
    streams: Vec<Option<ParquetRecordBatchStream<R>>>,
    blob_names: Vec<String>,
    sort_col_indices: Vec<usize>,
    heap: BinaryHeap<Reverse<HeapEntry>>,
    last_emitted: Option<Vec<SortKeyValue>>,
}

impl<R> MultiFileSortedStream<R>
where
    R: parquet::arrow::async_reader::AsyncFileReader + Unpin + Send + 'static,
{
    /// Build a new merger. `sort_col_indices` are 0-based PARQUET column
    /// indices (caller must translate from foreign-table attno).
    pub async fn new(
        mut streams: Vec<ParquetRecordBatchStream<R>>,
        blob_names: Vec<String>,
        sort_col_indices: Vec<usize>,
    ) -> FdwResult<Self> {
        if streams.len() > 256 {
            return Err(FdwError::SchemaMismatch(format!(
                "sorted-merge cannot open {} blobs at once (cap 256) — narrow the filename glob or partition filter",
                streams.len()
            )));
        }
        if streams.len() != blob_names.len() {
            return Err(FdwError::SchemaMismatch(
                "streams.len() != blob_names.len()".into(),
            ));
        }
        let mut heap: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::new();
        let mut wrapped_streams: Vec<Option<ParquetRecordBatchStream<R>>> =
            Vec::with_capacity(streams.len());
        for (i, stream) in streams.iter_mut().enumerate() {
            // Pull first batch from this stream.
            match stream.next().await {
                Some(Ok(batch)) if batch.num_rows() > 0 => {
                    let key = build_key(&batch, 0, &sort_col_indices)?;
                    heap.push(Reverse(HeapEntry {
                        key,
                        source_idx: i,
                        batch,
                        row_in_batch: 0,
                    }));
                }
                Some(Ok(_)) => { /* empty first batch — leave it out of the heap */ }
                Some(Err(e)) => return Err(FdwError::from(e)),
                None => { /* empty blob — leave it out of the heap */ }
            }
        }
        // Move the streams into our owned Vec.
        for s in streams {
            wrapped_streams.push(Some(s));
        }
        Ok(Self {
            streams: wrapped_streams,
            blob_names,
            sort_col_indices,
            heap,
            last_emitted: None,
        })
    }

    /// Return the next row (batch + row offset) in merged order. Returns
    /// `None` at end-of-stream. Raises `SchemaMismatch` if the
    /// `files_in_order` invariant is violated.
    pub async fn next_row(&mut self) -> FdwResult<Option<(RecordBatch, usize)>> {
        let entry = match self.heap.pop() {
            Some(Reverse(e)) => e,
            None => return Ok(None),
        };
        // Invariant check.
        if let Some(prev) = &self.last_emitted {
            if entry.key < *prev {
                return Err(FdwError::SchemaMismatch(format!(
                    "blob '{}' violates sorted-merge precondition: row key {:?} < previous {:?}",
                    self.blob_names[entry.source_idx], entry.key, prev
                )));
            }
        }
        self.last_emitted = Some(entry.key.clone());
        let result = (entry.batch.clone(), entry.row_in_batch);
        // Advance this source.
        let source_idx = entry.source_idx;
        let next_row = entry.row_in_batch + 1;
        if next_row < entry.batch.num_rows() {
            // Still rows left in current batch.
            let key = build_key(&entry.batch, next_row, &self.sort_col_indices)?;
            self.heap.push(Reverse(HeapEntry {
                key,
                source_idx,
                batch: entry.batch,
                row_in_batch: next_row,
            }));
        } else {
            // Current batch exhausted; pull next non-empty batch from stream.
            loop {
                let stream = self.streams[source_idx].as_mut().ok_or_else(|| {
                    FdwError::SchemaMismatch(format!("stream {source_idx} unexpectedly absent"))
                })?;
                match stream.next().await {
                    Some(Ok(batch)) => {
                        if batch.num_rows() > 0 {
                            let key = build_key(&batch, 0, &self.sort_col_indices)?;
                            self.heap.push(Reverse(HeapEntry {
                                key,
                                source_idx,
                                batch,
                                row_in_batch: 0,
                            }));
                            break;
                        }
                        // Empty batch — keep pulling.
                    }
                    Some(Err(e)) => return Err(FdwError::from(e)),
                    None => {
                        // Stream exhausted.
                        self.streams[source_idx] = None;
                        break;
                    }
                }
            }
        }
        Ok(Some(result))
    }
}

fn build_key(
    batch: &RecordBatch,
    row: usize,
    sort_col_indices: &[usize],
) -> FdwResult<Vec<SortKeyValue>> {
    let mut key = Vec::with_capacity(sort_col_indices.len());
    for &col_idx in sort_col_indices {
        let col = batch.column(col_idx).as_ref();
        key.push(extract_sort_key(col, row)?);
    }
    Ok(key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use arrow::datatypes::{Field, Schema};
    use bytes::Bytes;
    use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use std::sync::Arc;

    fn build_int_blob(ids: &[i64]) -> Bytes {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)]));
        let mut buf: Vec<u8> = Vec::new();
        let mut w = ArrowWriter::try_new(
            &mut buf,
            schema.clone(),
            Some(WriterProperties::builder().build()),
        )
        .unwrap();
        let arr: Vec<Option<i64>> = ids.iter().map(|v| Some(*v)).collect();
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(arr))]).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
        Bytes::from(buf)
    }

    async fn open_int_stream(bytes: Bytes) -> ParquetRecordBatchStream<std::io::Cursor<Bytes>> {
        let cursor = std::io::Cursor::new(bytes);
        ParquetRecordBatchStreamBuilder::new(cursor)
            .await
            .unwrap()
            .build()
            .unwrap()
    }

    #[test]
    fn sort_key_nulls_last() {
        assert!(SortKeyValue::Null > SortKeyValue::I64(i64::MAX));
        assert!(SortKeyValue::I64(0) < SortKeyValue::Null);
        assert!(SortKeyValue::Null == SortKeyValue::Null);
    }

    #[test]
    fn sort_key_ordering_within_type() {
        assert!(SortKeyValue::I64(1) < SortKeyValue::I64(2));
        assert!(SortKeyValue::Utf8("a".into()) < SortKeyValue::Utf8("b".into()));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn happy_path_three_blobs_merged() {
        let a = build_int_blob(&[1, 4, 7]);
        let b = build_int_blob(&[2, 5, 8]);
        let c = build_int_blob(&[3, 6, 9]);
        let streams = vec![
            open_int_stream(a).await,
            open_int_stream(b).await,
            open_int_stream(c).await,
        ];
        let names = vec!["a.parquet".into(), "b.parquet".into(), "c.parquet".into()];
        let mut merger = MultiFileSortedStream::new(streams, names, vec![0])
            .await
            .unwrap();
        let mut got = Vec::new();
        while let Some((batch, row)) = merger.next_row().await.unwrap() {
            let id = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(row);
            got.push(id);
        }
        assert_eq!(got, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn invariant_violation_raises() {
        // Blob "bad" has rows [5, 3] — second row violates ascending order.
        // good's first row is 10, so the heap pops bad's row 0 (=5) first,
        // then bad's row 1 (=3) next — 3 < 5 trips the invariant.
        let bad = build_int_blob(&[5, 3]);
        let good = build_int_blob(&[10]);
        let streams = vec![open_int_stream(bad).await, open_int_stream(good).await];
        let names = vec!["bad.parquet".into(), "good.parquet".into()];
        let mut merger = MultiFileSortedStream::new(streams, names, vec![0])
            .await
            .unwrap();
        let _ = merger.next_row().await.unwrap(); // emits 5
        let result = merger.next_row().await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{err}").contains("bad.parquet"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn empty_blob_in_mix_is_ok() {
        let empty = build_int_blob(&[]);
        let one = build_int_blob(&[42]);
        let streams = vec![open_int_stream(empty).await, open_int_stream(one).await];
        let names = vec!["empty.parquet".into(), "one.parquet".into()];
        let mut merger = MultiFileSortedStream::new(streams, names, vec![0])
            .await
            .unwrap();
        let (batch, row) = merger.next_row().await.unwrap().unwrap();
        assert_eq!(
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(row),
            42
        );
        assert!(merger.next_row().await.unwrap().is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn k_cap_exceeded_errors_at_construction() {
        let bytes = build_int_blob(&[1]);
        let mut streams = Vec::new();
        let mut names = Vec::new();
        for i in 0..257 {
            streams.push(open_int_stream(bytes.clone()).await);
            names.push(format!("b{i}.parquet"));
        }
        let result = MultiFileSortedStream::new(streams, names, vec![0]).await;
        let err = match result {
            Ok(_) => panic!("expected K-cap error, got Ok"),
            Err(e) => e,
        };
        assert!(format!("{err:?}").contains("256"));
    }
}
