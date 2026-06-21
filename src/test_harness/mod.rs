//! In-crate test harness, gated by `cfg(feature = "pg_test")`.
//!
//! pgrx's `#[pg_test]` macro requires the test function to live inside the
//! extension lib so the resulting SQL stub is installed into the regression
//! database. Integration tests under `tests/*.rs` are a separate cargo binary
//! whose `#[pg_extern]` registrations are invisible to the loaded `.so`, so
//! the SELECT/INSERT/UPDATE/DELETE suites live here.
//!
//! Storage is provided by [`fake_blob_store::FakeBlobStore`] — a stateful,
//! in-memory wiremock fake that speaks just enough of the Azure Blob REST
//! protocol for the FDW's happy paths. There is no Azurite, no Docker, no
//! Azure credential involved.

#![allow(dead_code)] // not every test binary consumes every helper.

pub mod fake_blob_store;

pub use fake_blob_store::FakeBlobStore;

// ---------------------------------------------------------------------------
// Parquet fixture builders — used by `#[pg_test]` cases to author tiny
// fixture blobs in-process. These are pure functions; they do not touch any
// HTTP / storage layer.
// ---------------------------------------------------------------------------

use arrow::array::{Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use bytes::Bytes;
use std::sync::Arc;

/// Build a tiny `(id BIGINT, name TEXT)` parquet blob.
pub fn build_simple_parquet(ids: &[i64], names: &[&str]) -> Bytes {
    assert_eq!(ids.len(), names.len());
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let id_array: Arc<dyn Array> = Arc::new(Int64Array::from(ids.to_vec()));
    let name_array: Arc<dyn Array> = Arc::new(StringArray::from(names.to_vec()));
    let batch =
        RecordBatch::try_new(schema.clone(), vec![id_array, name_array]).expect("RecordBatch");
    let mut w =
        crate::parquet_io::ParquetBatchWriter::new(schema, crate::parquet_io::Compression::Snappy)
            .expect("writer");
    w.write(&batch).expect("write batch");
    w.finish().expect("finish parquet")
}

/// Build a 12-row parquet file with three 4-row row-groups whose `id` columns
/// fall into disjoint ranges (0..3, 100..103, 200..203). Useful for any test
/// that wants to verify row-group pruning behaviour.
pub fn build_multi_rowgroup_parquet() -> Bytes {
    use parquet::arrow::ArrowWriter;
    use parquet::basic::Compression as PqCompression;
    use parquet::file::properties::WriterProperties;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let props = WriterProperties::builder()
        .set_compression(PqCompression::SNAPPY)
        .set_max_row_group_row_count(Some(4))
        .build();
    let mut buf: Vec<u8> = Vec::with_capacity(64 * 1024);
    let mut w = ArrowWriter::try_new(&mut buf, schema.clone(), Some(props)).expect("ArrowWriter");

    for base in [0_i64, 100, 200] {
        let ids: Vec<i64> = (base..base + 4).collect();
        let names: Vec<String> = ids.iter().map(|i| format!("n{i}")).collect();
        let id_array: Arc<dyn Array> = Arc::new(Int64Array::from(ids));
        let name_array: Arc<dyn Array> = Arc::new(StringArray::from(names));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![id_array, name_array]).expect("RecordBatch");
        w.write(&batch).expect("write batch");
        w.flush().expect("flush row-group");
    }
    w.close().expect("close writer");
    Bytes::from(buf)
}
