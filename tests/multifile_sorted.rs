//! End-to-end SP-3c sorted-merge tests against the in-process fake blob store.
//!
//! These drive `MultiFileSortedStream` through the REAL `AzureBlobClient` +
//! wiremock fake wiring (range-GET parquet reads), as opposed to the Task 2
//! in-crate unit tests which feed `Cursor<Bytes>` directly. The three cases
//! mirror the brief: a happy 3-blob global merge, an invariant-violation that
//! names the offending blob, and an empty blob mixed into the merge set.
//!
//! NULLS-LAST, multi-column key, and the K-cap (>256 blobs) edge cases are
//! already covered by the Task 2 in-crate unit tests in
//! `src/parquet_io/multifile.rs`; they are not re-run here.
#![cfg(feature = "pg_test")]

#[path = "common/mod.rs"]
mod common;

use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use bytes::Bytes;
use common::fake_blob_store::FakeBlobStore;
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use parquet_azure_fdw::azure::{AzureBlobClient, Credential};
use parquet_azure_fdw::parquet_io::multifile::MultiFileSortedStream;
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Build a single-column `(id BIGINT)` parquet blob from the given values.
/// All values are non-null.
fn make_int_parquet(ids: &[i64]) -> Bytes {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)]));
    let mut buf: Vec<u8> = Vec::new();
    let mut w = ArrowWriter::try_new(
        &mut buf,
        schema.clone(),
        Some(WriterProperties::builder().build()),
    )
    .unwrap();
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(ids.to_vec()))]).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    Bytes::from(buf)
}

fn make_client(fake: &FakeBlobStore, container: &str) -> AzureBlobClient {
    AzureBlobClient::new(
        "fake.invalid",
        "devstoreaccount1",
        Credential::SasUrl {
            container_url: fake.sas_url(container),
        },
        container,
    )
    .unwrap()
}

/// Open every named blob through the real client and build the K-way merger.
fn build_merger(
    rt: &Runtime,
    client: &AzureBlobClient,
    blobs: &[String],
    sort_cols: Vec<usize>,
) -> MultiFileSortedStream<parquet_azure_fdw::azure::AzureBlobReader> {
    let mut streams = Vec::new();
    for name in blobs {
        let reader = client.open_blob(name);
        streams.push(
            rt.block_on(ParquetRecordBatchStreamBuilder::new(reader))
                .unwrap()
                .build()
                .unwrap(),
        );
    }
    rt.block_on(MultiFileSortedStream::new(
        streams,
        blobs.to_vec(),
        sort_cols,
    ))
    .unwrap()
}

fn id_at(batch: &RecordBatch, row: usize) -> i64 {
    batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(row)
}

#[test]
fn happy_three_sorted_blobs_merged_globally() {
    let fake = FakeBlobStore::start_blocking();
    let rt = Runtime::new().unwrap();
    let client = make_client(&fake, "c");
    rt.block_on(client.put_if_none_match("a.parquet", make_int_parquet(&[1, 4, 7])))
        .unwrap();
    rt.block_on(client.put_if_none_match("b.parquet", make_int_parquet(&[2, 5, 8])))
        .unwrap();
    rt.block_on(client.put_if_none_match("c.parquet", make_int_parquet(&[3, 6, 9])))
        .unwrap();

    let blobs = vec![
        "a.parquet".to_string(),
        "b.parquet".to_string(),
        "c.parquet".to_string(),
    ];
    let mut merger = build_merger(&rt, &client, &blobs, vec![0]);

    let mut got = Vec::new();
    while let Some((batch, row)) = rt.block_on(merger.next_row()).unwrap() {
        got.push(id_at(&batch, row));
    }
    assert_eq!(got, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
}

#[test]
fn invariant_violation_raises_with_blob_name() {
    let fake = FakeBlobStore::start_blocking();
    let rt = Runtime::new().unwrap();
    let client = make_client(&fake, "c");
    // Blob has unsorted rows: [5, 3] — the second pop violates ascending order.
    rt.block_on(client.put_if_none_match("bad.parquet", make_int_parquet(&[5, 3])))
        .unwrap();

    let blobs = vec!["bad.parquet".to_string()];
    let mut merger = build_merger(&rt, &client, &blobs, vec![0]);

    // First row pops fine (=5).
    let (batch, row) = rt.block_on(merger.next_row()).unwrap().unwrap();
    assert_eq!(id_at(&batch, row), 5);
    // Second row (=3) violates the files_in_order precondition.
    let err = rt
        .block_on(merger.next_row())
        .expect_err("must raise on out-of-order row");
    assert!(
        format!("{err}").contains("bad.parquet"),
        "error must name the offending blob; got: {err}"
    );
}

#[test]
fn empty_blob_in_mix_does_not_panic() {
    let fake = FakeBlobStore::start_blocking();
    let rt = Runtime::new().unwrap();
    let client = make_client(&fake, "c");
    rt.block_on(client.put_if_none_match("empty.parquet", make_int_parquet(&[])))
        .unwrap();
    rt.block_on(client.put_if_none_match("one.parquet", make_int_parquet(&[42])))
        .unwrap();

    let blobs = vec!["empty.parquet".to_string(), "one.parquet".to_string()];
    let mut merger = build_merger(&rt, &client, &blobs, vec![0]);

    let (batch, row) = rt.block_on(merger.next_row()).unwrap().unwrap();
    assert_eq!(id_at(&batch, row), 42);
    assert!(rt.block_on(merger.next_row()).unwrap().is_none());
}

/// Build a single-column `(id UTF8)` parquet blob.
fn make_str_parquet(ids: &[&str]) -> Bytes {
    use arrow::array::StringArray;
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, true)]));
    let mut buf: Vec<u8> = Vec::new();
    let mut w = ArrowWriter::try_new(
        &mut buf,
        schema.clone(),
        Some(WriterProperties::builder().build()),
    )
    .unwrap();
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(ids.to_vec()))]).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    Bytes::from(buf)
}

/// SP-3c SAME-TYPE GUARD (Task 2/3 review): when two merged blobs declare the
/// same sort-column NAME but DIFFERENT physical arrow types (Int64 vs Utf8),
/// `MultiFileSortedStream` must NOT silently mis-order rows. The guard that
/// catches this lives in `fdw::scan::build_state`, which reads each blob's
/// arrow schema and compares the sort column's `DataType` across blobs BEFORE
/// building the merger. That guard requires a live `ForeignScanState`, so it
/// cannot be unit-tested in-process; raising it from a `#[pg_test]` aborts the
/// backend (ereport `siglongjmp` through the tokio/SDK Rust frames — see
/// `src/lib.rs` for the full deferral note).
///
/// This test reproduces the guard's EXACT comparison directly against the fake
/// blob store: it opens both builders, reads their schemas, and asserts the
/// sort column's `DataType` differs (which is precisely the condition
/// `build_state` rejects). It does NOT exercise `error::raise` — only the
/// detection logic. The widening case (Int64 vs Int32, both -> I64) is allowed
/// by the merger and is intentionally not flagged here.
#[test]
fn same_type_guard_detects_mismatched_sort_col_type() {
    let fake = FakeBlobStore::start_blocking();
    let rt = Runtime::new().unwrap();
    let client = make_client(&fake, "c");
    rt.block_on(client.put_if_none_match("a.parquet", make_int_parquet(&[1, 2])))
        .unwrap();
    rt.block_on(client.put_if_none_match("b.parquet", make_str_parquet(&["3", "4"])))
        .unwrap();

    // Mirror build_state: read each blob's arrow schema, compare the sort
    // column (parquet index 0) DataType across blobs.
    let sort_col_idx = 0usize;
    let mut first_type: Option<DataType> = None;
    let mut mismatch_blob: Option<String> = None;
    for name in ["a.parquet", "b.parquet"] {
        let reader = client.open_blob(name);
        let builder = rt
            .block_on(ParquetRecordBatchStreamBuilder::new(reader))
            .unwrap();
        let this = builder.schema().field(sort_col_idx).data_type().clone();
        match &first_type {
            None => first_type = Some(this),
            Some(first) => {
                if &this != first {
                    mismatch_blob = Some(name.to_string());
                }
            }
        }
    }

    assert_eq!(first_type, Some(DataType::Int64));
    assert_eq!(
        mismatch_blob.as_deref(),
        Some("b.parquet"),
        "guard must detect b.parquet's Utf8 sort col disagreeing with a.parquet's Int64"
    );
}

/// SP-3c C2 regression (merge-level large-result): the actual C2 panic lived in
/// `fdw::scan::iterate_foreign_scan`, where the non-sorted chunk-registration
/// path indexed an EMPTY `blob_id_table` at the first CHUNK_ROWS (65_536) row
/// boundary → out-of-bounds panic across the FFI boundary. That fix is a guard
/// in scan.rs (`if !sorted_mode { register_chunk_if_boundary(...) }`) only
/// reachable through a live PG scan, so a `cargo test` can't trigger it
/// directly — its end-to-end coverage is the guard + the in-line C2 comment +
/// code review (the fake-blob-in-pg_test limitation documented in
/// `same_type_guard_detects_mismatched_sort_col_type` blocks a >65k-row
/// `#[pg_test]`).
///
/// This test instead proves the MERGE ENGINE ITSELF handles >65_536 rows: two
/// blobs (evens 0..140000 and odds 1..140000) merge to ~140k globally-sorted
/// rows. It drives `MultiFileSortedStream::next_row()` to completion and
/// asserts the full count, monotonic non-decreasing order, and the first/last
/// values — documenting that the row volume that crashed scan.rs is handled
/// correctly at the merge layer.
#[test]
fn large_result_over_chunk_rows_merges_in_order() {
    let fake = FakeBlobStore::start_blocking();
    let rt = Runtime::new().unwrap();
    let client = make_client(&fake, "c");

    // 70_000 evens + 70_000 odds = 140_000 rows total — well past CHUNK_ROWS
    // (65_536), so a single blob's stream alone already crosses the boundary.
    let evens: Vec<i64> = (0..140_000).step_by(2).collect();
    let odds: Vec<i64> = (1..140_000).step_by(2).collect();
    let total = evens.len() + odds.len();
    assert!(
        total > 65_536,
        "test must exceed CHUNK_ROWS to be meaningful"
    );

    rt.block_on(client.put_if_none_match("evens.parquet", make_int_parquet(&evens)))
        .unwrap();
    rt.block_on(client.put_if_none_match("odds.parquet", make_int_parquet(&odds)))
        .unwrap();

    let blobs = vec!["evens.parquet".to_string(), "odds.parquet".to_string()];
    let mut merger = build_merger(&rt, &client, &blobs, vec![0]);

    let mut count = 0usize;
    let mut prev: Option<i64> = None;
    let mut first: Option<i64> = None;
    let mut last: i64 = i64::MIN;
    while let Some((batch, row)) = rt.block_on(merger.next_row()).unwrap() {
        let v = id_at(&batch, row);
        if let Some(p) = prev {
            assert!(v >= p, "merge output must be non-decreasing: {p} then {v}");
        }
        if first.is_none() {
            first = Some(v);
        }
        prev = Some(v);
        last = v;
        count += 1;
    }

    assert_eq!(count, total, "merge must emit every row across both blobs");
    assert_eq!(first, Some(0), "first sorted id is 0");
    assert_eq!(last, 139_999, "last sorted id is 139_999");
}
