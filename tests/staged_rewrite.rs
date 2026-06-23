#![cfg(feature = "pg_test")]
//! Two-phase staged-rewrite integration tests using the wiremock fake.
//! Verifies that mid-statement failures don't leak staging blobs.
//!
//! This is a `#[tokio::test]`, not a `#[pg_test]`: it drives the
//! `StatementCoordinator` + `AzureBlobClient` directly so the cleanup path
//! can be exercised without standing up Postgres. The xact callback that
//! runs cleanup inside a backend (see `fdw::modify::coordinator::xact`)
//! invokes the same `delete_unconditional` calls we issue here.

use bytes::Bytes;
use parquet_azure_fdw::azure::{AzureBlobClient, Credential};
use parquet_azure_fdw::fdw::modify::coordinator::{make_staging_name, StatementCoordinator};
use parquet_azure_fdw::test_harness::fake_blob_store::FakeBlobStore;
/// Build an `AzureBlobClient` pointing at the fake's SAS URL for `container`.
fn client_for(store: &FakeBlobStore, container: &str) -> AzureBlobClient {
    let url = store.sas_url(container);
    AzureBlobClient::new(
        "blob.core.windows.net", // unused for SAS
        "acct",
        Credential::SasUrl { container_url: url },
        container,
    )
    .expect("client")
}

/// If Phase 1 stages blob #1 and then fails before staging blob #2, the
/// controller (or in this non-PG test, the test itself) sweeps the pending
/// staging set. Originals must be untouched and no `.tmp.*` blob may
/// remain visible via a list.
#[tokio::test]
async fn phase1_failure_leaves_no_staging_blobs() {
    let store = FakeBlobStore::start().await;
    let client = client_for(&store, "c");

    // Setup: create two original blobs.
    let etag_a = client
        .put_if_none_match("a.parquet", Bytes::from_static(b"orig-a"))
        .await
        .expect("create a");
    let etag_b = client
        .put_if_none_match("b.parquet", Bytes::from_static(b"orig-b"))
        .await
        .expect("create b");

    // Simulate a Phase-1 write of a's staging blob, then a Phase-1 failure
    // before b's staging is written.
    let mut coord = StatementCoordinator::new();
    let staging_a = make_staging_name("a.parquet");
    coord.register_staging(staging_a.clone());
    client
        .put_if_none_match(&staging_a, Bytes::from_static(b"new-a"))
        .await
        .expect("stage a");

    // Phase 1 fails on b — controller (here: this test) runs cleanup.
    let pending: Vec<String> = coord.pending_staging().map(String::from).collect();
    assert_eq!(pending, vec![staging_a.clone()]);
    for name in &pending {
        client
            .delete_unconditional(name)
            .await
            .expect("cleanup staging");
    }

    // No staging blobs visible; originals intact.
    let names: Vec<String> = client
        .list_with_prefix_etags("")
        .await
        .expect("list")
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert_eq!(
        names,
        vec!["a.parquet".to_string(), "b.parquet".to_string()],
        "no staging blobs leaked"
    );

    let (body_a, etag_a_now) = client.get_with_etag("a.parquet").await.expect("get a");
    assert_eq!(&body_a[..], b"orig-a");
    assert_eq!(etag_a_now, etag_a, "original a etag unchanged");

    let (body_b, etag_b_now) = client.get_with_etag("b.parquet").await.expect("get b");
    assert_eq!(&body_b[..], b"orig-b");
    assert_eq!(etag_b_now, etag_b, "original b etag unchanged");
}

/// When a DELETE removes EVERY row of a blob, `commit_plan` must take the
/// empty-result path: the source blob is DELETEd outright rather than
/// rewritten, so afterwards the container holds neither the original blob
/// NOR any `*.tmp.*` staging blob. This drives the full
/// `ModifyPlan` + `commit_plan` pipeline against the fake.
#[test]
fn all_rows_deleted_routes_to_empty_delete() {
    use arrow::array::{ArrayRef, Int32Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet_azure_fdw::fdw::modify::kernel::BlobEdits;
    use parquet_azure_fdw::fdw::modify::update::{commit_plan, ModifyPlan};
    use parquet_azure_fdw::fdw::modify::BlobIdEntry;
    use parquet_azure_fdw::parquet_io::writer::{Compression, ParquetBatchWriter};
    use std::collections::HashMap;
    use std::sync::Arc;

    let fake = FakeBlobStore::start_blocking();
    let container = "c-empty-delete";
    let client = client_for(&fake, container);

    // Seed a 2-row parquet blob.
    let schema = Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int32, true),
        Field::new("s", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
            Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef,
        ],
    )
    .expect("build fixture batch");
    let mut w =
        ParquetBatchWriter::new(schema.clone(), Compression::Snappy).expect("ParquetBatchWriter");
    w.write(&batch).expect("write fixture batch");
    let body = w.finish().expect("finalise fixture parquet");
    let etag = fake.put_blob(container, "x.parquet", body);

    // Plan a DELETE of BOTH rows.
    let blob_table = vec![BlobIdEntry {
        name: "x.parquet".into(),
        chunk_base_row: 0,
        etag,
    }];
    let mut be = BlobEdits::default();
    be.deletes.insert(0);
    be.deletes.insert(1);
    let mut edits = HashMap::new();
    edits.insert(0u32, be);

    let plan = ModifyPlan {
        blob_table,
        edits,
        schema,
        pg_oids: vec![],
        update_attnums: vec![],
        client,
        compression: Compression::Snappy,
        is_delete: true,
        ctid_attno: 0, // unused: this test drives commit_plan directly
        edit_count: 0,
    };

    // (a) commit_plan must succeed.
    commit_plan(plan).expect("commit_plan");

    // (b) the original blob is gone, AND (c) no staging blob remains.
    // `list_blobs` returns EVERY blob in the container, so an empty list
    // proves both the original was DELETEd and no `*.tmp.*` staging blob
    // was left behind by the empty-result path.
    let remaining = fake.list_blobs(container, None);
    assert!(
        remaining.is_empty(),
        "empty-delete path must leave the container empty (no original, no staging), got: {remaining:?}"
    );
    assert!(
        !remaining.iter().any(|n| n.contains(".tmp.")),
        "no `*.tmp.*` staging blob may remain, got: {remaining:?}"
    );
}
