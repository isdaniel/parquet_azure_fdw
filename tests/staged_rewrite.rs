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
