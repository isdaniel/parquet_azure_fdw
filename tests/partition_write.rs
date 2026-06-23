//! SP-3b Task 6: INSERT per-partition routing tests.
//!
//! Drives the per-group finalize+upload path against the in-process
//! `FakeBlobStore` (no Docker, no Azurite, no real Azure credentials)
//! through the test-only `finalize_and_upload_for_test` entry, which is the
//! same code the executor takes after `append_slot` has populated the
//! per-group builders/writers — minus the live PG slot decode that we can't
//! synthesize from a Rust integration test.

#![cfg(feature = "pg_test")]

#[path = "common/mod.rs"]
mod common;

use arrow::array::{ArrayRef, Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use parquet_azure_fdw::azure::{AzureBlobClient, Credential};
use parquet_azure_fdw::fdw::modify::finalize_and_upload_for_test;
use parquet_azure_fdw::fdw::options::PgPartitionType;
use parquet_azure_fdw::fdw::partition::PartitionTupleKey;
use parquet_azure_fdw::parquet_io::Compression;
use std::collections::HashMap;
use std::sync::Arc;

use common::fake_blob_store::FakeBlobStore;

fn make_client(fake: &FakeBlobStore, container: &str) -> AzureBlobClient {
    let sas = fake.sas_url(container);
    let cred = Credential::SasUrl { container_url: sas };
    AzureBlobClient::new("fake.invalid", "fakeaccount", cred, container)
        .expect("AzureBlobClient::new")
}

fn storage_schema() -> SchemaRef {
    // Storage-only schema: partition columns (year, region) are NOT in the
    // parquet, only the storage column `v`. Mirrors `build_state`'s schema
    // construction after partition cols are stripped.
    Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, true)]))
}

fn one_row_batch(v: i32) -> RecordBatch {
    let schema = storage_schema();
    RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vec![v])) as ArrayRef],
    )
    .expect("build single-row record batch")
}

fn n_row_batch(values: &[i32]) -> RecordBatch {
    let schema = storage_schema();
    RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(values.to_vec())) as ArrayRef],
    )
    .expect("build n-row record batch")
}

fn part_decls() -> Vec<(String, PgPartitionType)> {
    vec![
        ("year".into(), PgPartitionType::Int4),
        ("region".into(), PgPartitionType::Text),
    ]
}

/// INSERT 2 rows with (year=2026, region=us) + 1 row with (year=2026, region=eu)
/// must produce exactly two blobs, each under the expected partition path.
#[test]
fn multi_tuple_routing_writes_one_blob_per_distinct_key() {
    let fake = FakeBlobStore::start_blocking();
    let container = "c-part-multi";
    let client = make_client(&fake, container);

    let mut groups: HashMap<PartitionTupleKey, Vec<RecordBatch>> = HashMap::new();
    groups.insert(
        PartitionTupleKey {
            values: vec!["2026".into(), "us".into()],
        },
        vec![n_row_batch(&[10, 20])],
    );
    groups.insert(
        PartitionTupleKey {
            values: vec!["2026".into(), "eu".into()],
        },
        vec![one_row_batch(99)],
    );

    let uploaded = finalize_and_upload_for_test(
        storage_schema(),
        Compression::Snappy,
        client,
        "events".into(),
        None,
        part_decls(),
        groups,
    )
    .expect("finalize_and_upload_for_test");

    assert_eq!(uploaded.len(), 2, "expected two distinct uploads");
    let names: Vec<String> = uploaded.iter().map(|(_, n)| n.clone()).collect();
    assert!(
        names
            .iter()
            .any(|n| n.starts_with("events/year=2026/region=eu/")),
        "missing eu blob: {names:?}"
    );
    assert!(
        names
            .iter()
            .any(|n| n.starts_with("events/year=2026/region=us/")),
        "missing us blob: {names:?}"
    );

    // And the fake should physically have both blobs under the expected
    // prefixes.
    let eu = fake.list_blobs(container, Some("events/year=2026/region=eu/"));
    let us = fake.list_blobs(container, Some("events/year=2026/region=us/"));
    assert_eq!(eu.len(), 1, "eu prefix should hold one blob: {eu:?}");
    assert_eq!(us.len(), 1, "us prefix should hold one blob: {us:?}");
}

/// A group with zero rows must NOT produce a blob — only the non-empty
/// group does. This guards against the "create empty parquet for every
/// declared tuple key" failure mode the brief flagged.
#[test]
fn empty_group_produces_no_blob() {
    let fake = FakeBlobStore::start_blocking();
    let container = "c-part-empty";
    let client = make_client(&fake, container);

    let empty_schema = storage_schema();
    let empty_batch = RecordBatch::new_empty(empty_schema.clone());

    let mut groups: HashMap<PartitionTupleKey, Vec<RecordBatch>> = HashMap::new();
    // The "ca" group never had a row routed to it — represent that as a
    // zero-row RecordBatch list (parity with the production path, where a
    // never-touched group has no entry in the builders/writers maps and
    // therefore yields no upload).
    groups.insert(
        PartitionTupleKey {
            values: vec!["2026".into(), "ca".into()],
        },
        vec![empty_batch],
    );
    groups.insert(
        PartitionTupleKey {
            values: vec!["2026".into(), "us".into()],
        },
        vec![n_row_batch(&[1, 2, 3])],
    );

    let uploaded = finalize_and_upload_for_test(
        empty_schema,
        Compression::Snappy,
        client,
        "events".into(),
        None,
        part_decls(),
        groups,
    )
    .expect("finalize_and_upload_for_test");

    assert_eq!(uploaded.len(), 1, "only the non-empty group should upload");
    assert!(
        uploaded[0].1.starts_with("events/year=2026/region=us/"),
        "got {}",
        uploaded[0].1
    );

    let ca = fake.list_blobs(container, Some("events/year=2026/region=ca/"));
    let us = fake.list_blobs(container, Some("events/year=2026/region=us/"));
    assert!(ca.is_empty(), "empty group must not produce a blob: {ca:?}");
    assert_eq!(
        us.len(),
        1,
        "non-empty group must produce exactly 1: {us:?}"
    );
}
