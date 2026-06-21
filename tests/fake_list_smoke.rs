//! Smoke test the FakeBlobStore's container-list path against the real FDW
//! list helper. Catches schema mismatches before they manifest as opaque
//! `#[pg_test]` SIGABRTs.

#![cfg(feature = "pg_test")]

use parquet_azure_fdw::azure::{AzureBlobClient, Credential};
use parquet_azure_fdw::test_harness::FakeBlobStore;

#[tokio::test]
async fn list_with_prefix_returns_seeded_blobs() {
    let fake = FakeBlobStore::start_blocking();
    let container = "c-list-1";
    fake.put_blob(
        container,
        "events/part-a.parquet",
        bytes::Bytes::from_static(b"a"),
    );
    fake.put_blob(
        container,
        "events/part-b.parquet",
        bytes::Bytes::from_static(b"b"),
    );
    fake.put_blob(container, "other.parquet", bytes::Bytes::from_static(b"c"));

    let client = AzureBlobClient::new(
        "fake.invalid",
        "fakeaccount",
        Credential::SasUrl {
            container_url: fake.sas_url(container),
        },
        container,
    )
    .expect("AzureBlobClient::new");

    let mut listed = client.list_with_prefix("events/").await.expect("list");
    listed.sort();
    assert_eq!(
        listed,
        vec![
            "events/part-a.parquet".to_string(),
            "events/part-b.parquet".to_string()
        ]
    );
}

/// Walk the same code path the FDW SELECT uses: list, then open each blob
/// and stream parquet rows. Reproduces the SIGABRT scenario from
/// `pg_glob_matches_multiple_blobs` outside a Postgres backend so the
/// failure has a real backtrace.
#[tokio::test]
async fn read_each_blob_from_listing() {
    use arrow::array::{Array, Int64Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use futures::StreamExt;
    use parquet_azure_fdw::parquet_io::reader::{open_stream, ParquetReadOptions};
    use std::sync::Arc;

    let fake = FakeBlobStore::start_blocking();
    let container = "c-list-read";

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    for (i, name) in ["events/a.parquet", "events/b.parquet", "events/c.parquet"]
        .iter()
        .enumerate()
    {
        let ids: Vec<i64> = (0..4).map(|j| (i as i64) * 4 + j).collect();
        let names: Vec<String> = ids.iter().map(|x| format!("n{x}")).collect();
        let id_arr: Arc<dyn Array> = Arc::new(Int64Array::from(ids));
        let name_arr: Arc<dyn Array> = Arc::new(StringArray::from(names));
        let batch = RecordBatch::try_new(schema.clone(), vec![id_arr, name_arr]).unwrap();
        let mut w = parquet_azure_fdw::parquet_io::ParquetBatchWriter::new(
            schema.clone(),
            parquet_azure_fdw::parquet_io::Compression::Snappy,
        )
        .unwrap();
        w.write(&batch).unwrap();
        let body = w.finish().unwrap();
        fake.put_blob(container, name, body);
    }

    let client = AzureBlobClient::new(
        "fake.invalid",
        "fakeaccount",
        Credential::SasUrl {
            container_url: fake.sas_url(container),
        },
        container,
    )
    .expect("client");

    let mut listed = client.list_with_prefix("events/").await.expect("list");
    listed.sort();
    assert_eq!(listed.len(), 3);

    let mut total = 0;
    for blob in &listed {
        let reader = client.open_blob(blob);
        let opts = ParquetReadOptions::default();
        let mut stream = open_stream(reader, opts).await.expect("open_stream");
        while let Some(batch) = stream.next().await {
            let batch = batch.expect("ok batch");
            total += batch.num_rows();
        }
    }
    assert_eq!(total, 12, "3 blobs × 4 rows");
}

/// Staging blobs (`*.tmp.<uuid>.parquet`) belong to in-flight or aborted
/// UPDATE/DELETE statements and must not be visible to external SELECTs.
#[tokio::test]
async fn list_filters_dot_tmp_dot_staging_blobs() {
    let fake = FakeBlobStore::start_blocking();
    let container = "c-list-staging";
    fake.put_blob(container, "a.parquet", bytes::Bytes::from_static(b"x"));
    fake.put_blob(
        container,
        "a.tmp.deadbeef.parquet",
        bytes::Bytes::from_static(b"y"),
    );
    fake.put_blob(container, "b.parquet", bytes::Bytes::from_static(b"z"));

    let client = AzureBlobClient::new(
        "fake.invalid",
        "fakeaccount",
        Credential::SasUrl {
            container_url: fake.sas_url(container),
        },
        container,
    )
    .expect("AzureBlobClient::new");

    let mut names: Vec<String> = client
        .list_with_prefix_etags("")
        .await
        .expect("list")
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    names.sort();
    assert_eq!(
        names,
        vec!["a.parquet".to_string(), "b.parquet".to_string()],
        "*.tmp.* staging must not appear in scan listing"
    );
}
