//! Integration: IMPORT FOREIGN SCHEMA end-to-end via the in-crate helpers.
//!
//! Avoids spinning up Postgres — drives `group_blobs_by_directory`,
//! `infer_columns` (against the fake), and `build_create_table_ddl`
//! directly.

#![cfg(feature = "pg_test")]

#[path = "common/mod.rs"]
mod common;

use arrow::array::{Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use bytes::Bytes;
use common::fake_blob_store::FakeBlobStore;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use parquet_azure_fdw::azure::{AzureBlobClient, Credential};
use parquet_azure_fdw::fdw::import_schema::{build_create_table_ddl, group_blobs_by_directory};
use std::sync::Arc;

fn make_users_parquet() -> Bytes {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let mut buf: Vec<u8> = Vec::new();
    let mut w = ArrowWriter::try_new(
        &mut buf,
        schema.clone(),
        Some(WriterProperties::builder().build()),
    )
    .unwrap();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(StringArray::from(vec!["alice"])),
        ],
    )
    .unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    Bytes::from(buf)
}

#[test]
fn groups_users_orders_correctly() {
    let blobs = vec![
        "users/2026.parquet".to_string(),
        "users/2027.parquet".to_string(),
        "orders/2026.parquet".to_string(),
    ];
    let groups = group_blobs_by_directory(&blobs, "data");
    assert!(groups.contains_key("users"));
    assert!(groups.contains_key("orders"));
    assert_eq!(groups["users"].len(), 2);
    assert_eq!(groups["orders"].len(), 1);
}

#[test]
fn ddl_round_trip_preserves_column_order_for_i2_invariant() {
    // SP-1 I2: foreign-table column order must match parquet column order
    // so PushedExprFilter's `col` indexing stays valid.
    let cols = vec![
        ("id".to_string(), "INTEGER".to_string()),
        ("name".to_string(), "TEXT".to_string()),
    ];
    let ddl = build_create_table_ddl("public", "users", "srv", "data", "users/*.parquet", &cols);
    let id_pos = ddl.find(r#""id" INTEGER"#).unwrap();
    let name_pos = ddl.find(r#""name" TEXT"#).unwrap();
    assert!(id_pos < name_pos, "DDL must preserve parquet column order");
}

#[test]
fn infer_columns_against_fake_blob_store() {
    use parquet_azure_fdw::fdw::import_schema as is_mod;
    let fake = FakeBlobStore::start_blocking();
    let container = "c";
    let client = AzureBlobClient::new(
        "fake.invalid",
        "fakeaccount",
        Credential::SasUrl {
            container_url: fake.sas_url(container),
        },
        container,
    )
    .unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(client.put_if_none_match("users/2026.parquet", make_users_parquet()))
        .unwrap();
    drop(rt);

    // Drive infer_columns via the public entry point — same as the IMPORT
    // FOREIGN SCHEMA callback.
    let cols = is_mod::infer_columns(&client, "users/2026.parquet").unwrap();
    assert_eq!(
        cols,
        vec![
            ("id".to_string(), "INTEGER".to_string()),
            ("name".to_string(), "TEXT".to_string()),
        ]
    );

    let _ = is_mod::group_blobs_by_directory; // ensure the helper is exposed
}
