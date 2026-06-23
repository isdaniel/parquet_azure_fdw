//! Verify GlobSource against the fake blob store for mid-segment `*`
//! and `?`-containing patterns.

#![cfg(feature = "pg_test")]

#[path = "common/mod.rs"]
mod common;

use bytes::Bytes;
use common::fake_blob_store::FakeBlobStore;
use parquet_azure_fdw::azure::{AzureBlobClient, Credential};
use parquet_azure_fdw::fdw::glob::parse_glob;
use parquet_azure_fdw::fdw::scan::GlobSource;
use parquet_azure_fdw::fdw::scan::PlanSource;

fn make_client(fake: &FakeBlobStore, container: &str) -> AzureBlobClient {
    AzureBlobClient::new(
        "fake.invalid",
        "fakeaccount",
        Credential::SasUrl {
            container_url: fake.sas_url(container),
        },
        container,
    )
    .expect("AzureBlobClient::new")
}

#[test]
fn mid_segment_star() {
    let fake = FakeBlobStore::start_blocking();
    let container = "c-glob-mid";
    for path in &[
        "logs/2026/access.log",
        "logs/2027/access.log",
        "logs/2026/error.log",
        "other/file.log",
    ] {
        fake.put_blob(container, path, Bytes::from_static(b"x"));
    }
    let client = make_client(&fake, container);

    let glob = parse_glob("logs/*/access.log").unwrap();
    let src = GlobSource { glob };
    let mut names: Vec<String> = src
        .list(&client)
        .unwrap()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    names.sort();
    assert_eq!(
        names,
        vec![
            "logs/2026/access.log".to_string(),
            "logs/2027/access.log".to_string()
        ]
    );
}

#[test]
fn question_mark_single_char() {
    let fake = FakeBlobStore::start_blocking();
    let container = "c-glob-q";
    for path in &["v1/data.parquet", "v2/data.parquet", "v10/data.parquet"] {
        fake.put_blob(container, path, Bytes::from_static(b"x"));
    }
    let client = make_client(&fake, container);

    let glob = parse_glob("v?/data.parquet").unwrap();
    let src = GlobSource { glob };
    let mut names: Vec<String> = src
        .list(&client)
        .unwrap()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    names.sort();
    assert_eq!(
        names,
        vec!["v1/data.parquet".to_string(), "v2/data.parquet".to_string()]
    );
}
