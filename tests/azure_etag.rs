//! Wiremock tests for the Azure etag wrappers. We stand up a local HTTP
//! mock, build an `AzureBlobClient` pointing at it via a fake SAS URL, and
//! verify request shape + error mapping.

use parquet_azure_fdw::azure::{AzureBlobClient, Credential};
use parquet_azure_fdw::error::FdwError;
use wiremock::matchers::{header, header_exists, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn client_for(server: &MockServer) -> AzureBlobClient {
    // SAS URL points at the mock server. Container is "c".
    let url = format!("{}/c?sv=fake", server.uri());
    AzureBlobClient::new(
        "blob.core.windows.net", // unused for SAS
        "acct",
        Credential::SasUrl { container_url: url },
        "c",
    )
    .expect("client")
}

#[tokio::test]
async fn get_with_etag_extracts_etag_header() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/c/a.parquet"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("ETag", "\"etag-1\"")
                .set_body_bytes(b"hello".as_ref()),
        )
        .expect(1)
        .mount(&server)
        .await;
    let c = client_for(&server).await;
    let (body, etag) = c.get_with_etag("a.parquet").await.expect("get");
    assert_eq!(&body[..], b"hello");
    assert_eq!(etag, "\"etag-1\"");
}

#[tokio::test]
async fn put_if_match_sends_if_match_header() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .and(path("/c/a.parquet"))
        .and(header("If-Match", "\"etag-1\""))
        .and(header_exists("Content-Length"))
        .respond_with(ResponseTemplate::new(201).insert_header("ETag", "\"etag-2\""))
        .expect(1)
        .mount(&server)
        .await;
    let c = client_for(&server).await;
    let new_etag = c
        .put_if_match(
            "a.parquet",
            bytes::Bytes::from_static(b"data"),
            "\"etag-1\"",
        )
        .await
        .expect("put");
    assert_eq!(new_etag, "\"etag-2\"");
}

#[tokio::test]
async fn put_if_match_returns_concurrent_update_on_412() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .and(path("/c/a.parquet"))
        .respond_with(ResponseTemplate::new(412))
        .mount(&server)
        .await;
    let c = client_for(&server).await;
    let err = c
        .put_if_match(
            "a.parquet",
            bytes::Bytes::from_static(b"data"),
            "\"etag-1\"",
        )
        .await
        .expect_err("412");
    match err {
        FdwError::ConcurrentUpdate { blob, reason } => {
            assert_eq!(blob, "a.parquet");
            assert!(reason.contains("etag mismatch"), "reason: {reason}");
        }
        other => panic!("expected ConcurrentUpdate, got {other:?}"),
    }
}

#[tokio::test]
async fn put_if_none_match_creates_new_blob_and_returns_etag() {
    let server = MockServer::start().await;
    // First PUT succeeds with 201 and an etag.
    Mock::given(method("PUT"))
        .and(path("/c/a.parquet"))
        .and(header("If-None-Match", "*"))
        .respond_with(ResponseTemplate::new(201).insert_header("ETag", "\"etag-new\""))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    // Subsequent PUT to same name → 412 (blob exists).
    Mock::given(method("PUT"))
        .and(path("/c/a.parquet"))
        .and(header("If-None-Match", "*"))
        .respond_with(ResponseTemplate::new(412))
        .mount(&server)
        .await;
    let c = client_for(&server).await;
    let etag = c
        .put_if_none_match("a.parquet", bytes::Bytes::from_static(b"abc"))
        .await
        .expect("create");
    assert!(!etag.is_empty(), "etag should be populated");
    let err = c
        .put_if_none_match("a.parquet", bytes::Bytes::from_static(b"xyz"))
        .await
        .expect_err("second create");
    match err {
        FdwError::ConcurrentUpdate { blob, reason } => {
            assert_eq!(blob, "a.parquet");
            assert!(
                reason.contains("staging-name collision"),
                "reason: {reason}"
            );
        }
        other => panic!("expected ConcurrentUpdate, got {other:?}"),
    }
}

#[tokio::test]
async fn delete_unconditional_swallows_404() {
    let server = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/c/nope.parquet"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;
    let c = client_for(&server).await;
    c.delete_unconditional("nope.parquet")
        .await
        .expect("404 swallowed");
}

#[tokio::test]
async fn delete_if_match_handles_404() {
    let server = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/c/a.parquet"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;
    let c = client_for(&server).await;
    let err = c
        .delete_if_match("a.parquet", "\"etag-1\"")
        .await
        .expect_err("404");
    match err {
        FdwError::ConcurrentUpdate { reason, .. } => {
            assert!(reason.contains("disappeared"), "reason: {reason}");
        }
        other => panic!("expected ConcurrentUpdate, got {other:?}"),
    }
}
