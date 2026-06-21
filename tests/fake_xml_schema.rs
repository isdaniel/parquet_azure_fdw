//! Smoke test: parse our fake's list-blobs XML through the same deserializer
//! the FDW uses, so we catch schema mismatches at build time rather than
//! mid-pg_test.

#![cfg(feature = "pg_test")]

#[test]
fn fake_list_blobs_xml_round_trips_through_sdk() {
    // Exactly what `FakeBlobStore::respond_list` writes.
    let xml = r#"<?xml version="1.0" encoding="utf-8"?><EnumerationResults ServiceEndpoint="http://127.0.0.1/" ContainerName="c"><Prefix>events/</Prefix><Blobs><Blob><Name>events/part-a.parquet</Name></Blob><Blob><Name>events/part-b.parquet</Name></Blob></Blobs><NextMarker/></EnumerationResults>"#;

    // Same path the SDK uses (azure_core::xml -> typespec_client_core::xml).
    let parsed: azure_storage_blob::models::ListBlobsResponse =
        azure_core::xml::from_xml(xml).expect("deserialize ListBlobsResponse");

    let names: Vec<String> = parsed
        .blob_items
        .iter()
        .filter_map(|b| b.name.clone())
        .collect();
    assert_eq!(
        names,
        vec![
            "events/part-a.parquet".to_string(),
            "events/part-b.parquet".to_string()
        ]
    );
}
