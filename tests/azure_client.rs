// Compile-only smoke test for the Azure facade. Confirms the type signatures
// resolve and the public surface is reachable. Live blob reads are exercised
// in the Azurite-backed Tier-2 suites later (Tasks 15+).

// Compile-only smoke test for the Azure facade. Confirms the type signatures
// resolve and the public surface is reachable. Live blob reads are exercised
// in the Azurite-backed Tier-2 suites later (Tasks 15+).

use parquet_azure_fdw::azure::writer::generate_blob_name;
use parquet_azure_fdw::azure::{AuthMethod, AzureBlobClient};

#[test]
fn types_compile() {
    fn _accepts(_: &AzureBlobClient) {}
    let _ = AuthMethod::ManagedIdentity;
}

#[test]
fn generated_name_has_prefix_and_extension() {
    let n = generate_blob_name("events/2024");
    assert!(n.starts_with("events/2024/"), "name was: {n}");
    assert!(n.ends_with(".parquet"), "name was: {n}");
}

#[test]
fn generated_name_handles_empty_prefix() {
    let n = generate_blob_name("");
    assert!(n.ends_with(".parquet"), "name was: {n}");
    assert!(!n.starts_with('/'), "name was: {n}");
}
