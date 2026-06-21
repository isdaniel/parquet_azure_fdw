//! Shared test infrastructure for the integration suites.
//!
//! Each `tests/<name>.rs` integration test is compiled as a standalone binary;
//! to reuse this module they include `#[path = "common/mod.rs"] mod common;`.
//!
//! Storage in tests is provided by the in-crate
//! [`parquet_azure_fdw::test_harness::fake_blob_store`] (gated on
//! `feature = "pg_test"`). This module re-exports the fake under a stable
//! `common::fake_blob_store` path so test bodies stay short.

// The harness exposes helpers that are not consumed by every test binary;
// silence dead-code warnings across the board.
#![allow(dead_code)]

#[cfg(feature = "pg_test")]
pub mod fake_blob_store {
    pub use parquet_azure_fdw::test_harness::fake_blob_store::FakeBlobStore;
}
