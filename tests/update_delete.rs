//! Integration tests for the UPDATE/DELETE rewrite path.
//!
//! These bypass Postgres entirely — they build a [`ModifyPlan`] in code,
//! seed blobs via the in-memory wiremock fake, call
//! [`parquet_azure_fdw::fdw::modify::update::commit_plan`], and assert the
//! resulting blob state. The goal is to exercise the full
//! `AzureBlobClient` + `apply_edits` + commit pipeline without Docker,
//! Azurite, or any real Azure credentials.
//!
//! Gated behind `pg_test` so the optional `wiremock` dependency is pulled in.

#![cfg(feature = "pg_test")]

#[path = "common/mod.rs"]
mod common;

use arrow::array::{Array, ArrayRef, Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use bytes::Bytes;
use parquet_azure_fdw::azure::{AzureBlobClient, Credential};
use parquet_azure_fdw::fdw::modify::kernel::{BlobEdits, RowOverride};
use parquet_azure_fdw::fdw::modify::update::{commit_plan, ModifyPlan};
use parquet_azure_fdw::fdw::modify::BlobIdEntry;
use parquet_azure_fdw::parquet_io::writer::{Compression, ParquetBatchWriter};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::runtime::Runtime;

use common::fake_blob_store::FakeBlobStore;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_parquet(values: &[i32], strs: &[&str]) -> Bytes {
    let schema = schema_two_cols();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(values.to_vec())) as ArrayRef,
            Arc::new(StringArray::from(strs.to_vec())) as ArrayRef,
        ],
    )
    .expect("build record batch fixture");
    let mut w =
        ParquetBatchWriter::new(schema, Compression::Snappy).expect("construct ParquetBatchWriter");
    w.write(&batch).expect("write fixture batch");
    w.finish().expect("finalise fixture parquet")
}

fn schema_two_cols() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int32, true),
        Field::new("s", DataType::Utf8, true),
    ]))
}

fn rt() -> Runtime {
    Runtime::new().expect("tokio runtime for read-back helper")
}

/// Build an FDW-facing client for `container` using the fake's SAS URL.
/// Mirrors how the production `build_credential` path would construct
/// things from the FDW user-mapping options.
fn make_client(fake: &FakeBlobStore, container: &str) -> AzureBlobClient {
    let sas = fake.sas_url(container);
    let cred = Credential::SasUrl { container_url: sas };
    AzureBlobClient::new(
        "fake.invalid", // endpoint string — unused for SasUrl
        "fakeaccount",  // account — unused for SasUrl
        cred,
        container,
    )
    .expect("AzureBlobClient::new")
}

fn read_back(body: Bytes) -> Vec<String> {
    use futures::StreamExt;
    use parquet_azure_fdw::parquet_io::reader::{open_stream_from_bytes, ParquetReadOptions};
    let mut out = Vec::new();
    rt().block_on(async {
        let mut s = open_stream_from_bytes(body, ParquetReadOptions::default())
            .await
            .expect("decode parquet bytes");
        while let Some(b) = s.next().await {
            let b = b.expect("batch ok");
            let arr = b
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("col 1 is utf8");
            for i in 0..arr.len() {
                out.push(arr.value(i).to_string());
            }
        }
    });
    out
}

fn read_back_int_col(body: Bytes) -> Vec<i32> {
    use futures::StreamExt;
    use parquet_azure_fdw::parquet_io::reader::{open_stream_from_bytes, ParquetReadOptions};
    let mut out = Vec::new();
    rt().block_on(async {
        let mut s = open_stream_from_bytes(body, ParquetReadOptions::default())
            .await
            .expect("decode parquet bytes");
        while let Some(b) = s.next().await {
            let b = b.expect("batch ok");
            let arr = b
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("col 0 is int32");
            for i in 0..arr.len() {
                out.push(arr.value(i));
            }
        }
    });
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn update_single_blob_single_row() {
    let fake = FakeBlobStore::start_blocking();
    let container = "c-update-1";
    let client = make_client(&fake, container);

    let body = make_parquet(&[10, 20, 30, 40], &["a", "b", "c", "d"]);
    let etag = fake.put_blob(container, "x.parquet", body);

    let blob_table = vec![BlobIdEntry {
        name: "x.parquet".into(),
        chunk_base_row: 0,
        etag,
    }];
    let mut edits = HashMap::new();
    let mut be = BlobEdits::default();
    be.updates.insert(
        1,
        RowOverride {
            values: vec![
                None,
                Some(Arc::new(StringArray::from(vec!["NEW"])) as ArrayRef),
            ],
        },
    );
    edits.insert(0u32, be);

    let plan = ModifyPlan {
        blob_table,
        edits,
        schema: schema_two_cols(),
        pg_oids: vec![],
        update_attnums: vec![1],
        client,
        compression: Compression::Snappy,
        is_delete: false,
        ctid_attno: 0, // unused: integration tests drive commit_plan directly
        edit_count: 0,
    };
    commit_plan(plan).expect("commit_plan");

    let body = fake
        .get_blob(container, "x.parquet")
        .expect("blob still present");
    let actual = read_back(body);
    assert_eq!(actual, vec!["a", "NEW", "c", "d"]);
}

#[test]
fn delete_all_rows_drops_blob() {
    let fake = FakeBlobStore::start_blocking();
    let container = "c-del-1";
    let client = make_client(&fake, container);

    let body = make_parquet(&[1, 2], &["a", "b"]);
    let etag = fake.put_blob(container, "x.parquet", body);

    let blob_table = vec![BlobIdEntry {
        name: "x.parquet".into(),
        chunk_base_row: 0,
        etag,
    }];
    let mut edits = HashMap::new();
    let mut be = BlobEdits::default();
    be.deletes.insert(0);
    be.deletes.insert(1);
    edits.insert(0u32, be);
    let plan = ModifyPlan {
        blob_table,
        edits,
        schema: schema_two_cols(),
        pg_oids: vec![],
        update_attnums: vec![],
        client,
        compression: Compression::Snappy,
        is_delete: true,
        ctid_attno: 0, // unused: integration tests drive commit_plan directly
        edit_count: 0,
    };
    commit_plan(plan).expect("commit_plan");

    let listed = fake.list_blobs(container, None);
    assert!(listed.is_empty(), "blob not removed: {listed:?}");
}

#[test]
fn concurrent_etag_conflict_returns_serialization_failure() {
    let fake = FakeBlobStore::start_blocking();
    let container = "c-conflict-1";
    let client = make_client(&fake, container);

    let body = make_parquet(&[1, 2], &["a", "b"]);
    fake.put_blob(container, "x.parquet", body.clone());

    // Read the etag, then mutate out-of-band so the etag we hold goes stale.
    let (_old_body, _old_etag) = rt()
        .block_on(async { client.get_with_etag("x.parquet").await })
        .expect("get_with_etag");
    fake.put_blob(container, "x.parquet", body); // overwrite — etag advances

    // Simulate a stale-etag rewrite by calling put_if_match directly with a
    // known-bad etag. `commit_plan` itself always fetches a fresh etag
    // first, so the cleanest way to exercise the 412 path is via the lower
    // primitive.
    let err = rt()
        .block_on(async {
            client
                .put_if_match("x.parquet", Bytes::from_static(b"x"), "\"obviously-wrong\"")
                .await
        })
        .expect_err("expected etag mismatch");
    match err {
        parquet_azure_fdw::error::FdwError::ConcurrentUpdate { reason, .. } => {
            assert!(reason.contains("etag"), "reason: {reason}");
        }
        other => panic!("expected ConcurrentUpdate, got {other:?}"),
    }
}

#[test]
fn blob_over_65k_rows_chunked_rewrite() {
    let fake = FakeBlobStore::start_blocking();
    let container = "c-chunk-1";
    let client = make_client(&fake, container);

    let ints: Vec<i32> = (0..70_000).collect();
    let strs: Vec<String> = (0..70_000).map(|i| format!("r{i}")).collect();
    let strs_ref: Vec<&str> = strs.iter().map(|s| s.as_str()).collect();
    let body = make_parquet(&ints, &strs_ref);
    let etag = fake.put_blob(container, "x.parquet", body);

    // Two 65_536-row chunks. The kernel works in absolute row indices
    // within the source blob, so the deletes target abs rows 0 and 65_536.
    let blob_table = vec![
        BlobIdEntry {
            name: "x.parquet".into(),
            chunk_base_row: 0,
            etag: etag.clone(),
        },
        BlobIdEntry {
            name: "x.parquet".into(),
            chunk_base_row: 65_536,
            etag,
        },
    ];
    let mut be = BlobEdits::default();
    be.deletes.insert(0);
    be.deletes.insert(65_536);
    let mut edits = HashMap::new();
    edits.insert(0u32, be);
    let plan = ModifyPlan {
        blob_table,
        edits,
        schema: schema_two_cols(),
        pg_oids: vec![],
        update_attnums: vec![],
        client,
        compression: Compression::Snappy,
        is_delete: true,
        ctid_attno: 0, // unused: integration tests drive commit_plan directly
        edit_count: 0,
    };
    commit_plan(plan).expect("commit_plan");

    let body = fake.get_blob(container, "x.parquet").expect("blob present");
    let ints_after = read_back_int_col(body);
    assert_eq!(ints_after.len(), 70_000 - 2);
    // Rows 0 and 65_536 should be gone; the remaining values are the
    // original integers minus those two.
    assert_eq!(ints_after[0], 1);
    assert!(!ints_after.contains(&0), "row 0 should be deleted");
    assert!(
        !ints_after.contains(&65_536),
        "row 65_536 should be deleted"
    );
}

#[test]
fn update_across_glob_touches_only_dirty_blobs() {
    let fake = FakeBlobStore::start_blocking();
    let container = "c-glob-1";
    let client = make_client(&fake, container);

    let mut etags = std::collections::HashMap::new();
    for name in ["a.parquet", "b.parquet", "c.parquet"] {
        let body = make_parquet(&[1, 2], &["x", "y"]);
        let e = fake.put_blob(container, name, body);
        etags.insert(name.to_string(), e);
    }

    let blob_table = vec![
        BlobIdEntry {
            name: "a.parquet".into(),
            chunk_base_row: 0,
            etag: etags["a.parquet"].clone(),
        },
        BlobIdEntry {
            name: "b.parquet".into(),
            chunk_base_row: 0,
            etag: etags["b.parquet"].clone(),
        },
        BlobIdEntry {
            name: "c.parquet".into(),
            chunk_base_row: 0,
            etag: etags["c.parquet"].clone(),
        },
    ];
    let pre_etag_b = rt()
        .block_on(async { client.get_with_etag("b.parquet").await })
        .expect("get b.parquet")
        .1;

    let mut edits = HashMap::new();
    let mut ea = BlobEdits::default();
    ea.deletes.insert(0);
    edits.insert(0u32, ea);
    let mut ec = BlobEdits::default();
    ec.deletes.insert(1);
    edits.insert(2u32, ec);

    let plan = ModifyPlan {
        blob_table,
        edits,
        schema: schema_two_cols(),
        pg_oids: vec![],
        update_attnums: vec![],
        client: client.clone(),
        compression: Compression::Snappy,
        is_delete: true,
        ctid_attno: 0, // unused: integration tests drive commit_plan directly
        edit_count: 0,
    };
    commit_plan(plan).expect("commit_plan");

    let post_etag_b = rt()
        .block_on(async { client.get_with_etag("b.parquet").await })
        .expect("get b.parquet post-commit")
        .1;
    assert_eq!(pre_etag_b, post_etag_b, "b.parquet must not be touched");
}

#[test]
fn stale_scan_etag_aborts_commit_with_serialization_failure() {
    // Realistic concurrency scenario: a SELECT captures etag E0; a
    // concurrent writer overwrites the blob, advancing the server's etag
    // to E1; the UPDATE statement then calls commit_plan with E0 stashed
    // in BlobIdEntry. The GET's If-Match must trip, surfacing
    // ConcurrentUpdate instead of silently writing stale ctids over the
    // new content.
    let fake = FakeBlobStore::start_blocking();
    let container = "c-stale-etag";
    let client = make_client(&fake, container);

    let body = make_parquet(&[1, 2], &["a", "b"]);
    let scan_etag = fake.put_blob(container, "x.parquet", body.clone());

    // Concurrent writer slips in between SELECT (etag=scan_etag) and UPDATE.
    fake.put_blob(container, "x.parquet", body);

    let blob_table = vec![BlobIdEntry {
        name: "x.parquet".into(),
        chunk_base_row: 0,
        etag: scan_etag, // captured at scan time, now stale
    }];
    let mut edits = HashMap::new();
    let mut be = BlobEdits::default();
    be.deletes.insert(0);
    edits.insert(0u32, be);
    let plan = ModifyPlan {
        blob_table,
        edits,
        schema: schema_two_cols(),
        pg_oids: vec![],
        update_attnums: vec![],
        client,
        compression: Compression::Snappy,
        is_delete: true,
        ctid_attno: 0,
        edit_count: 0,
    };
    let err = commit_plan(plan).expect_err("commit_plan must refuse stale etag");
    match err {
        parquet_azure_fdw::error::FdwError::ConcurrentUpdate { reason, .. } => {
            assert!(
                reason.contains("etag") || reason.contains("changed"),
                "reason: {reason}"
            );
        }
        other => panic!("expected ConcurrentUpdate, got {other:?}"),
    }
}

#[test]
fn fresh_scan_etag_commits_successfully() {
    // Mirror of the previous test: confirm the happy path still works when
    // no concurrent writer interferes between scan and commit.
    let fake = FakeBlobStore::start_blocking();
    let container = "c-fresh-etag";
    let client = make_client(&fake, container);

    let body = make_parquet(&[1, 2], &["a", "b"]);
    let etag = fake.put_blob(container, "x.parquet", body);

    let blob_table = vec![BlobIdEntry {
        name: "x.parquet".into(),
        chunk_base_row: 0,
        etag,
    }];
    let mut edits = HashMap::new();
    let mut be = BlobEdits::default();
    be.deletes.insert(0);
    edits.insert(0u32, be);
    let plan = ModifyPlan {
        blob_table,
        edits,
        schema: schema_two_cols(),
        pg_oids: vec![],
        update_attnums: vec![],
        client,
        compression: Compression::Snappy,
        is_delete: false,
        ctid_attno: 0,
        edit_count: 0,
    };
    commit_plan(plan).expect("happy-path commit must succeed");

    let body = fake.get_blob(container, "x.parquet").expect("blob present");
    let strs = read_back(body);
    assert_eq!(strs, vec!["b"]);
}

/// End-to-end coverage of the lost-update guard through the `scan_handoff`
/// layer — bridging the load-bearing invariant that
/// `tests/update_delete.rs::stale_scan_etag_aborts_commit_with_serialization_failure`
/// fences at the commit_plan level and that `src/fdw/modify/scan_handoff.rs`'s
/// unit tests fence at the primitive level. The SQL `#[pg_test]`
/// counterpart (`pg_update_serialization_failure`) is intentionally
/// `#[ignore]`'d because `pgrx test` is single-session — see its doc-comment
/// for details. This test fills that gap by:
///
///   1. seeding a blob,
///   2. publishing a `(name, etag)` handoff via `scan_handoff::publish`
///      EXACTLY as `begin_foreign_scan` does,
///   3. taking it back via `scan_handoff::take` EXACTLY as `build_plan`
///      does, building a `ModifyPlan` from the taken entry,
///   4. mutating the blob out-of-band so the published etag goes stale,
///   5. asserting `commit_plan` returns `FdwError::ConcurrentUpdate`
///      (which `error::raise` maps to SQLSTATE 40001).
#[test]
fn scan_handoff_to_commit_plan_lost_update_guard() {
    use parquet_azure_fdw::error::FdwError;
    use parquet_azure_fdw::fdw::modify::scan_handoff;
    use pgrx::pg_sys;

    let fake = FakeBlobStore::start_blocking();
    let container = "c-handoff-guard";
    let client = make_client(&fake, container);

    let body = make_parquet(&[1, 2], &["a", "b"]);
    let scan_etag = fake.put_blob(container, "h.parquet", body.clone());

    // (1) Scan publishes the (name, etag) it observed at LIST/HEAD time.
    // Use a synthetic relid — scan_handoff is keyed by `pg_sys::Oid` only.
    let relid = pg_sys::Oid::from(424242u32);
    scan_handoff::publish(relid, vec![("h.parquet".into(), scan_etag.clone())]);

    // (2) Modify side takes the handoff, identical to `build_plan`.
    let taken = scan_handoff::take(relid).expect("scan published");
    assert_eq!(taken.len(), 1);
    assert_eq!(taken[0].0, "h.parquet");
    assert_eq!(taken[0].1, scan_etag);

    // (3) Build the ModifyPlan from the taken entry — preserving the
    // captured etag exactly as build_plan does into BlobIdEntry.etag.
    let blob_table = vec![BlobIdEntry {
        name: taken[0].0.clone(),
        chunk_base_row: 0,
        etag: taken[0].1.clone(),
    }];
    let mut edits = HashMap::new();
    let mut be = BlobEdits::default();
    // Override row 0's string column with a known sentinel — this is the
    // change commit_plan WOULD persist if the guard failed.
    be.updates.insert(
        0,
        RowOverride {
            values: vec![
                None,
                Some(Arc::new(StringArray::from(vec!["should-never-land"])) as ArrayRef),
            ],
        },
    );
    edits.insert(0u32, be);
    let plan = ModifyPlan {
        blob_table,
        edits,
        schema: schema_two_cols(),
        pg_oids: vec![],
        update_attnums: vec![],
        client,
        compression: Compression::Snappy,
        is_delete: false,
        ctid_attno: 0,
        edit_count: 0,
    };

    // (4) Concurrent writer slips in between scan and commit. New body
    // (different content) so we can also verify it survives.
    let pre_concurrent_etag = fake
        .read_etag(container, "h.parquet")
        .expect("blob present pre");
    let concurrent_body = make_parquet(&[99], &["concurrent"]);
    let post_concurrent_etag = fake.put_blob(container, "h.parquet", concurrent_body.clone());
    assert_ne!(
        pre_concurrent_etag, post_concurrent_etag,
        "concurrent writer should advance the etag"
    );

    // (5) commit_plan must refuse to overwrite the concurrent writer's work.
    let err = commit_plan(plan)
        .expect_err("commit_plan must surface a ConcurrentUpdate when the etag is stale");
    match err {
        FdwError::ConcurrentUpdate { blob, reason } => {
            assert_eq!(blob, "h.parquet");
            assert!(
                reason.contains("etag") || reason.contains("changed"),
                "reason should mention the etag mismatch, got: {reason}"
            );
        }
        other => panic!("expected FdwError::ConcurrentUpdate, got: {other:?}"),
    }

    // And the blob in the fake is STILL the concurrent writer's content,
    // not our would-be overwrite. This is the actual invariant: a stale
    // scan must not silently clobber concurrent work.
    let surviving = fake.get_blob(container, "h.parquet").expect("blob present");
    assert_eq!(surviving.as_ref(), concurrent_body.as_ref());
    assert_eq!(
        fake.read_etag(container, "h.parquet"),
        Some(post_concurrent_etag),
        "etag must still be the concurrent writer's"
    );
}
