#![cfg(feature = "pg_test")]
//! Soundness: for randomly generated quals, the SELECT result must be
//! identical whether pushdown is on or off. Pushdown is a perf
//! optimization — output must not differ.
//!
//! Gated `#[ignore]` by default — slow. Run with:
//!   cargo pgrx test pg14 -- --ignored pushdown_soundness
//!
//! Skeleton only — full implementation (~80 lines) will use the existing
//! `fake_blob_store` harness and toggle `enable_pushdown` via ALTER SERVER
//! between the two SELECTs, asserting row sets identical via EXCEPT.

/// Fixed handwritten seed set — no PRNG dependency. Each tuple is
/// (description, where_clause). When the full harness lands these become
/// the deterministic corpus driven through both pushdown on/off SELECTs.
const SEED_CASES: &[(&str, &str)] = &[
    ("eq_int", "id = 100"),
    ("ne_int", "id <> 100"),
    ("lt_int", "id < 50"),
    ("ge_int", "id >= 200"),
    ("is_null", "name IS NULL"),
    ("is_not_null", "name IS NOT NULL"),
    ("and_combo", "id > 10 AND id < 90"),
    ("or_combo", "id = 1 OR id = 999"),
    ("in_list", "id IN (1, 2, 3, 5, 8, 13)"),
    ("eq_text", "name = 'alice'"),
];

#[ignore]
#[test]
fn pushdown_soundness_random_quals() {
    // Sanity: corpus is wired up and the test binary links.
    // Full impl will:
    //   1. fake_blob_store::start() + seed a parquet blob
    //   2. CREATE SERVER / FOREIGN TABLE pointed at the fake
    //   3. for each seed: run SELECT with enable_pushdown=true, then
    //      ALTER SERVER ... OPTIONS (SET enable_pushdown 'false'), rerun,
    //      assert (a EXCEPT b) UNION (b EXCEPT a) is empty.
    assert!(!SEED_CASES.is_empty(), "corpus must be non-empty");
    for (name, clause) in SEED_CASES {
        assert!(!name.is_empty(), "seed case must be named");
        assert!(!clause.is_empty(), "seed case {name} must have a clause");
    }
}
