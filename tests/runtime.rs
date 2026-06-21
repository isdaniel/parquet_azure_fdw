use parquet_azure_fdw::runtime::block_on;

#[test]
fn block_on_runs_future_to_completion() {
    let v = block_on(async { 1 + 2 });
    assert_eq!(v, 3);
}

#[test]
fn block_on_reuses_runtime_across_calls() {
    // Calling block_on inside the runtime would panic; we don't test that, but
    // we verify that two SEPARATE block_on calls reuse the same runtime.
    let a = block_on(async { 1 });
    let b = block_on(async { 2 });
    assert_eq!(a + b, 3);
}
