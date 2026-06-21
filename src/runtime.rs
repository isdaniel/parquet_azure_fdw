#![forbid(unsafe_code)]
use std::cell::OnceCell;
use std::future::Future;
use tokio::runtime::{Builder, Runtime};

thread_local! {
    static RT: OnceCell<Runtime> = const { OnceCell::new() };
}

fn build_runtime() -> Runtime {
    Builder::new_current_thread()
        .enable_all()
        .thread_name("paf-bg")
        .build()
        .expect("failed to build tokio current-thread runtime")
}

/// Runs the future on this backend's lazily-initialized current-thread runtime.
/// Must NOT be called from inside another `block_on`.
pub fn block_on<F: Future>(fut: F) -> F::Output {
    RT.with(|cell| cell.get_or_init(build_runtime).block_on(fut))
}
