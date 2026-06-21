#![forbid(unsafe_code)]
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FdwError {
    #[error("invalid option: {0}")]
    InvalidOption(String),
    #[error("missing required option: {0}")]
    MissingOption(&'static str),
    #[error("azure storage error: {0}")]
    Azure(String),
    #[error("parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("unsupported type mapping: pg {pg_type} ↔ arrow {arrow_type}")]
    UnsupportedType { pg_type: String, arrow_type: String },
    #[error("schema mismatch: {0}")]
    SchemaMismatch(String),
    #[error("concurrent update on blob {blob}: {reason}")]
    ConcurrentUpdate { blob: String, reason: String },
}

impl FdwError {
    /// Construct an `Azure` variant from any displayable error, redacting any SAS signatures, AAD bearer tokens, or other secret-bearing tokens that the Azure SDK may have embedded in its message text.
    pub fn azure<E: std::fmt::Display>(e: E) -> Self {
        FdwError::Azure(crate::redact::redact(&e.to_string()))
    }

    /// Like [`azure`](Self::azure) but with an extra context string. Both the
    /// context and the inner message are redacted.
    pub fn azure_ctx<E: std::fmt::Display>(ctx: &str, e: E) -> Self {
        FdwError::Azure(crate::redact::redact(&format!("{ctx}: {e}")))
    }
}

pub type FdwResult<T> = Result<T, FdwError>;

/// Convert any `FdwError` into an `ereport!(ERROR)`. Diverges.
pub fn raise(e: FdwError) -> ! {
    // Defense in depth: redact at the boundary too, in case any caller built raw `Azure(String)` (or another variant) with un-redacted SDK output.
    let msg = crate::redact::redact(&e.to_string());
    if let FdwError::ConcurrentUpdate { .. } = &e {
        // SQLSTATE 40001 — serialization_failure. `ereport!(ERROR, ...)`
        // expands to an `unreachable!()` after `report()`, so it never returns.
        pgrx::ereport!(
            ERROR,
            pgrx::PgSqlErrorCode::ERRCODE_T_R_SERIALIZATION_FAILURE,
            msg,
        );
    }
    pgrx::error!("{msg}");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn concurrent_update_display() {
        let e = FdwError::ConcurrentUpdate {
            blob: "dir/a.parquet".into(),
            reason: "etag mismatch".into(),
        };
        assert_eq!(
            e.to_string(),
            "concurrent update on blob dir/a.parquet: etag mismatch"
        );
    }
}
