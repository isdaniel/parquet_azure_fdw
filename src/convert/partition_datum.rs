#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]
//! Canonicalize a PG datum into the string form used for Hive partition
//! segments (path encoding + per-tuple grouping key).
//!
//! This is the inverse of `arrow_to_pg::parse_text_to_datum`: text in →
//! datum out (read side); datum in → text out (write side, here).
//!
//! Lives in its own file because [`pg_to_arrow`] carries
//! `#![forbid(unsafe_code)]` and the text/varchar arm needs to walk a
//! varlena via the same detoast pattern the INSERT slot decoder uses.

use crate::error::{FdwError, FdwResult};
use pgrx::pg_sys;
use std::ffi::CStr;

/// Convert a non-null Postgres datum of type `oid` into its canonical
/// partition-segment string. Caller must have established that `is_null` is
/// false for this datum.
///
/// # Safety
///
/// - `datum` is a valid datum of type `oid` produced by the executor for a
///   live tuple slot column.
/// - For varlena types (TEXT/VARCHAR), the datum pointer must be a live
///   `varlena*` owned by the surrounding memory context. The fn detoasts via
///   the documented `pg_detoast_datum` / `text_to_cstring` pattern, identical
///   to `insert.rs::text_datum_to_str`.
pub unsafe fn datum_to_partition_string(
    datum: pg_sys::Datum,
    oid: pg_sys::Oid,
) -> FdwResult<String> {
    // Primitive integer/date cases: pgrx exposes `Datum::value()` as a safe
    // accessor returning `usize` (the underlying machine word). The bit-cast
    // to the signed primitive width matches the static-inline `DatumGet*`
    // helpers PG inlines on every supported version.
    match oid {
        x if x == pg_sys::INT2OID => {
            let v = datum.value() as i16;
            if v < 0 {
                return Err(FdwError::SchemaMismatch(
                    "negative partition value not supported in v1".into(),
                ));
            }
            Ok(v.to_string())
        }
        x if x == pg_sys::INT4OID => {
            let v = datum.value() as i32;
            if v < 0 {
                return Err(FdwError::SchemaMismatch(
                    "negative partition value not supported in v1".into(),
                ));
            }
            Ok(v.to_string())
        }
        x if x == pg_sys::INT8OID => {
            let v = datum.value() as i64;
            if v < 0 {
                return Err(FdwError::SchemaMismatch(
                    "negative partition value not supported in v1".into(),
                ));
            }
            Ok(v.to_string())
        }
        x if x == pg_sys::DATEOID => {
            // PG DateADT is i32 days since 2000-01-01 (PG epoch). Render as
            // YYYY-MM-DD so the path round-trips with
            // `parse_text_to_datum(PgPartitionType::Date, "...")`.
            let pg_days = datum.value() as i32;
            let pg_epoch = chrono::NaiveDate::from_ymd_opt(2000, 1, 1)
                .ok_or_else(|| FdwError::SchemaMismatch("pg epoch construction failed".into()))?;
            let d = pg_epoch
                .checked_add_signed(chrono::Duration::days(pg_days as i64))
                .ok_or_else(|| {
                    FdwError::SchemaMismatch(format!(
                        "partition date out of range: pg-days={pg_days}"
                    ))
                })?;
            Ok(d.format("%Y-%m-%d").to_string())
        }
        x if x == pg_sys::TEXTOID || x == pg_sys::VARCHAROID => {
            // SAFETY: matches `insert::text_datum_to_str` — detoast and
            // convert via `text_to_cstring`; the datum's pointer payload is a
            // live `varlena*` (caller invariant). We free the palloc'd copy
            // (and the detoast result when it differs from the input pointer)
            // to keep peak memory bounded for long INSERTs.
            let owned = unsafe {
                let detoasted = pg_sys::pg_detoast_datum(datum.cast_mut_ptr::<pg_sys::varlena>());
                let cstr_ptr = pg_sys::text_to_cstring(detoasted as *const pg_sys::text);
                let s = CStr::from_ptr(cstr_ptr).to_string_lossy().into_owned();
                pg_sys::pfree(cstr_ptr.cast());
                if !std::ptr::eq(detoasted, datum.cast_mut_ptr::<pg_sys::varlena>()) {
                    pg_sys::pfree(detoasted.cast());
                }
                s
            };
            // Reject path-breaking characters up front. The Hive convention
            // is `key=value/`, so `/` and `=` in a value collide with the
            // segment grammar; an empty string would produce `key=/` which
            // the read-side parser rejects.
            if owned.is_empty() {
                return Err(FdwError::SchemaMismatch(
                    "partition text value cannot be empty".into(),
                ));
            }
            if owned.contains('/') || owned.contains('=') {
                return Err(FdwError::SchemaMismatch(format!(
                    "partition text value '{owned}' contains '/' or '='; \
                     these characters collide with the Hive path grammar"
                )));
            }
            if owned.contains("..") {
                return Err(FdwError::SchemaMismatch(format!(
                    "text partition value '{owned}' must not contain '..' (path-traversal symmetry with read-side SSRF validator)"
                )));
            }
            Ok(owned)
        }
        other => Err(FdwError::UnsupportedType {
            pg_type: format!("oid={} (partition string)", other.to_u32()),
            arrow_type: "<n/a>".to_string(),
        }),
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use super::*;
    use pgrx::{pg_test, IntoDatum};

    #[pg_test]
    fn text_value_rejects_dotdot() {
        let datum_text = "../etc/passwd".into_datum().unwrap();
        let err = unsafe { datum_to_partition_string(datum_text, pg_sys::TEXTOID) }
            .expect_err("'..' must reject");
        assert!(format!("{err}").contains(".."));
    }
}
