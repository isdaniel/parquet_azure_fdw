#![forbid(unsafe_code)]
//! Arrow `Array` value → Postgres `Datum` conversion for the type matrix
//! defined in design spec §7.4.
//!
//! This module is intentionally narrow: it converts a single Arrow value
//! (identified by an array reference and a row index) into an `Option<Datum>`,
//! where `None` represents SQL `NULL`. The slot-filling adapter that walks a
//! `RecordBatch` and writes into a `TupleTableSlot` lives in `fdw::scan`
//! (Task 12) and is built on top of this primitive.
//!
//! All conversions go through pgrx's safe `IntoDatum` trait so this module
//! can keep `#![forbid(unsafe_code)]`.

use crate::error::{FdwError, FdwResult};
use arrow::array::{
    Array, BinaryArray, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array,
    Int16Array, Int32Array, Int64Array, LargeBinaryArray, LargeStringArray, StringArray,
    TimestampMicrosecondArray,
};
use arrow::datatypes::{DataType, TimeUnit};
use pgrx::{pg_sys, AnyNumeric, IntoDatum, JsonB};

/// Days between the UNIX epoch (1970-01-01) and the Postgres epoch
/// (2000-01-01). Arrow `Date32` is days since UNIX epoch; Postgres `DATE`
/// is days since the Postgres epoch.
pub(crate) const UNIX_TO_PG_EPOCH_DAYS: i32 = 10957;

/// Microseconds between the UNIX epoch (1970-01-01) and the Postgres epoch
/// (2000-01-01). Arrow `Timestamp(Microsecond, _)` is microseconds since
/// UNIX epoch; pgrx's `Timestamp::try_from(i64)` takes microseconds since
/// the Postgres epoch.
pub(crate) const UNIX_TO_PG_EPOCH_MICROS: i64 = 946_684_800_000_000;

/// Convert a single Arrow array value into a Postgres `Datum` matching `pg_oid`.
///
/// Returns `Ok(None)` if the source value is SQL `NULL`. Returns
/// `Err(FdwError::UnsupportedType)` for any `(Arrow DataType, pg OID)` pair
/// that is not in the §7.4 matrix.
pub fn arrow_value_to_datum(
    arr: &dyn Array,
    row: usize,
    pg_oid: pg_sys::Oid,
) -> FdwResult<Option<pg_sys::Datum>> {
    if arr.is_null(row) {
        return Ok(None);
    }

    match (arr.data_type(), pg_oid) {
        // bool
        (DataType::Boolean, oid) if oid == pg_sys::BOOLOID => {
            let v = downcast::<BooleanArray>(arr)?.value(row);
            Ok(v.into_datum())
        }

        // int2 / int4 / int8
        (DataType::Int16, oid) if oid == pg_sys::INT2OID => {
            Ok(downcast::<Int16Array>(arr)?.value(row).into_datum())
        }
        (DataType::Int32, oid) if oid == pg_sys::INT4OID => {
            Ok(downcast::<Int32Array>(arr)?.value(row).into_datum())
        }
        (DataType::Int64, oid) if oid == pg_sys::INT8OID => {
            Ok(downcast::<Int64Array>(arr)?.value(row).into_datum())
        }

        // float4 / float8
        (DataType::Float32, oid) if oid == pg_sys::FLOAT4OID => {
            Ok(downcast::<Float32Array>(arr)?.value(row).into_datum())
        }
        (DataType::Float64, oid) if oid == pg_sys::FLOAT8OID => {
            Ok(downcast::<Float64Array>(arr)?.value(row).into_datum())
        }

        // text / varchar from Utf8
        (DataType::Utf8, oid) if oid == pg_sys::TEXTOID || oid == pg_sys::VARCHAROID => {
            let s = downcast::<StringArray>(arr)?.value(row);
            Ok(s.into_datum())
        }
        // text / varchar from LargeUtf8
        (DataType::LargeUtf8, oid) if oid == pg_sys::TEXTOID || oid == pg_sys::VARCHAROID => {
            let s = downcast::<LargeStringArray>(arr)?.value(row);
            Ok(s.into_datum())
        }

        // jsonb from Utf8 (serialized text → jsonb_in)
        (DataType::Utf8, oid) if oid == pg_sys::JSONBOID => {
            let s = downcast::<StringArray>(arr)?.value(row);
            json_value_to_datum(s)
        }
        (DataType::LargeUtf8, oid) if oid == pg_sys::JSONBOID => {
            let s = downcast::<LargeStringArray>(arr)?.value(row);
            json_value_to_datum(s)
        }

        // bytea
        (DataType::Binary, oid) if oid == pg_sys::BYTEAOID => {
            let b = downcast::<BinaryArray>(arr)?.value(row).to_vec();
            Ok(b.into_datum())
        }
        (DataType::LargeBinary, oid) if oid == pg_sys::BYTEAOID => {
            let b = downcast::<LargeBinaryArray>(arr)?.value(row).to_vec();
            Ok(b.into_datum())
        }

        // date
        (DataType::Date32, oid) if oid == pg_sys::DATEOID => {
            let unix_days = downcast::<Date32Array>(arr)?.value(row);
            let pg_days = unix_days
                .checked_sub(UNIX_TO_PG_EPOCH_DAYS)
                .ok_or_else(|| date_out_of_range(unix_days))?;
            let d = pgrx::datum::Date::try_from(pg_days as pg_sys::DateADT)
                .map_err(|e| date_conv_err(format!("{e:?}")))?;
            Ok(d.into_datum())
        }

        // timestamp (no tz)
        (DataType::Timestamp(TimeUnit::Microsecond, None), oid) if oid == pg_sys::TIMESTAMPOID => {
            let unix_us = downcast::<TimestampMicrosecondArray>(arr)?.value(row);
            let pg_us = unix_us
                .checked_sub(UNIX_TO_PG_EPOCH_MICROS)
                .ok_or_else(|| ts_out_of_range(unix_us))?;
            let ts = pgrx::datum::Timestamp::try_from(pg_us as pg_sys::Timestamp)
                .map_err(|raw| date_conv_err(format!("timestamp out of range: {raw}")))?;
            Ok(ts.into_datum())
        }

        // timestamptz
        (DataType::Timestamp(TimeUnit::Microsecond, Some(_)), oid)
            if oid == pg_sys::TIMESTAMPTZOID =>
        {
            let unix_us = downcast::<TimestampMicrosecondArray>(arr)?.value(row);
            let pg_us = unix_us
                .checked_sub(UNIX_TO_PG_EPOCH_MICROS)
                .ok_or_else(|| ts_out_of_range(unix_us))?;
            let ts = pgrx::datum::TimestampWithTimeZone::try_from(pg_us as pg_sys::TimestampTz)
                .map_err(|e| date_conv_err(format!("timestamptz out of range: {e:?}")))?;
            Ok(ts.into_datum())
        }

        // numeric from Decimal128
        (DataType::Decimal128(_p, scale), oid) if oid == pg_sys::NUMERICOID => {
            let arr = downcast::<Decimal128Array>(arr)?;
            let v = arr.value(row);
            let s = format_decimal128(v, *scale);
            let n = AnyNumeric::try_from(s.as_str())
                .map_err(|e| FdwError::SchemaMismatch(format!("invalid numeric {s:?}: {e:?}")))?;
            Ok(n.into_datum())
        }

        (a, _) => Err(FdwError::UnsupportedType {
            pg_type: format!("oid={}", pg_oid.to_u32()),
            arrow_type: format!("{a:?}"),
        }),
    }
}

/// Map an Arrow `DataType` to a PG type name suitable for `CREATE FOREIGN
/// TABLE` DDL emission (SP-2 IMPORT FOREIGN SCHEMA).
///
/// Returns an owned `String` uniformly (small allocation; emitted once per
/// column at IMPORT time) so callers don't have to special-case parameterised
/// types like `NUMERIC(p, s)`.
///
/// Supported types match `arrow_value_to_datum`'s read path so a generated
/// foreign table can read what its inferred schema describes.
pub fn arrow_type_to_pg_typename(
    ty: &arrow::datatypes::DataType,
) -> crate::error::FdwResult<String> {
    use arrow::datatypes::{DataType, TimeUnit};
    Ok(match ty {
        DataType::Boolean => "BOOLEAN".to_string(),
        DataType::Int16 => "SMALLINT".to_string(),
        DataType::Int32 => "INTEGER".to_string(),
        DataType::Int64 => "BIGINT".to_string(),
        DataType::Float32 => "REAL".to_string(),
        DataType::Float64 => "DOUBLE PRECISION".to_string(),
        DataType::Utf8 | DataType::LargeUtf8 => "TEXT".to_string(),
        DataType::Binary | DataType::LargeBinary => "BYTEA".to_string(),
        DataType::Date32 => "DATE".to_string(),
        DataType::Timestamp(TimeUnit::Microsecond, None) => "TIMESTAMP".to_string(),
        DataType::Timestamp(TimeUnit::Microsecond, Some(_)) => "TIMESTAMPTZ".to_string(),
        DataType::Decimal128(p, s) => format!("NUMERIC({p},{s})"),
        other => {
            return Err(crate::error::FdwError::SchemaMismatch(format!(
                "arrow type {other:?} has no SP-2 DDL mapping"
            )))
        }
    })
}

/// Parse a raw partition-key string into a PG datum of the declared type.
///
/// Used by SP-3b's Hive partition support: one cast per blob (not per row).
/// Returns `FdwError::SchemaMismatch` on cast failure so the caller can
/// skip the blob with a NOTICE.
pub fn parse_text_to_datum(
    ty: crate::fdw::options::PgPartitionType,
    raw: &str,
) -> FdwResult<pg_sys::Datum> {
    use crate::fdw::options::PgPartitionType;

    match ty {
        PgPartitionType::Int2 => {
            let v: i16 = raw
                .parse()
                .map_err(|_| FdwError::SchemaMismatch(format!("'{raw}' is not int2")))?;
            if v < 0 {
                return Err(FdwError::SchemaMismatch(format!(
                    "negative partition value '{raw}' not supported in v1"
                )));
            }
            v.into_datum()
                .ok_or_else(|| FdwError::SchemaMismatch("int2 into_datum returned None".into()))
        }
        PgPartitionType::Int4 => {
            let v: i32 = raw
                .parse()
                .map_err(|_| FdwError::SchemaMismatch(format!("'{raw}' is not int4")))?;
            if v < 0 {
                return Err(FdwError::SchemaMismatch(format!(
                    "negative partition value '{raw}' not supported in v1"
                )));
            }
            v.into_datum()
                .ok_or_else(|| FdwError::SchemaMismatch("int4 into_datum returned None".into()))
        }
        PgPartitionType::Int8 => {
            let v: i64 = raw
                .parse()
                .map_err(|_| FdwError::SchemaMismatch(format!("'{raw}' is not int8")))?;
            if v < 0 {
                return Err(FdwError::SchemaMismatch(format!(
                    "negative partition value '{raw}' not supported in v1"
                )));
            }
            v.into_datum()
                .ok_or_else(|| FdwError::SchemaMismatch("int8 into_datum returned None".into()))
        }
        PgPartitionType::Text => raw
            .to_string()
            .into_datum()
            .ok_or_else(|| FdwError::SchemaMismatch("text into_datum returned None".into())),
        PgPartitionType::Date => {
            // PG Date = days since 2000-01-01 (PG epoch).
            let parsed = chrono::NaiveDate::parse_from_str(raw, "%Y-%m-%d")
                .map_err(|e| FdwError::SchemaMismatch(format!("date '{raw}' parse failed: {e}")))?;
            let pg_epoch = chrono::NaiveDate::from_ymd_opt(2000, 1, 1)
                .ok_or_else(|| FdwError::SchemaMismatch("pg epoch construction failed".into()))?;
            let days = (parsed - pg_epoch).num_days();
            let pg_days: i32 = i32::try_from(days)
                .map_err(|_| FdwError::SchemaMismatch(format!("date '{raw}' out of i32 range")))?;
            let d = pgrx::datum::Date::try_from(pg_days as pg_sys::DateADT)
                .map_err(|e| FdwError::SchemaMismatch(format!("date '{raw}' invalid: {e:?}")))?;
            d.into_datum()
                .ok_or_else(|| FdwError::SchemaMismatch("date into_datum returned None".into()))
        }
    }
}

/// Safely downcast an Arrow array; returns `SchemaMismatch` on failure.
fn downcast<T: 'static>(arr: &dyn Array) -> FdwResult<&T> {
    arr.as_any().downcast_ref::<T>().ok_or_else(|| {
        FdwError::SchemaMismatch(format!(
            "downcast to {} failed for arrow type {:?}",
            std::any::type_name::<T>(),
            arr.data_type()
        ))
    })
}

fn json_value_to_datum(s: &str) -> FdwResult<Option<pg_sys::Datum>> {
    let value: serde_json::Value = serde_json::from_str(s)
        .map_err(|e| FdwError::SchemaMismatch(format!("invalid jsonb text: {e}")))?;
    Ok(JsonB(value).into_datum())
}

fn date_out_of_range(unix_days: i32) -> FdwError {
    FdwError::SchemaMismatch(format!("date out of range: unix-days={unix_days}"))
}

fn ts_out_of_range(unix_us: i64) -> FdwError {
    FdwError::SchemaMismatch(format!("timestamp out of range: unix-micros={unix_us}"))
}

fn date_conv_err(detail: String) -> FdwError {
    FdwError::SchemaMismatch(detail)
}

/// Render a `Decimal128` (i128 unscaled value + arrow `scale`) as a decimal
/// string suitable for `NumericIn`. Arrow's `scale` can be negative.
fn format_decimal128(value: i128, scale: i8) -> String {
    if value == 0 {
        return if scale > 0 {
            format!("0.{}", "0".repeat(scale as usize))
        } else {
            "0".to_string()
        };
    }
    let negative = value < 0;
    // i128::MIN abs() would overflow; use unsigned absolute value.
    let abs: u128 = if negative {
        value.unsigned_abs()
    } else {
        value as u128
    };
    let mut digits = abs.to_string();

    let s = match scale {
        s if s > 0 => {
            let scale = s as usize;
            if digits.len() <= scale {
                let pad = scale - digits.len();
                format!("0.{}{}", "0".repeat(pad), digits)
            } else {
                let split = digits.len() - scale;
                let frac = digits.split_off(split);
                format!("{digits}.{frac}")
            }
        }
        0 => digits,
        s => {
            // negative scale: append |s| zeros
            digits.push_str(&"0".repeat((-s) as usize));
            digits
        }
    };

    if negative {
        format!("-{s}")
    } else {
        s
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use super::{arrow_value_to_datum, FdwError, UNIX_TO_PG_EPOCH_MICROS};
    use arrow::array::{
        Array, ArrayRef, BinaryArray, BooleanArray, Date32Array, Decimal128Array, Float32Array,
        Float64Array, Int16Array, Int32Array, Int64Array, StringArray, TimestampMicrosecondArray,
    };
    use pgrx::prelude::*;
    use std::sync::Arc;

    /// Helper: assert the converter returns a non-null datum.
    fn must_get(arr: &dyn Array, row: usize, oid: pg_sys::Oid) -> pg_sys::Datum {
        arrow_value_to_datum(arr, row, oid).unwrap().unwrap()
    }

    /// Helper: assert the converter returns SQL NULL for the row.
    fn assert_null(arr: &dyn Array, row: usize, oid: pg_sys::Oid) {
        assert!(arrow_value_to_datum(arr, row, oid).unwrap().is_none());
    }

    #[pg_test]
    fn null_returns_none() {
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![None, Some(1)]));
        assert_null(arr.as_ref(), 0, pg_sys::INT8OID);
        assert!(arrow_value_to_datum(arr.as_ref(), 1, pg_sys::INT8OID)
            .unwrap()
            .is_some());
    }

    #[pg_test]
    fn bool_roundtrip() {
        let arr: ArrayRef = Arc::new(BooleanArray::from(vec![Some(true), Some(false), None]));
        // bool datums are stored in the low bit of the Datum word
        let d0 = must_get(arr.as_ref(), 0, pg_sys::BOOLOID);
        let d1 = must_get(arr.as_ref(), 1, pg_sys::BOOLOID);
        assert_eq!(d0.value() as u8 & 1, 1);
        assert_eq!(d1.value() as u8 & 1, 0);
        assert_null(arr.as_ref(), 2, pg_sys::BOOLOID);
    }

    #[pg_test]
    fn int16_roundtrip() {
        let arr: ArrayRef = Arc::new(Int16Array::from(vec![Some(i16::MAX), Some(-7), None]));
        let d_max = must_get(arr.as_ref(), 0, pg_sys::INT2OID);
        // sign-extend the low 16 bits
        assert_eq!(d_max.value() as i16, i16::MAX);
        let d_neg = must_get(arr.as_ref(), 1, pg_sys::INT2OID);
        assert_eq!(d_neg.value() as i16, -7);
        assert_null(arr.as_ref(), 2, pg_sys::INT2OID);
    }

    #[pg_test]
    fn int32_roundtrip() {
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![Some(i32::MAX), Some(-1), None]));
        let d = must_get(arr.as_ref(), 0, pg_sys::INT4OID);
        assert_eq!(d.value() as i32, i32::MAX);
        assert_eq!(
            must_get(arr.as_ref(), 1, pg_sys::INT4OID).value() as i32,
            -1
        );
        assert_null(arr.as_ref(), 2, pg_sys::INT4OID);
    }

    #[pg_test]
    fn int64_roundtrip() {
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![Some(42i64), Some(i64::MIN), None]));
        let d = must_get(arr.as_ref(), 0, pg_sys::INT8OID);
        assert_eq!(d.value() as i64, 42);
        let d2 = must_get(arr.as_ref(), 1, pg_sys::INT8OID);
        assert_eq!(d2.value() as i64, i64::MIN);
        assert_null(arr.as_ref(), 2, pg_sys::INT8OID);
    }

    #[pg_test]
    fn float32_roundtrip() {
        let arr: ArrayRef = Arc::new(Float32Array::from(vec![Some(1.5f32), None]));
        let d = must_get(arr.as_ref(), 0, pg_sys::FLOAT4OID);
        // float4 datum is the bit pattern in the low 32 bits
        assert_eq!(f32::from_bits(d.value() as u32), 1.5);
        assert_null(arr.as_ref(), 1, pg_sys::FLOAT4OID);
    }

    #[pg_test]
    fn float64_roundtrip() {
        let arr: ArrayRef = Arc::new(Float64Array::from(vec![Some(-2.25f64), None]));
        let d = must_get(arr.as_ref(), 0, pg_sys::FLOAT8OID);
        assert_eq!(f64::from_bits(d.value() as u64), -2.25);
        assert_null(arr.as_ref(), 1, pg_sys::FLOAT8OID);
    }

    #[pg_test]
    fn text_returns_datum() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec![Some("hi"), Some(""), None]));
        // We can't safely round-trip a TEXT varlena pointer back to a Rust
        // String without `unsafe` (pgrx's `FromDatum::from_datum` is unsafe by
        // contract, and this module enforces `#![forbid(unsafe_code)]`). The
        // assertion below verifies the converter dispatch + NULL handling;
        // datum-payload correctness is exercised end-to-end by the §16/17
        // pg_test SELECT suites that scan real parquet files.
        assert!(arrow_value_to_datum(arr.as_ref(), 0, pg_sys::TEXTOID)
            .unwrap()
            .is_some());
        assert!(arrow_value_to_datum(arr.as_ref(), 1, pg_sys::TEXTOID)
            .unwrap()
            .is_some());
        assert_null(arr.as_ref(), 2, pg_sys::TEXTOID);
    }

    #[pg_test]
    fn varchar_returns_datum() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec![Some("vc")]));
        assert!(arrow_value_to_datum(arr.as_ref(), 0, pg_sys::VARCHAROID)
            .unwrap()
            .is_some());
    }

    #[pg_test]
    fn largeutf8_text_returns_datum() {
        let arr: ArrayRef = Arc::new(arrow::array::LargeStringArray::from(vec![Some("big")]));
        assert!(arrow_value_to_datum(arr.as_ref(), 0, pg_sys::TEXTOID)
            .unwrap()
            .is_some());
    }

    #[pg_test]
    fn bytea_returns_datum() {
        let arr: ArrayRef = Arc::new(BinaryArray::from_iter_values([
            b"\x00\x01\x02".as_slice(),
            b"".as_slice(),
        ]));
        assert!(arrow_value_to_datum(arr.as_ref(), 0, pg_sys::BYTEAOID)
            .unwrap()
            .is_some());
        assert!(arrow_value_to_datum(arr.as_ref(), 1, pg_sys::BYTEAOID)
            .unwrap()
            .is_some());
    }

    #[pg_test]
    fn date_roundtrip_2025_01_01() {
        // Arrow Date32 day 20089 = 2025-01-01 (days since UNIX epoch).
        let arr: ArrayRef = Arc::new(Date32Array::from(vec![Some(20089), None]));
        let d = must_get(arr.as_ref(), 0, pg_sys::DATEOID);
        // DATE datum = pg-epoch days (i32). TryFrom is safe and validating.
        let date = pgrx::datum::Date::try_from(d.value() as pg_sys::DateADT).unwrap();
        assert_eq!(date.year(), 2025);
        assert_eq!(date.month(), 1);
        assert_eq!(date.day(), 1);
        assert_null(arr.as_ref(), 1, pg_sys::DATEOID);
    }

    #[pg_test]
    fn timestamp_roundtrip_2025_01_01() {
        // 2025-01-01 00:00:00 UTC = 1_735_689_600_000_000 μs since UNIX epoch.
        let unix_us: i64 = 1_735_689_600_000_000;
        let arr: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![Some(unix_us), None]));
        let d = must_get(arr.as_ref(), 0, pg_sys::TIMESTAMPOID);
        let ts = pgrx::datum::Timestamp::try_from(d.value() as pg_sys::Timestamp).unwrap();
        assert_eq!(ts.year(), 2025);
        assert_eq!(ts.month(), 1);
        assert_eq!(ts.day(), 1);
        assert_eq!(ts.hour(), 0);
        assert_eq!(ts.minute(), 0);
        assert_null(arr.as_ref(), 1, pg_sys::TIMESTAMPOID);
    }

    #[pg_test]
    fn timestamptz_roundtrip_2025_01_01() {
        let unix_us: i64 = 1_735_689_600_000_000;
        let arr: ArrayRef = Arc::new(
            TimestampMicrosecondArray::from(vec![Some(unix_us), None])
                .with_timezone("UTC".to_string()),
        );
        let d = must_get(arr.as_ref(), 0, pg_sys::TIMESTAMPTZOID);
        let ts =
            pgrx::datum::TimestampWithTimeZone::try_from(d.value() as pg_sys::TimestampTz).unwrap();
        // Year extraction depends on the session TimeZone GUC for tstz; the
        // critical invariant is that the raw stored value matches the expected
        // PG-epoch micros (UNIX micros − offset).
        let raw: pg_sys::TimestampTz = ts.into();
        assert_eq!(raw, unix_us - UNIX_TO_PG_EPOCH_MICROS);
        assert_null(arr.as_ref(), 1, pg_sys::TIMESTAMPTZOID);
    }

    #[pg_test]
    fn decimal128_returns_datum() {
        // Full string-formatting correctness is verified by
        // `format_decimal_cases` below; this test exercises the array
        // dispatch and NULL handling.
        let arr = Decimal128Array::from(vec![Some(12345i128), Some(-50i128), None])
            .with_precision_and_scale(10, 2)
            .unwrap();
        let arr: ArrayRef = Arc::new(arr);
        assert!(arrow_value_to_datum(arr.as_ref(), 0, pg_sys::NUMERICOID)
            .unwrap()
            .is_some());
        assert!(arrow_value_to_datum(arr.as_ref(), 1, pg_sys::NUMERICOID)
            .unwrap()
            .is_some());
        assert_null(arr.as_ref(), 2, pg_sys::NUMERICOID);
    }

    #[pg_test]
    fn jsonb_returns_datum() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec![Some(r#"{"k":1}"#), None]));
        assert!(arrow_value_to_datum(arr.as_ref(), 0, pg_sys::JSONBOID)
            .unwrap()
            .is_some());
        assert_null(arr.as_ref(), 1, pg_sys::JSONBOID);
    }

    #[pg_test]
    fn jsonb_invalid_text_errors() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec![Some("not-json{")]));
        let err = arrow_value_to_datum(arr.as_ref(), 0, pg_sys::JSONBOID).unwrap_err();
        match err {
            FdwError::SchemaMismatch(_) => {}
            other => panic!("expected SchemaMismatch, got {other:?}"),
        }
    }

    #[pg_test]
    fn unsupported_combo_errors() {
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![Some(1)]));
        // Int64 → BOOL is not in §7.4: should fail loudly.
        let err = arrow_value_to_datum(arr.as_ref(), 0, pg_sys::BOOLOID).unwrap_err();
        match err {
            FdwError::UnsupportedType { .. } => {}
            other => panic!("expected UnsupportedType, got {other:?}"),
        }
    }

    #[test]
    fn format_decimal_cases() {
        assert_eq!(super::format_decimal128(0, 0), "0");
        assert_eq!(super::format_decimal128(0, 2), "0.00");
        assert_eq!(super::format_decimal128(12345, 2), "123.45");
        assert_eq!(super::format_decimal128(-12345, 2), "-123.45");
        assert_eq!(super::format_decimal128(5, 3), "0.005");
        assert_eq!(super::format_decimal128(-5, 3), "-0.005");
        assert_eq!(super::format_decimal128(7, 0), "7");
        assert_eq!(super::format_decimal128(7, -2), "700");
    }

    #[pg_test]
    fn parse_int4_returns_datum() {
        use crate::fdw::options::PgPartitionType;
        let d = super::parse_text_to_datum(PgPartitionType::Int4, "2026").unwrap();
        // Datum bit-pattern for i32(2026) is the integer value itself.
        assert_eq!(d.value() as i32, 2026);
    }

    #[pg_test]
    fn parse_int4_negative_errors() {
        use crate::fdw::options::PgPartitionType;
        // SP-3b explicitly rejects negative partition values.
        assert!(super::parse_text_to_datum(PgPartitionType::Int4, "-1").is_err());
    }

    #[pg_test]
    fn parse_int4_non_numeric_errors() {
        use crate::fdw::options::PgPartitionType;
        assert!(super::parse_text_to_datum(PgPartitionType::Int4, "abc").is_err());
    }

    #[pg_test]
    fn parse_text_returns_datum() {
        use crate::fdw::options::PgPartitionType;
        // A text datum is a varlena pointer; assert it's non-null.
        let d = super::parse_text_to_datum(PgPartitionType::Text, "us-west").unwrap();
        assert!(d.value() != 0);
    }

    #[pg_test]
    fn parse_date_iso_returns_datum() {
        use crate::fdw::options::PgPartitionType;
        let _ = super::parse_text_to_datum(PgPartitionType::Date, "2026-06-22").unwrap();
    }

    #[pg_test]
    fn parse_date_malformed_errors() {
        use crate::fdw::options::PgPartitionType;
        assert!(super::parse_text_to_datum(PgPartitionType::Date, "06/22/2026").is_err());
    }

    #[test]
    fn typename_int32_returns_integer() {
        use arrow::datatypes::DataType;
        assert_eq!(
            super::arrow_type_to_pg_typename(&DataType::Int32).unwrap(),
            "INTEGER"
        );
    }

    #[test]
    fn typename_int64_returns_bigint() {
        use arrow::datatypes::DataType;
        assert_eq!(
            super::arrow_type_to_pg_typename(&DataType::Int64).unwrap(),
            "BIGINT"
        );
    }

    #[test]
    fn typename_utf8_returns_text() {
        use arrow::datatypes::DataType;
        assert_eq!(
            super::arrow_type_to_pg_typename(&DataType::Utf8).unwrap(),
            "TEXT"
        );
    }

    #[test]
    fn typename_timestamp_no_tz() {
        use arrow::datatypes::{DataType, TimeUnit};
        assert_eq!(
            super::arrow_type_to_pg_typename(&DataType::Timestamp(TimeUnit::Microsecond, None))
                .unwrap(),
            "TIMESTAMP"
        );
    }

    #[test]
    fn typename_timestamp_with_tz() {
        use arrow::datatypes::{DataType, TimeUnit};
        assert_eq!(
            super::arrow_type_to_pg_typename(&DataType::Timestamp(
                TimeUnit::Microsecond,
                Some("UTC".into())
            ))
            .unwrap(),
            "TIMESTAMPTZ"
        );
    }

    #[test]
    fn typename_decimal128_returns_numeric() {
        use arrow::datatypes::DataType;
        assert_eq!(
            super::arrow_type_to_pg_typename(&DataType::Decimal128(18, 2)).unwrap(),
            "NUMERIC(18,2)"
        );
    }

    #[test]
    fn typename_unsupported_errors() {
        use arrow::datatypes::DataType;
        let item = std::sync::Arc::new(arrow::datatypes::Field::new("i", DataType::Int32, true));
        assert!(super::arrow_type_to_pg_typename(&DataType::List(item)).is_err());
    }
}
