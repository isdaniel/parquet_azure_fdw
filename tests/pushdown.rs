use arrow::datatypes::{DataType, Field, TimeUnit};
use parquet_azure_fdw::fdw::pushdown::is_pushable;
use std::sync::Arc;

#[test]
fn whitelist_accepts_eq_on_int64() {
    assert!(is_pushable("=", &DataType::Int64));
}

#[test]
fn whitelist_accepts_le_on_utf8() {
    assert!(is_pushable("<=", &DataType::Utf8));
}

#[test]
fn whitelist_accepts_is_null_on_boolean() {
    assert!(is_pushable("IS NULL", &DataType::Boolean));
    assert!(is_pushable("IS NOT NULL", &DataType::Boolean));
}

#[test]
fn whitelist_accepts_eq_on_timestamp_and_decimal() {
    assert!(is_pushable(
        "=",
        &DataType::Timestamp(TimeUnit::Microsecond, None)
    ));
    assert!(is_pushable("<", &DataType::Decimal128(18, 2)));
}

#[test]
fn whitelist_rejects_like_operator() {
    // LIKE is not in the supported operator set even on a supported type.
    assert!(!is_pushable("LIKE", &DataType::Utf8));
}

#[test]
fn whitelist_rejects_unsupported_type() {
    // List is outside the supported type set even with a supported op.
    let inner = Arc::new(Field::new("item", DataType::Int32, true));
    assert!(!is_pushable("=", &DataType::List(inner)));
}
