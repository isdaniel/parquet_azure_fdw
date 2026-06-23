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

#[test]
fn whitelist_rejects_timestamptz() {
    use arrow::datatypes::TimeUnit;
    // TIMESTAMPTZ carries session timezone semantics we don't model.
    let ts_tz = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
    assert!(
        !is_pushable("=", &ts_tz),
        "TIMESTAMPTZ must be non-pushable"
    );
}

#[test]
fn whitelist_accepts_timestamp_without_tz() {
    use arrow::datatypes::TimeUnit;
    let ts_no_tz = DataType::Timestamp(TimeUnit::Microsecond, None);
    assert!(is_pushable("=", &ts_no_tz));
}

#[test]
fn whitelist_rejects_decimal_scale_over_18() {
    // Precision is bounded by `i128` (parquet repr); scale >18 isn't well
    // supported in our stats handling, so we drop the qual.
    assert!(!is_pushable("=", &DataType::Decimal128(20, 19)));
    assert!(is_pushable("=", &DataType::Decimal128(18, 2)));
}

#[test]
fn next_lex_upper_ascii_prefix() {
    assert_eq!(
        parquet_azure_fdw::fdw::pushdown::next_lex_upper("al"),
        Some("am".into())
    );
}

#[test]
fn next_lex_upper_empty_is_none() {
    assert_eq!(parquet_azure_fdw::fdw::pushdown::next_lex_upper(""), None);
}
