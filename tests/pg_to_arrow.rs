//! Pure-Rust unit tests for the PG→Arrow type-mapping core. Exercises
//! `pg_oid_to_arrow_type` and the `RecordBatchBuilders` typed `append_*`
//! helpers without a Postgres backend. End-to-end INSERT/COPY behavior is
//! covered by the §17 `#[pg_test]` suite.

use arrow::array::{Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use parquet_azure_fdw::convert::pg_to_arrow::{pg_oid_to_arrow_type, RecordBatchBuilders};
use pgrx::pg_sys;
use std::sync::Arc;

#[test]
fn primitive_mapping() {
    assert_eq!(
        pg_oid_to_arrow_type(pg_sys::INT8OID).unwrap(),
        DataType::Int64
    );
    assert_eq!(
        pg_oid_to_arrow_type(pg_sys::BOOLOID).unwrap(),
        DataType::Boolean
    );
    assert_eq!(
        pg_oid_to_arrow_type(pg_sys::TEXTOID).unwrap(),
        DataType::Utf8
    );
}

#[test]
fn numeric_and_temporal_mapping() {
    assert_eq!(
        pg_oid_to_arrow_type(pg_sys::INT2OID).unwrap(),
        DataType::Int16
    );
    assert_eq!(
        pg_oid_to_arrow_type(pg_sys::INT4OID).unwrap(),
        DataType::Int32
    );
    assert_eq!(
        pg_oid_to_arrow_type(pg_sys::FLOAT4OID).unwrap(),
        DataType::Float32
    );
    assert_eq!(
        pg_oid_to_arrow_type(pg_sys::FLOAT8OID).unwrap(),
        DataType::Float64
    );
    assert_eq!(
        pg_oid_to_arrow_type(pg_sys::DATEOID).unwrap(),
        DataType::Date32
    );
    assert_eq!(
        pg_oid_to_arrow_type(pg_sys::TIMESTAMPOID).unwrap(),
        DataType::Timestamp(TimeUnit::Microsecond, None)
    );
}

#[test]
fn varchar_maps_to_utf8() {
    assert_eq!(
        pg_oid_to_arrow_type(pg_sys::VARCHAROID).unwrap(),
        DataType::Utf8
    );
}

#[test]
fn unsupported_type_errors() {
    assert!(pg_oid_to_arrow_type(pg_sys::POINTOID).is_err());
}

#[test]
fn builders_roundtrip_two_columns() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let mut b = RecordBatchBuilders::new(schema.clone(), 4).unwrap();
    b.append_i64(0, Some(1)).unwrap();
    b.append_str(1, Some("alice")).unwrap();
    b.append_i64(0, Some(2)).unwrap();
    b.append_str(1, None).unwrap();
    assert_eq!(b.len(), 2);
    let batch = b.finish().unwrap();
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.num_columns(), 2);

    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 2);

    let names = batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(names.value(0), "alice");
    assert!(names.is_null(1));
}

#[test]
fn builders_reject_wrong_typed_append() {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let mut b = RecordBatchBuilders::new(schema, 1).unwrap();
    // append_str on an Int64 column should fail loudly.
    assert!(b.append_str(0, Some("nope")).is_err());
}

use arrow::array::{Date32Array, TimestampMicrosecondArray};

#[test]
fn date_oid_maps_to_date32() {
    let dt = pg_oid_to_arrow_type(pg_sys::DATEOID).unwrap();
    assert_eq!(dt, DataType::Date32);
}

#[test]
fn timestamp_oid_maps_to_microseconds_no_tz() {
    let dt = pg_oid_to_arrow_type(pg_sys::TIMESTAMPOID).unwrap();
    assert_eq!(dt, DataType::Timestamp(TimeUnit::Microsecond, None));
}

#[test]
fn timestamptz_oid_maps_to_microseconds_utc() {
    let dt = pg_oid_to_arrow_type(pg_sys::TIMESTAMPTZOID).unwrap();
    assert_eq!(
        dt,
        DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
    );
}

#[test]
fn builders_round_trip_date_and_timestamps() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("d", DataType::Date32, true),
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new(
            "tstz",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            true,
        ),
    ]));
    let mut b = RecordBatchBuilders::new(schema, 4).unwrap();
    b.append_date(0, Some(19_000)).unwrap();
    b.append_ts_us(1, Some(1_700_000_000_000_000)).unwrap();
    b.append_tstz_us(2, Some(1_700_000_000_000_000)).unwrap();
    let batch = b.finish().unwrap();
    assert_eq!(
        batch
            .column(0)
            .as_any()
            .downcast_ref::<Date32Array>()
            .unwrap()
            .value(0),
        19_000
    );
    assert_eq!(
        batch
            .column(1)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .value(0),
        1_700_000_000_000_000
    );
}

#[test]
fn bytea_oid_maps_to_binary() {
    let dt = pg_oid_to_arrow_type(pg_sys::BYTEAOID).unwrap();
    assert_eq!(dt, DataType::Binary);
}

#[test]
fn builders_round_trip_bytea() {
    use arrow::array::BinaryArray;
    let schema = Arc::new(Schema::new(vec![Field::new("b", DataType::Binary, true)]));
    let mut b = RecordBatchBuilders::new(schema, 2).unwrap();
    b.append_bytes(0, Some(b"hello")).unwrap();
    b.append_bytes(0, None).unwrap();
    let batch = b.finish().unwrap();
    let arr = batch
        .column(0)
        .as_any()
        .downcast_ref::<BinaryArray>()
        .unwrap();
    assert_eq!(arr.value(0), b"hello");
    assert!(arr.is_null(1));
}

#[test]
fn numeric_oid_maps_to_decimal128_38_9() {
    let dt = pg_oid_to_arrow_type(pg_sys::NUMERICOID).unwrap();
    assert_eq!(dt, DataType::Decimal128(38, 9));
}

#[test]
fn jsonb_oid_maps_to_utf8() {
    let dt = pg_oid_to_arrow_type(pg_sys::JSONBOID).unwrap();
    assert_eq!(dt, DataType::Utf8);
}

#[test]
fn builders_round_trip_decimal128_38_9() {
    use arrow::array::Decimal128Array;
    let schema = Arc::new(Schema::new(vec![Field::new(
        "n",
        DataType::Decimal128(38, 9),
        true,
    )]));
    let mut b = RecordBatchBuilders::new(schema, 1).unwrap();
    // 1.5 at scale 9 = 1_500_000_000
    b.append_decimal128(0, Some(1_500_000_000_i128)).unwrap();
    let batch = b.finish().unwrap();
    let arr = batch
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .unwrap();
    assert_eq!(arr.value(0), 1_500_000_000_i128);
}

#[test]
fn builders_round_trip_jsonb_text() {
    use arrow::array::StringArray;
    let schema = Arc::new(Schema::new(vec![Field::new("j", DataType::Utf8, true)]));
    let mut b = RecordBatchBuilders::new(schema, 1).unwrap();
    b.append_jsonb_text(0, Some(r#"{"a":1}"#)).unwrap();
    let batch = b.finish().unwrap();
    let arr = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(arr.value(0), r#"{"a":1}"#);
}

#[test]
fn finish_rejects_uneven_columns() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Utf8, true),
    ]));
    let mut b = RecordBatchBuilders::new(schema, 4).unwrap();
    b.append_i64(0, Some(1)).unwrap();
    b.append_i64(0, Some(2)).unwrap();
    // forget to append column b for the second row
    b.append_str(1, Some("x")).unwrap();
    let err = b.finish().expect_err("must fail");
    let msg = format!("{err:?}");
    assert!(msg.contains("column 1") || msg.contains("rows"), "{msg}");
}

#[test]
fn len_reflects_first_column_length() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Utf8, true),
    ]));
    let mut b = RecordBatchBuilders::new(schema, 4).unwrap();
    assert!(b.is_empty());
    b.append_i64(0, Some(1)).unwrap();
    b.append_str(1, Some("x")).unwrap();
    assert_eq!(b.len(), 1);
    b.append_i64(0, Some(2)).unwrap();
    b.append_str(1, Some("y")).unwrap();
    assert_eq!(b.len(), 2);
}
