//! Row-group pruning unit tests — drive `prune_row_groups` directly against
//! in-memory parquet metadata. Fast (no fake/blob).

use arrow::array::{Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ArrowReaderMetadata;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use parquet_azure_fdw::fdw::pushdown::{
    prune_row_groups, PushedExpr, PushedOp, PushedQual, ScalarValueRepr,
};
use std::sync::Arc;

/// Build an in-memory parquet with two row groups: rows 0..50 then 50..100.
/// Returns the bytes + the arrow schema.
fn two_rg_int_parquet() -> (bytes::Bytes, Arc<Schema>) {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let mut buf: Vec<u8> = Vec::new();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(50))
        .build();
    let mut w = ArrowWriter::try_new(&mut buf, schema.clone(), Some(props)).unwrap();
    let batch_a = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from((0..50).collect::<Vec<_>>()))],
    )
    .unwrap();
    let batch_b = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from((50..100).collect::<Vec<_>>()))],
    )
    .unwrap();
    w.write(&batch_a).unwrap();
    w.write(&batch_b).unwrap();
    w.close().unwrap();
    (bytes::Bytes::from(buf), schema)
}

#[test]
fn lt_25_prunes_second_row_group() {
    let (bytes, schema) = two_rg_int_parquet();
    let md = ArrowReaderMetadata::load(&bytes, Default::default()).unwrap();
    let meta = md.metadata();
    assert_eq!(meta.num_row_groups(), 2);

    let exprs = vec![PushedExpr::Leaf(PushedQual {
        col: 0,
        op: PushedOp::Lt,
        value: ScalarValueRepr::I32(25),
    })];
    let keep = prune_row_groups(meta, &exprs, schema.as_ref())
        .expect("a determinable predicate must yield Some");
    assert_eq!(keep, vec![0], "row group 1 [50..100) cannot satisfy id<25");
}

#[test]
fn eq_outside_min_max_prunes_all_groups() {
    let (bytes, schema) = two_rg_int_parquet();
    let md = ArrowReaderMetadata::load(&bytes, Default::default()).unwrap();
    let meta = md.metadata();

    let exprs = vec![PushedExpr::Leaf(PushedQual {
        col: 0,
        op: PushedOp::Eq,
        value: ScalarValueRepr::I32(9999),
    })];
    let keep = prune_row_groups(meta, &exprs, schema.as_ref()).expect("determinable");
    assert!(keep.is_empty(), "no group contains 9999");
}

#[test]
fn empty_exprs_returns_none() {
    let (bytes, schema) = two_rg_int_parquet();
    let md = ArrowReaderMetadata::load(&bytes, Default::default()).unwrap();
    let meta = md.metadata();
    assert!(prune_row_groups(meta, &[], schema.as_ref()).is_none());
}

#[test]
fn out_of_range_column_index_is_indeterminate() {
    let (bytes, schema) = two_rg_int_parquet();
    let md = ArrowReaderMetadata::load(&bytes, Default::default()).unwrap();
    let meta = md.metadata();

    let exprs = vec![PushedExpr::Leaf(PushedQual {
        col: 99, // does not exist
        op: PushedOp::Lt,
        value: ScalarValueRepr::I32(0),
    })];
    // Only Indeterminate signals were produced → returns None (no pruning).
    assert!(prune_row_groups(meta, &exprs, schema.as_ref()).is_none());
}

#[test]
fn and_prunes_when_one_branch_cannot_match() {
    let (bytes, schema) = two_rg_int_parquet();
    let md = ArrowReaderMetadata::load(&bytes, Default::default()).unwrap();
    let meta = md.metadata();

    // id < 25 AND id >= 60 — group 0 fails 2nd branch, group 1 fails 1st.
    let exprs = vec![PushedExpr::And(vec![
        PushedExpr::Leaf(PushedQual {
            col: 0,
            op: PushedOp::Lt,
            value: ScalarValueRepr::I32(25),
        }),
        PushedExpr::Leaf(PushedQual {
            col: 0,
            op: PushedOp::Ge,
            value: ScalarValueRepr::I32(60),
        }),
    ])];
    let keep = prune_row_groups(meta, &exprs, schema.as_ref()).expect("determinable");
    assert!(keep.is_empty());
}
