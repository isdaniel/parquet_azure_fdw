use arrow::array::{ArrayRef, Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use parquet_azure_fdw::fdw::modify::kernel::{apply_edits, BlobEdits, RowOverride};
use std::sync::Arc;

fn batch_of_ints(name: &str, vals: &[i32]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Int32, true)]));
    RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vals.to_vec())) as ArrayRef],
    )
    .unwrap()
}

#[test]
fn pure_delete_drops_listed_rows() {
    let b = batch_of_ints("x", &[10, 20, 30, 40]);
    let schema = b.schema();
    let mut edits = BlobEdits {
        deletes: roaring::RoaringBitmap::new(),
        updates: Default::default(),
    };
    edits.deletes.insert(1); // drop value 20
    edits.deletes.insert(3); // drop value 40
    let out = apply_edits(vec![b], schema, &edits).unwrap();
    let merged = arrow::compute::concat_batches(&out[0].schema(), &out).unwrap();
    let col = merged
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(col.values(), &[10, 30]);
}

#[test]
fn pure_update_overrides_listed_rows() {
    let b = batch_of_ints("x", &[10, 20, 30]);
    let schema = b.schema();
    let mut edits = BlobEdits {
        deletes: roaring::RoaringBitmap::new(),
        updates: Default::default(),
    };
    edits.updates.insert(
        1,
        RowOverride {
            values: vec![Some(Arc::new(Int32Array::from(vec![999])) as ArrayRef)],
        },
    );
    let out = apply_edits(vec![b], schema, &edits).unwrap();
    let merged = arrow::compute::concat_batches(&out[0].schema(), &out).unwrap();
    let col = merged
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(col.values(), &[10, 999, 30]);
}

#[test]
fn delete_all_rows_yields_empty() {
    let b = batch_of_ints("x", &[1, 2]);
    let schema = b.schema();
    let mut edits = BlobEdits {
        deletes: roaring::RoaringBitmap::new(),
        updates: Default::default(),
    };
    edits.deletes.insert(0);
    edits.deletes.insert(1);
    let out = apply_edits(vec![b], schema, &edits).unwrap();
    let merged = arrow::compute::concat_batches(&out[0].schema(), &out).unwrap();
    assert_eq!(merged.num_rows(), 0);
}

#[test]
fn empty_edits_pass_through() {
    let b = batch_of_ints("x", &[1, 2, 3]);
    let schema = b.schema();
    let edits = BlobEdits {
        deletes: roaring::RoaringBitmap::new(),
        updates: Default::default(),
    };
    let out = apply_edits(vec![b.clone()], schema, &edits).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].num_rows(), 3);
}

#[test]
fn update_and_delete_combined_with_string_col() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int32, true),
        Field::new("s", DataType::Utf8, true),
    ]));
    let b = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
            Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef,
        ],
    )
    .unwrap();
    let mut edits = BlobEdits {
        deletes: roaring::RoaringBitmap::new(),
        updates: Default::default(),
    };
    edits.deletes.insert(0); // drop row 0
    edits.updates.insert(
        2,
        RowOverride {
            values: vec![
                None,                                                       // leave i unchanged
                Some(Arc::new(StringArray::from(vec!["zzz"])) as ArrayRef), // override s
            ],
        },
    );
    let out = apply_edits(vec![b], schema.clone(), &edits).unwrap();
    let merged = arrow::compute::concat_batches(&schema, &out).unwrap();
    assert_eq!(merged.num_rows(), 2);
    let i = merged
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let s = merged
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(i.values(), &[2, 3]);
    assert_eq!(s.value(0), "b");
    assert_eq!(s.value(1), "zzz");
}
