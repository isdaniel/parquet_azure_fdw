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

#[test]
fn streaming_multi_batch_deletes_and_updates_straddle_boundaries() {
    use arrow::array::{ArrayRef, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet_azure_fdw::fdw::modify::kernel::{apply_edits, BlobEdits, RowOverride};
    use std::sync::Arc;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("name", DataType::Utf8, true),
    ]));
    let mk = |ids: &[i64], names: &[&str]| {
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef,
                Arc::new(StringArray::from(names.to_vec())) as ArrayRef,
            ],
        )
        .unwrap()
    };
    // Absolute rows: 0,1,2 | 3,4,5 | 6,7,8
    let input = vec![
        mk(&[0, 1, 2], &["a", "b", "c"]),
        mk(&[3, 4, 5], &["d", "e", "f"]),
        mk(&[6, 7, 8], &["g", "h", "i"]),
    ];
    let mut edits = BlobEdits::default();
    edits.deletes.insert(1); // batch 0
    edits.deletes.insert(5); // batch 1
    edits.deletes.insert(6); // batch 2 (first row)
    edits.updates.insert(
        4, // batch 1, local 1 — override name
        RowOverride {
            values: vec![
                None,
                Some(Arc::new(StringArray::from(vec!["E"])) as ArrayRef),
            ],
        },
    );
    edits.updates.insert(
        8, // batch 2, local 2 — override id
        RowOverride {
            values: vec![Some(Arc::new(Int64Array::from(vec![88])) as ArrayRef), None],
        },
    );

    let out = apply_edits(input, schema.clone(), &edits).unwrap();
    // Flatten output for assertion.
    let mut got_ids = Vec::new();
    let mut got_names = Vec::new();
    for b in &out {
        let ic = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let nc = b.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..b.num_rows() {
            got_ids.push(ic.value(i));
            got_names.push(nc.value(i).to_string());
        }
    }
    // Deleted: 1, 5, 6. Updated: row4 name→E, row8 id→88.
    // Surviving absolute rows: 0,2,3,4,7,8.
    assert_eq!(got_ids, vec![0, 2, 3, 4, 7, 88]);
    assert_eq!(got_names, vec!["a", "c", "d", "E", "h", "i"]);
}
