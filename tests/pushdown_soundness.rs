#![cfg(feature = "pg_test")]
//! Soundness: for each hand-mapped (clause, PushedExpr) pair, the rows
//! emitted by the pushdown machinery must equal the rows that satisfy the
//! Rust-side truth evaluation of the clause. No SQL/pgrx round-trip; this is
//! a fast, deterministic corpus check.

use arrow::array::{Array, Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use bytes::Bytes;
use futures::StreamExt;
use parquet::arrow::arrow_reader::ArrowReaderMetadata;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use parquet_azure_fdw::fdw::pushdown::{
    build_row_filter, prune_row_groups, PushedExpr, PushedOp, PushedQual, ScalarValueRepr,
};
use std::sync::Arc;

type CorpusData = (Bytes, Arc<Schema>, Vec<Option<i32>>, Vec<Option<String>>);
type TruthFn = fn(&[Option<i32>], &[Option<String>]) -> Vec<usize>;
type PushFn = fn() -> Vec<PushedExpr>;

fn corpus_parquet() -> CorpusData {
    // Build 100 rows across 2 row groups (50 + 50). `id` is unique 0..100.
    // `name` is "user_<i>" for 80% of rows, NULL when i % 5 == 0.
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let mut buf: Vec<u8> = Vec::new();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(50))
        .build();
    let mut w = ArrowWriter::try_new(&mut buf, schema.clone(), Some(props)).unwrap();
    let mut all_ids: Vec<Option<i32>> = Vec::with_capacity(100);
    let mut all_names: Vec<Option<String>> = Vec::with_capacity(100);
    for chunk_start in [0i32, 50] {
        let ids: Vec<i32> = (chunk_start..chunk_start + 50).collect();
        let names_owned: Vec<String> = ids.iter().map(|i| format!("user_{i}")).collect();
        let names_for_arrow: Vec<Option<&str>> = ids
            .iter()
            .zip(names_owned.iter())
            .map(|(i, s)| if i % 5 == 0 { None } else { Some(s.as_str()) })
            .collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(ids.clone())),
                Arc::new(StringArray::from(names_for_arrow.clone())),
            ],
        )
        .unwrap();
        w.write(&batch).unwrap();
        for (i, n) in ids.iter().zip(names_for_arrow.iter()) {
            all_ids.push(Some(*i));
            all_names.push(n.map(|s| s.to_string()));
        }
    }
    w.close().unwrap();
    (Bytes::from(buf), schema, all_ids, all_names)
}

#[derive(Debug)]
struct Case {
    name: &'static str,
    /// Rust-side truth: returns the indices of rows that match.
    truth: TruthFn,
    /// Equivalent PushedExpr tree.
    push: PushFn,
}

fn cases() -> Vec<Case> {
    fn eq_id_42_truth(ids: &[Option<i32>], _names: &[Option<String>]) -> Vec<usize> {
        ids.iter()
            .enumerate()
            .filter_map(|(i, v)| if *v == Some(42) { Some(i) } else { None })
            .collect()
    }
    fn eq_id_42_push() -> Vec<PushedExpr> {
        vec![PushedExpr::Leaf(PushedQual {
            col: 0,
            op: PushedOp::Eq,
            value: ScalarValueRepr::I32(42),
        })]
    }
    fn name_is_null_truth(_ids: &[Option<i32>], names: &[Option<String>]) -> Vec<usize> {
        names
            .iter()
            .enumerate()
            .filter_map(|(i, v)| if v.is_none() { Some(i) } else { None })
            .collect()
    }
    fn name_is_null_push() -> Vec<PushedExpr> {
        vec![PushedExpr::Leaf(PushedQual {
            col: 1,
            op: PushedOp::IsNull,
            value: ScalarValueRepr::Null,
        })]
    }
    fn id_lt_30_and_ge_10_truth(ids: &[Option<i32>], _names: &[Option<String>]) -> Vec<usize> {
        ids.iter()
            .enumerate()
            .filter_map(|(i, v)| match v {
                Some(x) if *x >= 10 && *x < 30 => Some(i),
                _ => None,
            })
            .collect()
    }
    fn id_lt_30_and_ge_10_push() -> Vec<PushedExpr> {
        vec![PushedExpr::And(vec![
            PushedExpr::Leaf(PushedQual {
                col: 0,
                op: PushedOp::Ge,
                value: ScalarValueRepr::I32(10),
            }),
            PushedExpr::Leaf(PushedQual {
                col: 0,
                op: PushedOp::Lt,
                value: ScalarValueRepr::I32(30),
            }),
        ])]
    }
    fn id_in_set_truth(ids: &[Option<i32>], _names: &[Option<String>]) -> Vec<usize> {
        let set = [1, 3, 17, 60, 99];
        ids.iter()
            .enumerate()
            .filter_map(|(i, v)| match v {
                Some(x) if set.contains(x) => Some(i),
                _ => None,
            })
            .collect()
    }
    fn id_in_set_push() -> Vec<PushedExpr> {
        vec![PushedExpr::Or(
            [1, 3, 17, 60, 99]
                .iter()
                .map(|&n| {
                    PushedExpr::Leaf(PushedQual {
                        col: 0,
                        op: PushedOp::Eq,
                        value: ScalarValueRepr::I32(n),
                    })
                })
                .collect(),
        )]
    }

    vec![
        Case {
            name: "eq_id_42",
            truth: eq_id_42_truth,
            push: eq_id_42_push,
        },
        Case {
            name: "name_is_null",
            truth: name_is_null_truth,
            push: name_is_null_push,
        },
        Case {
            name: "id_in_10_30_range",
            truth: id_lt_30_and_ge_10_truth,
            push: id_lt_30_and_ge_10_push,
        },
        Case {
            name: "id_in_set",
            truth: id_in_set_truth,
            push: id_in_set_push,
        },
    ]
}

#[tokio::test(flavor = "current_thread")]
async fn pushdown_soundness_corpus() {
    let (bytes, schema, ids, names) = corpus_parquet();
    let arrow_schema = schema.clone();

    for case in cases() {
        let truth_rows = (case.truth)(&ids, &names);
        let truth_ids: Vec<i32> = truth_rows.iter().map(|i| ids[*i].unwrap()).collect();

        // Row-group pruning (Slice 1).
        let md = ArrowReaderMetadata::load(&bytes, Default::default()).unwrap();
        let exprs = (case.push)();
        let kept = prune_row_groups(md.metadata(), &exprs, arrow_schema.as_ref());

        let parquet_schema = md.metadata().file_metadata().schema_descr_ptr();

        let cursor2 = std::io::Cursor::new(bytes.clone());
        let mut b = parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder::new(cursor2)
            .await
            .unwrap();
        if let Some(k) = kept {
            b = b.with_row_groups(k);
        }
        if let Some(rf) = build_row_filter(&exprs, arrow_schema.as_ref(), &parquet_schema) {
            b = b.with_row_filter(rf);
        }
        let mut stream = b.build().unwrap();
        let mut got_ids: Vec<i32> = Vec::new();
        while let Some(batch) = stream.next().await {
            let batch = batch.unwrap();
            let col = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            for i in 0..batch.num_rows() {
                if !col.is_null(i) {
                    got_ids.push(col.value(i));
                }
            }
        }
        got_ids.sort();
        let mut truth_sorted = truth_ids.clone();
        truth_sorted.sort();
        assert_eq!(
            got_ids, truth_sorted,
            "case {} — pushdown rows differ from truth",
            case.name
        );
    }
}
