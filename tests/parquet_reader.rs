use futures::StreamExt;
use parquet_azure_fdw::parquet_io::{open_local_stream, ParquetReadOptions};
use parquet_azure_fdw::runtime::block_on;
use std::path::{Path, PathBuf};

fn fixture_path() -> PathBuf {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/two_rowgroups.parquet");
    if !p.exists() {
        write_fixture(&p);
    }
    p
}

fn write_fixture(p: &Path) {
    use arrow::array::{Int64Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use std::fs::File;
    use std::sync::Arc;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(2))
        .build();
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    let f = File::create(p).unwrap();
    let mut w = ArrowWriter::try_new(f, schema.clone(), Some(props)).unwrap();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
            Arc::new(StringArray::from(vec![
                Some("a"),
                Some("b"),
                Some("c"),
                Some("d"),
            ])),
        ],
    )
    .unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

#[test]
fn projection_only_selects_one_column() {
    let p = fixture_path();
    let batches: Vec<_> = block_on(async {
        let s = open_local_stream(
            &p,
            ParquetReadOptions {
                projection: Some(vec![0]),
                row_filter: None,
            },
        )
        .await
        .unwrap();
        s.collect::<Vec<_>>().await
    });
    let batch = batches.into_iter().next().unwrap().unwrap();
    assert_eq!(batch.num_columns(), 1);
    assert_eq!(batch.schema().field(0).name(), "id");
}

#[test]
fn full_scan_yields_all_rows() {
    let p = fixture_path();
    let total: usize = block_on(async {
        let s = open_local_stream(
            &p,
            ParquetReadOptions {
                projection: None,
                row_filter: None,
            },
        )
        .await
        .unwrap();
        s.fold(0usize, |acc, b| async move { acc + b.unwrap().num_rows() })
            .await
    });
    assert_eq!(total, 4);
}

#[tokio::test]
async fn open_stream_from_bytes_round_trips_a_small_file() {
    use arrow::array::{Int32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use bytes::Bytes;
    use futures::StreamExt;
    use parquet_azure_fdw::parquet_io::reader::{open_stream_from_bytes, ParquetReadOptions};
    use parquet_azure_fdw::parquet_io::writer::{Compression, ParquetBatchWriter};
    use std::sync::Arc;

    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    let mut w = ParquetBatchWriter::new(schema, Compression::Snappy).unwrap();
    w.write(&batch).unwrap();
    let bytes: Bytes = w.finish().unwrap();

    let mut s = open_stream_from_bytes(bytes, ParquetReadOptions::default())
        .await
        .unwrap();
    let mut total = 0usize;
    while let Some(b) = s.next().await {
        total += b.unwrap().num_rows();
    }
    assert_eq!(total, 3);
}
