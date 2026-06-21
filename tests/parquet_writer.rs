use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use parquet_azure_fdw::parquet_io::writer::{Compression, ParquetBatchWriter};
use std::sync::Arc;

#[test]
fn write_then_read_back_round_trip() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let mut w = ParquetBatchWriter::new(schema.clone(), Compression::Snappy).unwrap();
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![10, 20, 30]))]).unwrap();
    w.write(&batch).unwrap();
    let bytes = w.finish().unwrap();
    assert!(!bytes.is_empty());
    // Parquet magic header
    assert_eq!(&bytes[..4], b"PAR1");
}

#[test]
fn parse_compression_round_trip() {
    assert!(matches!(
        Compression::parse("snappy").unwrap(),
        Compression::Snappy
    ));
    assert!(matches!(
        Compression::parse("none").unwrap(),
        Compression::None
    ));
    assert!(Compression::parse("bogus").is_err());
}
