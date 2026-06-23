#![forbid(unsafe_code)]
//! `ArrowWriter` wrapper that produces an in-memory parquet blob.
//!
//! The Azure write path stages a whole parquet file in memory before issuing a
//! single put-blob (or upload-block sequence). This module owns the
//! `RecordBatch` → bytes conversion; the Azure uploader stays oblivious to the
//! parquet format.

use crate::error::{FdwError, FdwResult};
use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use bytes::Bytes;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression as PqCompression;
use parquet::file::properties::WriterProperties;

/// User-facing compression choice. Maps to the parquet column codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    None,
    Snappy,
    Zstd,
    Gzip,
}

impl Compression {
    /// Parse from FDW option string. Case-insensitive; empty string maps to
    /// `None`.
    pub fn parse(s: &str) -> FdwResult<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "none" | "" => Compression::None,
            "snappy" => Compression::Snappy,
            "zstd" => Compression::Zstd,
            "gzip" => Compression::Gzip,
            other => return Err(FdwError::InvalidOption(format!("compression='{other}'"))),
        })
    }

    fn to_pq(self) -> PqCompression {
        match self {
            Compression::None => PqCompression::UNCOMPRESSED,
            Compression::Snappy => PqCompression::SNAPPY,
            Compression::Zstd => PqCompression::ZSTD(Default::default()),
            Compression::Gzip => PqCompression::GZIP(Default::default()),
        }
    }

    /// Map a parquet-rs `Compression` value back to our user-facing enum.
    ///
    /// Parquet codec parameters (e.g. gzip level, zstd level) are NOT preserved
    /// across the round-trip — we always rewrite using the codec's default
    /// level. Acceptable because UPDATE/DELETE rewrites are an exact-row
    /// operation, not a re-encode-for-storage optimization; default levels are
    /// what fresh INSERTs would have produced anyway.
    pub fn from_parquet(c: PqCompression) -> Self {
        match c {
            PqCompression::UNCOMPRESSED => Compression::None,
            PqCompression::SNAPPY => Compression::Snappy,
            PqCompression::GZIP(_) => Compression::Gzip,
            PqCompression::ZSTD(_) => Compression::Zstd,
            // Codecs we don't expose as INSERT options (LZ4, BROTLI, LZO, LZ4_RAW)
            // round-trip to None — the rewrite is still correct, only the
            // re-encoded blob loses the rare-codec choice. Acceptable v1.
            _ => Compression::None,
        }
    }
}

/// In-memory parquet writer. `finish` consumes the writer and returns the
/// finalized file bytes (footer written) ready for upload.
pub struct ParquetBatchWriter {
    inner: ArrowWriter<Vec<u8>>,
}

impl ParquetBatchWriter {
    /// Create a writer for `schema` using `compression`.
    pub fn new(schema: SchemaRef, compression: Compression) -> FdwResult<Self> {
        let props = WriterProperties::builder()
            .set_compression(compression.to_pq())
            .build();
        let inner = ArrowWriter::try_new(Vec::with_capacity(64 * 1024), schema, Some(props))?;
        Ok(Self { inner })
    }

    /// Append a `RecordBatch`.
    pub fn write(&mut self, batch: &RecordBatch) -> FdwResult<()> {
        self.inner.write(batch)?;
        Ok(())
    }

    /// Number of bytes the inner `ArrowWriter` has flushed to its underlying
    /// `Vec<u8>` so far. The INSERT/COPY path queries this after each batch
    /// flush to enforce `MAX_BLOB_BYTES` on the in-progress accumulator —
    /// without it, a multi-GiB `COPY ... FROM` accumulates parquet bytes
    /// unbounded and OOMs the backend long before `AzureBlobWriter::upload`
    /// gets a chance to refuse the body. Note this is a lower bound: row
    /// groups may still be in-memory pre-flush, but cumulative size grows
    /// monotonically.
    pub fn bytes_written(&self) -> usize {
        self.inner.bytes_written()
    }

    /// Finalize the file (writes the footer) and return the bytes.
    pub fn finish(self) -> FdwResult<Bytes> {
        // `ArrowWriter::into_inner` flushes outstanding data and the underlying
        // `SerializedFileWriter::into_inner` writes the parquet footer before
        // returning the owned buffer.
        let buf = self.inner.into_inner()?;
        Ok(Bytes::from(buf))
    }
}
