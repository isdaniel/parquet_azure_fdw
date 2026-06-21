#![forbid(unsafe_code)]
//! Async parquet `RecordBatch` stream construction with optional column
//! projection and row-group/page row filtering.
//!
//! Two entry points:
//!
//! - [`open_stream`] — production path; takes an
//!   [`AzureBlobReader`](crate::azure::AzureBlobReader) (an
//!   `AsyncFileReader` backed by Azure Blob Storage).
//! - [`open_local_stream`] — test/diagnostic path; reads a local file via
//!   `tokio::fs::File`. Used by unit tests to exercise projection + filter
//!   without spinning up Azurite.

use crate::azure::AzureBlobReader;
use crate::error::FdwResult;
use bytes::Bytes;
use parquet::arrow::arrow_reader::RowFilter;
use parquet::arrow::async_reader::{ParquetRecordBatchStream, ParquetRecordBatchStreamBuilder};
use parquet::arrow::ProjectionMask;
use std::path::Path;
use tokio::fs::File;

/// Options controlling how a parquet scan stream is built.
///
/// `projection` is a list of **top-level** column indices into the parquet
/// schema (mapped via `ProjectionMask::roots`). `row_filter` is an
/// already-constructed `RowFilter` — building it (predicate compilation,
/// projection mask for predicate columns) is the caller's job.
#[derive(Default)]
pub struct ParquetReadOptions {
    pub projection: Option<Vec<usize>>,
    pub row_filter: Option<RowFilter>,
}

/// Build an async `RecordBatch` stream over an Azure-backed parquet blob.
pub async fn open_stream(
    reader: AzureBlobReader,
    opts: ParquetReadOptions,
) -> FdwResult<ParquetRecordBatchStream<AzureBlobReader>> {
    let mut b = ParquetRecordBatchStreamBuilder::new(reader).await?;
    if let Some(cols) = opts.projection {
        let schema = b.parquet_schema().clone();
        b = b.with_projection(ProjectionMask::roots(&schema, cols));
    }
    if let Some(rf) = opts.row_filter {
        b = b.with_row_filter(rf);
    }
    Ok(b.build()?)
}

/// Build an async `RecordBatch` stream over a parquet file held entirely in
/// memory. Used by the UPDATE/DELETE rewrite kernel.
///
/// `parquet` v59 does not ship an `AsyncFileReader` impl for `Bytes` directly,
/// so we wrap in `std::io::Cursor<Bytes>` — which implements `AsyncRead +
/// AsyncSeek` via tokio and therefore satisfies `AsyncFileReader`.
pub async fn open_stream_from_bytes(
    bytes: Bytes,
    opts: ParquetReadOptions,
) -> FdwResult<ParquetRecordBatchStream<std::io::Cursor<Bytes>>> {
    let cursor = std::io::Cursor::new(bytes);
    let mut b = ParquetRecordBatchStreamBuilder::new(cursor).await?;
    if let Some(cols) = opts.projection {
        let schema = b.parquet_schema().clone();
        b = b.with_projection(ProjectionMask::roots(&schema, cols));
    }
    if let Some(rf) = opts.row_filter {
        b = b.with_row_filter(rf);
    }
    Ok(b.build()?)
}

/// Build an async `RecordBatch` stream over a local parquet file.
///
/// Always-available helper (not gated behind `cfg(test)`) — the surface is
/// tiny and useful for ad-hoc CLI debugging. Production scans go through
/// [`open_stream`].
pub async fn open_local_stream(
    path: &Path,
    opts: ParquetReadOptions,
) -> FdwResult<ParquetRecordBatchStream<File>> {
    let f = File::open(path).await?;
    let mut b = ParquetRecordBatchStreamBuilder::new(f).await?;
    if let Some(cols) = opts.projection {
        let schema = b.parquet_schema().clone();
        b = b.with_projection(ProjectionMask::roots(&schema, cols));
    }
    if let Some(rf) = opts.row_filter {
        b = b.with_row_filter(rf);
    }
    Ok(b.build()?)
}
