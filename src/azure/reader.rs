#![forbid(unsafe_code)]
//! `AsyncFileReader` over Azure Blob Storage range gets.
//!
//! The parquet async reader needs two things from us:
//!
//! 1. `get_bytes(range)` — issue an HTTP Range GET and return the body.
//! 2. `get_metadata()` — fetch the parquet footer (and column/offset indexes
//!    if requested). We do a single HEAD (`get_properties`) to learn the
//!    blob size, cache it, then hand `&mut self` to
//!    `ParquetMetaDataReader::load_and_finish`; the blanket
//!    `impl<T: AsyncFileReader> MetadataFetch for &mut T` routes all
//!    subsequent footer/page-index reads through our `get_bytes`.

use azure_storage_blob::clients::BlobClient;
use bytes::Bytes;
use futures::future::{BoxFuture, FutureExt};
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::errors::ParquetError;
use parquet::file::metadata::{ParquetMetaData, ParquetMetaDataReader};
use std::ops::Range;
use std::sync::Arc;

/// Async parquet reader backed by a single blob in Azure Storage.
///
/// `Arc<BlobClient>` is cheap; we hold it shared so `open_blob` can hand out
/// readers without cloning the underlying HTTP pipeline.
pub struct AzureBlobReader {
    client: Arc<BlobClient>,
    cached_size: Option<u64>,
}

impl AzureBlobReader {
    pub fn new(client: Arc<BlobClient>) -> Self {
        Self {
            client,
            cached_size: None,
        }
    }

    /// HEAD the blob and cache its size for footer math.
    async fn fetch_size(&mut self) -> parquet::errors::Result<u64> {
        if let Some(s) = self.cached_size {
            return Ok(s);
        }
        use azure_storage_blob::models::BlobClientGetPropertiesResultHeaders;
        let resp = self
            .client
            .get_properties(None)
            .await
            .map_err(to_parquet_err)?;
        let len = resp
            .content_length()
            .map_err(to_parquet_err)?
            .ok_or_else(|| ParquetError::General("blob has no Content-Length header".into()))?;
        self.cached_size = Some(len);
        Ok(len)
    }
}

/// Map any error type into the `ParquetError::External` carrier the parquet
/// async reader expects.
fn to_parquet_err<E: std::fmt::Display>(e: E) -> ParquetError {
    ParquetError::External(Box::new(std::io::Error::other(e.to_string())))
}

impl AsyncFileReader for AzureBlobReader {
    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
        use azure_storage_blob::models::{BlobClientDownloadOptions, HttpRange};

        let client = self.client.clone();
        async move {
            let opts = BlobClientDownloadOptions {
                range: Some(HttpRange::from(range)),
                ..Default::default()
            };
            let resp = client.download(Some(opts)).await.map_err(to_parquet_err)?;
            resp.body.collect().await.map_err(to_parquet_err)
        }
        .boxed()
    }

    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, parquet::errors::Result<Arc<ParquetMetaData>>> {
        async move {
            let size = self.fetch_size().await?;

            // Build the reader, threading caller-supplied policies through
            // where present (matches the parquet crate's default `File` impl).
            let metadata_opts = options.map(|o| o.metadata_options().clone());
            let mut metadata_reader =
                ParquetMetaDataReader::new().with_metadata_options(metadata_opts);
            if let Some(opts) = options {
                metadata_reader = metadata_reader
                    .with_column_index_policy(opts.column_index_policy())
                    .with_offset_index_policy(opts.offset_index_policy());
            }
            // The blanket `MetadataFetch for &mut T: AsyncFileReader` routes
            // the loader's range gets back through our `get_bytes`.
            let md = metadata_reader.load_and_finish(self, size).await?;
            Ok(Arc::new(md))
        }
        .boxed()
    }
}
