#![forbid(unsafe_code)]
//! Single-shot block-blob uploader.
//!
//! The writer wraps an [`azure_storage_blob::clients::BlobClient`] and exposes
//! a one-call `upload` that puts the supplied bytes as a block blob. This
//! matches the parquet write path: we buffer the encoded parquet file into a
//! `Bytes` in memory, then upload once.
//!
//! ## API note
//!
//! In `azure_storage_blob` 1.0 [`BlobClient::upload`] takes only the request
//! body and an options struct — the content length is derived from the body
//! and the SDK overwrites by default. Callers that need fail-on-exists
//! semantics must populate `BlobClientUploadOptions::if_none_match` (or use
//! `BlockBlobClient` directly); for this FDW [`generate_blob_name`] yields a
//! UUID-suffixed name per call, so collisions are effectively impossible and
//! the default overwrite behavior is harmless.

use crate::azure::{AzureBlobClient, MAX_BLOB_BYTES};
use crate::error::{FdwError, FdwResult};
use bytes::Bytes;
use std::sync::Arc;

/// A handle to a not-yet-uploaded block blob.
///
/// `name` is the blob path within the container (no leading `/`).
pub struct AzureBlobWriter {
    client: Arc<azure_storage_blob::clients::BlobClient>,
    name: String,
}

impl AzureBlobWriter {
    /// Build a writer for `name` inside `container`. Network is not touched
    /// until [`upload`](Self::upload) is awaited.
    pub fn new(container: &AzureBlobClient, name: &str) -> Self {
        let client = Arc::new(container.inner().blob_client(name));
        Self {
            client,
            name: name.to_string(),
        }
    }

    /// Blob path within the container.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// PUT the entire body as a single block blob. Consumes `self` because the
    /// writer represents a one-shot upload.
    ///
    /// Enforces [`MAX_BLOB_BYTES`] symmetrically with the read path: a single
    /// INSERT / COPY that buffered more than the cap is rejected before we
    /// pay the network round-trip, matching the bound the UPDATE/DELETE
    /// rewrite path can later re-read.
    pub async fn upload(self, body: Bytes) -> FdwResult<()> {
        if body.len() as u64 > MAX_BLOB_BYTES {
            return Err(FdwError::Azure(format!(
                "refusing to upload blob '{}': {} bytes exceeds MAX_BLOB_BYTES={MAX_BLOB_BYTES}",
                self.name,
                body.len()
            )));
        }
        self.client
            .upload(body.into(), None)
            .await
            .map_err(FdwError::azure)?;
        Ok(())
    }
}

/// Build a deterministic-shape, collision-free blob name.
///
/// Format: `{prefix}/{utc_iso8601}-{uuid_v4}.parquet`. Trailing slashes on
/// `prefix` are normalized away; an empty prefix yields a name with no
/// leading slash. The timestamp gives humans a chronological ordering when
/// listing; the UUID guarantees uniqueness even under sub-millisecond
/// concurrent inserts.
pub fn generate_blob_name(prefix: &str) -> String {
    let ts = chrono::Utc::now().format("%Y%m%dT%H%M%S%.3fZ");
    let id = uuid::Uuid::new_v4();
    let trimmed = prefix.trim_end_matches('/');
    if trimmed.is_empty() {
        format!("{ts}-{id}.parquet")
    } else {
        format!("{trimmed}/{ts}-{id}.parquet")
    }
}
