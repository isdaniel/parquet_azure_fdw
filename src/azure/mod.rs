#![forbid(unsafe_code)]
//! Azure Blob Storage facade: a thin, Arc-cloneable wrapper around a
//! `BlobContainerClient` plus a parquet-friendly `AsyncFileReader`.
//!
//! Construction matches the 2-variant [`auth::Credential`]:
//!
//! - `Token(Arc<dyn TokenCredential>)` — used for Managed Identity and
//!   AAD service principals.
//! - `SasUrl { container_url }` — the URL itself is the full container
//!   endpoint (with `?sv=...` query), so the container client is built
//!   directly from it with no credential.
//!
//! Shared-key (account_key) auth is not supported because
//! `azure_storage_blob` 1.0 has no shared-key authorization policy.
//! `parse_auth_method` rejects `account_key` up front with a clear message
//! telling users to generate a SAS instead.

pub mod auth;
pub mod reader;
pub mod writer;

use crate::error::{FdwError, FdwResult};
use azure_storage_blob::clients::BlobContainerClient;
use std::sync::Arc;

pub use auth::{build_credential, parse_auth_method, AuthMethod, Credential};
pub use reader::AzureBlobReader;
pub use writer::{generate_blob_name, AzureBlobWriter};

/// Soft cap on the number of blob names returned by `list_with_prefix`. The
/// modify/scan paths buffer this whole list in RAM; allowing it to grow without
/// bound lets a (malicious or accidental) huge container OOM the backend. 100k
/// entries ≈ a few MB of `String`, which is comfortable while still rejecting
/// runaway listings before they hurt.
pub const MAX_LIST_RESULTS: usize = 100_000;

/// Hard cap on the size of a single blob loaded into RAM by `get_with_etag`.
/// The UPDATE/DELETE rewrite path reads the entire blob into memory AND then
/// the pure kernel materialises a concat (original + per-column rebuild +
/// final), so peak RSS per modify statement is roughly **3× the decoded
/// blob size** — not the encoded parquet size. Until the kernel goes
/// streaming (see `fdw/modify/kernel.rs` doc), we keep this cap deliberately
/// well below the Postgres `MaxAllocSize` (~1 GiB) so a worst-case rewrite
/// of a highly-compressible parquet blob can't OOM the backend. Users with
/// blobs larger than this should split them at INSERT time.
pub const MAX_BLOB_BYTES: u64 = 512 * 1024 * 1024;

/// True if a blob name matches the staging convention used by the
/// UPDATE/DELETE coordinator (`make_staging_name`): contains the `.tmp.`
/// infix. These belong to in-flight or aborted statements and must not
/// be visible to external SELECTs.
fn is_staging_name(name: &str) -> bool {
    name.contains(".tmp.")
}

/// Container-scoped Azure Blob Storage client.
///
/// Cheap to clone (`Arc` inside); pass around freely between the scan and
/// modify code paths.
#[derive(Clone)]
pub struct AzureBlobClient {
    inner: Arc<BlobContainerClient>,
    account: String,
    container: String,
}

impl AzureBlobClient {
    /// Build a container client.
    ///
    /// `endpoint` is the storage host suffix, e.g. `blob.core.windows.net`
    /// for the public cloud or `127.0.0.1:10000` for Azurite. `account` is
    /// the storage account name. `container` is the container name. The
    /// final URL is `https://{account}.{endpoint}/{container}`, except for
    /// `Credential::SasUrl` where the supplied URL is used as-is.
    pub fn new(
        endpoint: &str,
        account: &str,
        cred: Credential,
        container: &str,
    ) -> FdwResult<Self> {
        use azure_core::http::Url;

        let inner = match cred {
            Credential::Token(tc) => {
                let url_str = format!("https://{account}.{endpoint}/{container}");
                let url = Url::parse(&url_str).map_err(|e| {
                    FdwError::azure_ctx(&format!("bad container url '{url_str}'"), e)
                })?;
                BlobContainerClient::new(url, Some(tc), None).map_err(FdwError::azure)?
            }
            Credential::SasUrl { container_url } => {
                let url = Url::parse(&container_url)
                    .map_err(|e| FdwError::azure_ctx("bad sas container url", e))?;
                BlobContainerClient::new(url, None, None).map_err(FdwError::azure)?
            }
        };

        Ok(Self {
            inner: Arc::new(inner),
            account: account.to_string(),
            container: container.to_string(),
        })
    }

    /// List blob names under `prefix` (server-side prefix filter).
    ///
    /// Iterates all pages. Globbing beyond a simple prefix is the caller's
    /// job — Azure list only supports prefix (+ delimiter, not used here).
    pub async fn list_with_prefix(&self, prefix: &str) -> FdwResult<Vec<String>> {
        Ok(self
            .list_with_prefix_etags(prefix)
            .await?
            .into_iter()
            .map(|(n, _)| n)
            .collect())
    }

    /// Like [`list_with_prefix`](Self::list_with_prefix) but also returns each
    /// blob's current etag. Used by the scan path so the modify path can
    /// later precondition GET/PUT on the *scan-time* etag — preventing
    /// silent corruption from concurrent writers between SELECT and UPDATE.
    pub async fn list_with_prefix_etags(&self, prefix: &str) -> FdwResult<Vec<(String, String)>> {
        use azure_storage_blob::models::BlobContainerClientListBlobsOptions;
        use futures::TryStreamExt;

        let opts = BlobContainerClientListBlobsOptions {
            prefix: Some(prefix.to_string()),
            ..Default::default()
        };
        let mut pager = self.inner.list_blobs(Some(opts)).map_err(FdwError::azure)?;

        let mut out = Vec::new();
        while let Some(item) = pager.try_next().await.map_err(FdwError::azure)? {
            let Some(name) = item.name else { continue };
            // Drop staging blobs (`*.tmp.<uuid>.parquet`) — they belong to
            // in-flight or aborted UPDATE/DELETE statements and are not
            // externally visible data. Filter BEFORE the MAX_LIST_RESULTS
            // check so a flood of leaked staging blobs cannot exhaust the
            // listing budget.
            if is_staging_name(&name) {
                continue;
            }
            if out.len() >= MAX_LIST_RESULTS {
                return Err(FdwError::Azure(format!(
                    "list_with_prefix exceeded MAX_LIST_RESULTS={MAX_LIST_RESULTS}; \
                     narrow the prefix or split the container"
                )));
            }
            // Listings missing an etag are surfaced as an error — without an
            // etag the modify path can't precondition writes and we'd lose the
            // lost-update guarantee.
            let etag = item
                .properties
                .as_ref()
                .and_then(|p| p.etag.clone())
                .ok_or_else(|| {
                    FdwError::Azure(format!("list response for '{name}' missing etag"))
                })?;
            out.push((name, etag.to_string()));
        }
        Ok(out)
    }

    /// HEAD a single blob and return its current etag. Used by the scan path
    /// for the non-glob (single blob) case where no listing is performed.
    pub async fn head_etag(&self, blob: &str) -> FdwResult<String> {
        let bc = self.inner.blob_client(blob);
        let resp = bc.get_properties(None).await.map_err(|e| {
            if e.http_status() == Some(azure_core::http::StatusCode::NotFound) {
                FdwError::ConcurrentUpdate {
                    blob: blob.to_string(),
                    reason: "blob disappeared".into(),
                }
            } else {
                FdwError::azure(e)
            }
        })?;
        use azure_storage_blob::models::BlobClientGetPropertiesResultHeaders;
        let etag = resp
            .etag()
            .map_err(FdwError::azure)?
            .ok_or_else(|| FdwError::Azure(format!("HEAD '{blob}' missing etag")))?;
        Ok(etag.to_string())
    }

    /// Construct an `AsyncFileReader` for a single blob within this container.
    pub fn open_blob(&self, blob: &str) -> AzureBlobReader {
        let blob_client = self.inner.blob_client(blob);
        AzureBlobReader::new(Arc::new(blob_client))
    }

    pub fn container(&self) -> &str {
        &self.container
    }

    pub fn account(&self) -> &str {
        &self.account
    }

    /// Underlying SDK client, for callers that need the raw API.
    pub fn inner(&self) -> &BlobContainerClient {
        &self.inner
    }

    /// GET the blob body, preconditioned on `if_match` being the current
    /// etag. 412 (etag mismatch) and 404 both surface as
    /// `ConcurrentUpdate` — both mean the blob has changed since the scan
    /// captured this etag, so the row identifiers we're about to write are
    /// no longer valid.
    ///
    /// This is the function the UPDATE/DELETE rewrite path uses: passing the
    /// *scan-time* etag (captured at LIST time during BeginForeignScan) makes
    /// the download itself the lost-update guard, instead of a separate
    /// post-hoc check.
    pub async fn get_body_if_match(&self, blob: &str, if_match: &str) -> FdwResult<bytes::Bytes> {
        use azure_storage_blob::models::BlobClientDownloadOptions;
        let bc = self.inner.blob_client(blob);
        let opts = BlobClientDownloadOptions {
            if_match: Some(if_match.to_string().into()),
            ..Default::default()
        };
        let resp = bc.download(Some(opts)).await.map_err(|e| {
            let code = e.http_status();
            if code == Some(azure_core::http::StatusCode::PreconditionFailed) {
                FdwError::ConcurrentUpdate {
                    blob: blob.to_string(),
                    reason: "blob changed since SELECT (etag mismatch on GET)".into(),
                }
            } else if code == Some(azure_core::http::StatusCode::NotFound) {
                FdwError::ConcurrentUpdate {
                    blob: blob.to_string(),
                    reason: "blob disappeared".into(),
                }
            } else {
                FdwError::azure(e)
            }
        })?;
        if let Some(len) = resp.properties.content_length {
            if len > MAX_BLOB_BYTES {
                return Err(FdwError::Azure(format!(
                    "blob '{blob}' is {len} bytes, exceeds MAX_BLOB_BYTES={MAX_BLOB_BYTES}"
                )));
            }
        }
        let body = resp.body.collect().await.map_err(FdwError::azure)?;
        if body.len() as u64 > MAX_BLOB_BYTES {
            return Err(FdwError::Azure(format!(
                "blob '{blob}' body is {} bytes, exceeds MAX_BLOB_BYTES={MAX_BLOB_BYTES}",
                body.len()
            )));
        }
        Ok(body)
    }

    /// GET the blob body and return `(bytes, etag)`. 404 → ConcurrentUpdate.
    ///
    /// Refuses bodies larger than [`MAX_BLOB_BYTES`] up front (via the
    /// response's `Content-Length`) to bound peak RSS in the modify path.
    pub async fn get_with_etag(&self, blob: &str) -> FdwResult<(bytes::Bytes, String)> {
        let bc = self.inner.blob_client(blob);
        let resp = bc.download(None).await.map_err(|e| {
            if e.http_status() == Some(azure_core::http::StatusCode::NotFound) {
                FdwError::ConcurrentUpdate {
                    blob: blob.to_string(),
                    reason: "blob disappeared".into(),
                }
            } else {
                FdwError::azure(e)
            }
        })?;
        if let Some(len) = resp.properties.content_length {
            if len > MAX_BLOB_BYTES {
                return Err(FdwError::Azure(format!(
                    "blob '{blob}' is {len} bytes, exceeds MAX_BLOB_BYTES={MAX_BLOB_BYTES}; \
                     refusing to buffer in memory"
                )));
            }
        }
        let etag = resp
            .properties
            .etag
            .clone()
            .ok_or_else(|| FdwError::Azure("response missing ETag".into()))?;
        let body = resp.body.collect().await.map_err(FdwError::azure)?;
        // Belt-and-braces: enforce the cap on the actual body length too, in
        // case the server omitted Content-Length or lied about it.
        if body.len() as u64 > MAX_BLOB_BYTES {
            return Err(FdwError::Azure(format!(
                "blob '{blob}' body is {} bytes, exceeds MAX_BLOB_BYTES={MAX_BLOB_BYTES}",
                body.len()
            )));
        }
        Ok((body, etag.to_string()))
    }

    /// PUT the blob with an `If-Match` precondition. 412 → ConcurrentUpdate.
    pub async fn put_if_match(
        &self,
        blob: &str,
        body: bytes::Bytes,
        etag: &str,
    ) -> FdwResult<String> {
        use azure_storage_blob::models::BlobClientUploadOptions;
        if body.len() as u64 > MAX_BLOB_BYTES {
            return Err(FdwError::Azure(format!(
                "refusing to upload blob '{blob}': {} bytes exceeds MAX_BLOB_BYTES={MAX_BLOB_BYTES}",
                body.len()
            )));
        }
        let bc = self.inner.blob_client(blob);
        let opts = BlobClientUploadOptions {
            if_match: Some(etag.to_string().into()),
            ..Default::default()
        };
        let resp = bc.upload(body.into(), Some(opts)).await.map_err(|e| {
            if e.http_status() == Some(azure_core::http::StatusCode::PreconditionFailed) {
                FdwError::ConcurrentUpdate {
                    blob: blob.to_string(),
                    reason: "etag mismatch".into(),
                }
            } else {
                FdwError::azure(e)
            }
        })?;
        let new_etag = resp
            .etag
            .clone()
            .ok_or_else(|| FdwError::Azure("upload response missing ETag".into()))?;
        Ok(new_etag.to_string())
    }

    /// PUT the blob with `If-None-Match: *` (create-only). 412 → ConcurrentUpdate
    /// (a blob with this name already exists). Used by the staging path so a
    /// uuid collision is surfaced rather than silently overwriting.
    pub async fn put_if_none_match(&self, blob: &str, body: bytes::Bytes) -> FdwResult<String> {
        use azure_storage_blob::models::BlobClientUploadOptions;
        if body.len() as u64 > MAX_BLOB_BYTES {
            return Err(FdwError::Azure(format!(
                "refusing to upload blob '{blob}': {} bytes exceeds MAX_BLOB_BYTES={MAX_BLOB_BYTES}",
                body.len()
            )));
        }
        let bc = self.inner.blob_client(blob);
        let opts = BlobClientUploadOptions {
            if_none_match: Some("*".to_string().into()),
            ..Default::default()
        };
        let resp = bc.upload(body.into(), Some(opts)).await.map_err(|e| {
            if e.http_status() == Some(azure_core::http::StatusCode::PreconditionFailed) {
                FdwError::ConcurrentUpdate {
                    blob: blob.to_string(),
                    reason: "staging-name collision (If-None-Match)".into(),
                }
            } else {
                FdwError::azure(e)
            }
        })?;
        let new_etag = resp
            .etag
            .clone()
            .ok_or_else(|| FdwError::Azure("upload response missing ETag".into()))?;
        Ok(new_etag.to_string())
    }

    /// DELETE the blob with `If-Match`. 412 → ConcurrentUpdate (etag), 404 →
    /// ConcurrentUpdate (disappeared) — both surfaced because the FDW's
    /// "blob is gone" recovery is to bail on the statement, not silently
    /// pretend the delete succeeded.
    pub async fn delete_if_match(&self, blob: &str, etag: &str) -> FdwResult<()> {
        use azure_storage_blob::models::BlobClientDeleteOptions;
        let bc = self.inner.blob_client(blob);
        let opts = BlobClientDeleteOptions {
            if_match: Some(etag.to_string().into()),
            ..Default::default()
        };
        bc.delete(Some(opts)).await.map_err(|e| {
            let code = e.http_status();
            if code == Some(azure_core::http::StatusCode::PreconditionFailed) {
                FdwError::ConcurrentUpdate {
                    blob: blob.to_string(),
                    reason: "etag mismatch".into(),
                }
            } else if code == Some(azure_core::http::StatusCode::NotFound) {
                FdwError::ConcurrentUpdate {
                    blob: blob.to_string(),
                    reason: "blob disappeared".into(),
                }
            } else {
                FdwError::azure(e)
            }
        })?;
        Ok(())
    }

    /// Best-effort DELETE without an etag precondition. 404 swallowed (the
    /// goal is "make it gone"). Used by the xact-abort cleanup hook, which
    /// cannot raise without crashing the backend.
    pub async fn delete_unconditional(&self, blob: &str) -> FdwResult<()> {
        let bc = self.inner.blob_client(blob);
        match bc.delete(None).await {
            Ok(_) => Ok(()),
            Err(e) if e.http_status() == Some(azure_core::http::StatusCode::NotFound) => Ok(()),
            Err(e) => Err(FdwError::azure(e)),
        }
    }
}
