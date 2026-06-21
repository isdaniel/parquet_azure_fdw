//! In-memory, stateful Azure Blob Storage fake backed by `wiremock`.
//!
//! Replaces the Azurite test harness so the test suite runs without Docker.
//! The fake speaks just enough of the Azure Blob REST protocol for the FDW's
//! happy paths:
//!
//! - `GET /{container}/{blob}`        → download blob (body + `ETag` + status)
//! - `PUT /{container}/{blob}`        → upload (honors `If-Match` ⇒ 412)
//! - `DELETE /{container}/{blob}`     → delete (honors `If-Match` ⇒ 412/404)
//! - `GET /{container}?restype=container&comp=list[&prefix=...]`
//!   → XML `EnumerationResults` body
//!
//! Auth is ignored: any SAS-shaped URL works because the SDK passes the SAS
//! query string verbatim to the server and the server does not validate it.
//!
//! State is held in an `Arc<Mutex<State>>` shared between every route
//! handler and the public helpers (`put_blob`, `get_blob`, etc.). ETags
//! advance via a strictly-monotonic counter so the
//! "concurrent-overwrite ⇒ stale-etag" path is exercisable.

#![allow(dead_code)] // not every helper is consumed by every binary

use bytes::Bytes;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use tokio::runtime::Runtime;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

/// All state mutated by request handlers. Shared via `Arc<Mutex<_>>`.
struct State {
    /// Key: `"{container}/{blob}"`.
    blobs: HashMap<String, BlobEntry>,
    /// Strictly-monotonic counter feeding the ETag generator.
    next_etag_seq: u64,
}

struct BlobEntry {
    body: Bytes,
    /// Quoted, e.g. `"\"0x000000000000000A\""`.
    etag: String,
}

impl State {
    fn new() -> Self {
        Self {
            blobs: HashMap::new(),
            next_etag_seq: 1,
        }
    }

    fn mint_etag(&mut self) -> String {
        let seq = self.next_etag_seq;
        self.next_etag_seq = self
            .next_etag_seq
            .checked_add(1)
            .expect("etag counter overflow");
        format!("\"0x{seq:016X}\"")
    }
}

/// A running fake Azure Blob Storage server.
///
/// All `FakeBlobStore` instances in a process share a single dedicated
/// multi-thread tokio runtime (lazily started on first use). wiremock's
/// internal `MockServer` pool is a `once_cell::Lazy` keyed off of the
/// runtime that first touches it, so funneling everything through one
/// runtime keeps the pool consistent across tests. The shared runtime is
/// hosted on a dedicated worker thread so that callers running inside a
/// Postgres backend (no tokio context of their own) and callers running
/// under `#[tokio::test]` both work transparently.
pub struct FakeBlobStore {
    state: Arc<Mutex<State>>,
    #[allow(dead_code)] // kept alive for the lifetime of the fake
    server: MockServer,
    base_url: String,
}

impl FakeBlobStore {
    /// Start the fake on a random local port. Asynchronous variant.
    ///
    /// Boots on the shared background runtime regardless of which runtime
    /// the caller is using, so that `start` and `start_blocking` produce
    /// semantically-equivalent instances.
    pub async fn start() -> Self {
        Self::start_blocking()
    }

    /// Synchronous boot for tests that don't have an `async` context.
    ///
    /// Safe to call from both runtime-less threads (e.g. inside a Postgres
    /// backend) AND from inside an outer `#[tokio::test]` runtime — the
    /// boot itself runs on a dedicated OS thread so we never trip
    /// "Cannot start a runtime from within a runtime".
    pub fn start_blocking() -> Self {
        let handle = std::thread::spawn(|| {
            let rt = shared_runtime();
            rt.block_on(boot_server())
        });
        let (state, server, base_url) =
            handle.join().expect("fake-blob-store boot thread panicked");
        Self {
            state,
            server,
            base_url,
        }
    }

    /// Base URL of the running server, e.g. `http://127.0.0.1:39873`.
    pub fn base_url(&self) -> String {
        self.base_url.clone()
    }

    /// SAS-style URL suitable as the FDW `sas_url` user-mapping option:
    /// `"{base_url}/{container}?sv=fake"`.
    ///
    /// The fake ignores the query string; `sv=fake` is included so any
    /// downstream code that asserts the URL "looks like" a SAS URL is
    /// satisfied.
    pub fn sas_url(&self, container: &str) -> String {
        format!("{}/{}?sv=fake", self.base_url, container)
    }

    /// Seed a blob. Bumps the etag automatically. Returns the new etag.
    pub fn put_blob(&self, container: &str, blob: &str, body: Bytes) -> String {
        let key = format!("{container}/{blob}");
        let mut st = self.state.lock().expect("state mutex poisoned");
        let etag = st.mint_etag();
        st.blobs.insert(
            key,
            BlobEntry {
                body,
                etag: etag.clone(),
            },
        );
        etag
    }

    /// Read a blob's body. `None` if not present.
    pub fn get_blob(&self, container: &str, blob: &str) -> Option<Bytes> {
        let key = format!("{container}/{blob}");
        let st = self.state.lock().expect("state mutex poisoned");
        st.blobs.get(&key).map(|e| e.body.clone())
    }

    /// Current etag for a blob, in quoted form (`"\"0x...\""`). `None` if
    /// absent.
    pub fn read_etag(&self, container: &str, blob: &str) -> Option<String> {
        let key = format!("{container}/{blob}");
        let st = self.state.lock().expect("state mutex poisoned");
        st.blobs.get(&key).map(|e| e.etag.clone())
    }

    /// List blob names under `container`, optionally filtered by `prefix`.
    /// Names returned are the blob path within the container, NOT including
    /// the container segment.
    pub fn list_blobs(&self, container: &str, prefix: Option<&str>) -> Vec<String> {
        let st = self.state.lock().expect("state mutex poisoned");
        let container_prefix = format!("{container}/");
        let mut out: Vec<String> = st
            .blobs
            .keys()
            .filter_map(|k| k.strip_prefix(&container_prefix))
            .filter(|name| match prefix {
                Some(p) => name.starts_with(p),
                None => true,
            })
            .map(str::to_string)
            .collect();
        out.sort();
        out
    }
}

fn build_runtime() -> Runtime {
    // Spawn the builder on a fresh OS thread so we don't inherit a tokio
    // context from the caller (which trips
    // "Cannot start a runtime from within a runtime" when this is invoked
    // under `#[tokio::test]`).
    std::thread::spawn(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("fake-blob-store")
            .build()
            .expect("build multi-thread runtime for FakeBlobStore")
    })
    .join()
    .expect("fake-blob-store runtime builder thread panicked")
}

/// Shared, lazily-initialised multi-thread tokio runtime used by every
/// `FakeBlobStore` in the process. A `&'static Runtime` lets `block_on` be
/// called from any thread, including callers without a runtime of their
/// own and callers running inside an outer `#[tokio::test]` runtime.
fn shared_runtime() -> &'static Runtime {
    static SHARED_RT: OnceLock<Runtime> = OnceLock::new();
    SHARED_RT.get_or_init(build_runtime)
}

async fn boot_server() -> (Arc<Mutex<State>>, MockServer, String) {
    let state = Arc::new(Mutex::new(State::new()));
    let server = MockServer::start().await;
    let base_url = server.uri();

    // ----- GET: blob download (full or ranged) OR container list -----
    Mock::given(method("GET"))
        .respond_with(GetResponder {
            state: state.clone(),
        })
        .mount(&server)
        .await;

    // ----- HEAD: blob properties (Content-Length + ETag) -----
    Mock::given(method("HEAD"))
        .respond_with(HeadResponder {
            state: state.clone(),
        })
        .mount(&server)
        .await;

    // ----- PUT: blob upload (honors If-Match) -----
    Mock::given(method("PUT"))
        .respond_with(PutResponder {
            state: state.clone(),
        })
        .mount(&server)
        .await;

    // ----- DELETE: blob delete (honors If-Match) -----
    Mock::given(method("DELETE"))
        .respond_with(DeleteResponder {
            state: state.clone(),
        })
        .mount(&server)
        .await;

    (state, server, base_url)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse `"/{container}/{blob...}"` out of a request URL.
///
/// The SDK's `BlobContainerClient::blob_client(name)` appends the full blob
/// name as a single path segment, percent-encoding the `/` inside as
/// `%2F`. To make the blob name we recover match the form callers seeded
/// via [`FakeBlobStore::put_blob`] (`"events/a.parquet"`), we percent-
/// decode each path segment after the container. Returns `None` for
/// paths that have only the container segment (container-level
/// operations, e.g. list-blobs).
fn split_container_blob(url: &url::Url) -> Option<(String, String)> {
    let raw: Vec<&str> = url
        .path_segments()
        .map(|s| s.filter(|seg| !seg.is_empty()).collect())
        .unwrap_or_default();
    if raw.len() < 2 {
        return None;
    }
    let container = percent_decode(raw[0]);
    let blob_segments: Vec<String> = raw[1..].iter().map(|s| percent_decode(s)).collect();
    let blob = blob_segments.join("/");
    Some((container, blob))
}

/// Minimal percent-decoder for path segments. Returns the input unchanged
/// for any unparseable sequence (we'd rather generate a 404 than panic).
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = (bytes[i + 1] as char).to_digit(16);
            let lo = (bytes[i + 2] as char).to_digit(16);
            if let (Some(h), Some(l)) = (hi, lo) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| s.to_string())
}

/// Extract the value of the first `If-Match` header, if any.
fn if_match_header(req: &Request) -> Option<String> {
    req.headers
        .get("If-Match")
        .and_then(|v| v.to_str().ok().map(str::to_string))
}

/// Parse an Azure-style `Range: bytes=START-END` (or `bytes=START-`) header
/// value into an inclusive `[start, end]` byte range. Returns `None` for any
/// unparseable input — the caller falls back to returning the full body.
fn parse_range_header(req: &Request, body_len: usize) -> Option<(usize, usize)> {
    let raw = req
        .headers
        .get("x-ms-range")
        .or_else(|| req.headers.get("Range"))?
        .to_str()
        .ok()?;
    let spec = raw.strip_prefix("bytes=")?;
    let (start_s, end_s) = spec.split_once('-')?;
    let start: usize = start_s.trim().parse().ok()?;
    let end: usize = if end_s.trim().is_empty() {
        body_len.saturating_sub(1)
    } else {
        end_s.trim().parse().ok()?
    };
    let end = end.min(body_len.saturating_sub(1));
    if start > end {
        return None;
    }
    Some((start, end))
}

/// Format `Last-Modified` RFC-1123 for the current instant. Static-looking
/// is fine — Azure clients only parse, not compare, this field.
fn last_modified_now() -> String {
    // chrono's RFC2822 output ("%a, %d %b %Y %H:%M:%S +0000") matches the
    // RFC1123 wire format Azure uses, except we want "GMT" at the tail.
    let now = chrono::Utc::now();
    now.format("%a, %d %b %Y %H:%M:%S GMT").to_string()
}

// ---------------------------------------------------------------------------
// Responders
// ---------------------------------------------------------------------------

struct GetResponder {
    state: Arc<Mutex<State>>,
}

impl Respond for GetResponder {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        // Two GET shapes:
        //   1. blob download:  /{container}/{blob...}    (no comp=list)
        //   2. container list: /{container}?restype=container&comp=list
        let is_list = req
            .url
            .query_pairs()
            .any(|(k, v)| k == "comp" && v == "list");

        if is_list {
            return self.respond_list(req);
        }

        let Some((container, blob)) = split_container_blob(&req.url) else {
            return ResponseTemplate::new(400).set_body_string("bad path");
        };
        let key = format!("{container}/{blob}");
        let st = self.state.lock().expect("state mutex poisoned");
        let Some(entry) = st.blobs.get(&key) else {
            return ResponseTemplate::new(404);
        };

        // Honor `Range` / `x-ms-range` for partial downloads. The parquet
        // reader issues ranged GETs against the footer and column chunks;
        // the Azure SDK's partitioned download also relies on these.
        let total_len = entry.body.len();
        if let Some((start, end)) = parse_range_header(req, total_len) {
            let slice = entry.body.slice(start..=end);
            let content_range = format!("bytes {start}-{end}/{total_len}");
            return ResponseTemplate::new(206)
                .insert_header("ETag", entry.etag.as_str())
                .insert_header("Last-Modified", last_modified_now().as_str())
                .insert_header("Content-Length", slice.len().to_string().as_str())
                .insert_header("Content-Range", content_range.as_str())
                .insert_header("x-ms-version", "2022-11-02")
                .insert_header("x-ms-blob-type", "BlockBlob")
                .set_body_bytes(slice.to_vec());
        }

        ResponseTemplate::new(200)
            .insert_header("ETag", entry.etag.as_str())
            .insert_header("Last-Modified", last_modified_now().as_str())
            .insert_header("Content-Length", total_len.to_string().as_str())
            .insert_header("x-ms-version", "2022-11-02")
            .insert_header("x-ms-blob-type", "BlockBlob")
            .set_body_bytes(entry.body.to_vec())
    }
}

struct HeadResponder {
    state: Arc<Mutex<State>>,
}

impl Respond for HeadResponder {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        let Some((container, blob)) = split_container_blob(&req.url) else {
            return ResponseTemplate::new(400);
        };
        let key = format!("{container}/{blob}");
        let st = self.state.lock().expect("state mutex poisoned");
        match st.blobs.get(&key) {
            None => ResponseTemplate::new(404),
            Some(entry) => ResponseTemplate::new(200)
                .insert_header("ETag", entry.etag.as_str())
                .insert_header("Last-Modified", last_modified_now().as_str())
                .insert_header("Content-Length", entry.body.len().to_string().as_str())
                .insert_header("x-ms-version", "2022-11-02")
                .insert_header("x-ms-blob-type", "BlockBlob")
                .insert_header("Accept-Ranges", "bytes"),
        }
    }
}

impl GetResponder {
    fn respond_list(&self, req: &Request) -> ResponseTemplate {
        // Container name is the first path segment.
        let segments: Vec<String> = req
            .url
            .path_segments()
            .map(|s| {
                s.filter(|seg| !seg.is_empty())
                    .map(percent_decode)
                    .collect()
            })
            .unwrap_or_default();
        let Some(container) = segments.first() else {
            return ResponseTemplate::new(400).set_body_string("missing container");
        };
        let prefix: Option<String> = req
            .url
            .query_pairs()
            .find_map(|(k, v)| (k == "prefix").then(|| v.into_owned()));

        let st = self.state.lock().expect("state mutex poisoned");
        let container_prefix = format!("{container}/");
        let mut names_etags: Vec<(String, String)> = st
            .blobs
            .iter()
            .filter_map(|(k, e)| {
                k.strip_prefix(&container_prefix)
                    .map(|n| (n.to_string(), e.etag.clone()))
            })
            .filter(|(name, _)| match prefix.as_deref() {
                Some(p) => name.starts_with(p),
                None => true,
            })
            .collect();
        names_etags.sort_by(|a, b| a.0.cmp(&b.0));

        let mut xml = String::with_capacity(256 + names_etags.len() * 96);
        xml.push_str(r#"<?xml version="1.0" encoding="utf-8"?>"#);
        xml.push_str(&format!(
            r#"<EnumerationResults ServiceEndpoint="http://127.0.0.1/" ContainerName="{container}">"#
        ));
        if let Some(p) = prefix.as_deref() {
            xml.push_str(&format!("<Prefix>{}</Prefix>", xml_escape(p)));
        }
        xml.push_str("<Blobs>");
        for (name, etag) in &names_etags {
            // The SDK's `BlobItem` deserializer accepts either a bare
            // `<Name>foo</Name>` text node OR a `<Name Encoded="false">foo</Name>`
            // structured form. Plain text is fine. We additionally emit a
            // `<Properties><Etag>...</Etag></Properties>` block so the scan
            // path can capture per-blob etags from listing alone (no extra
            // HEAD), matching real Azure list responses.
            xml.push_str("<Blob><Name>");
            xml.push_str(&xml_escape(name));
            xml.push_str("</Name><Properties><Etag>");
            xml.push_str(&xml_escape(etag));
            xml.push_str("</Etag></Properties></Blob>");
        }
        xml.push_str("</Blobs><NextMarker/></EnumerationResults>");

        ResponseTemplate::new(200)
            .insert_header("Content-Type", "application/xml")
            .insert_header("Content-Length", xml.len().to_string().as_str())
            .insert_header("x-ms-version", "2022-11-02")
            .set_body_string(xml)
    }
}

fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

struct PutResponder {
    state: Arc<Mutex<State>>,
}

impl Respond for PutResponder {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        let Some((container, blob)) = split_container_blob(&req.url) else {
            return ResponseTemplate::new(400).set_body_string("bad path");
        };
        let key = format!("{container}/{blob}");

        let mut st = self.state.lock().expect("state mutex poisoned");

        // Honour If-Match for concurrency-safe replace.
        if let Some(want) = if_match_header(req) {
            match st.blobs.get(&key) {
                None => return ResponseTemplate::new(412),
                Some(entry) if entry.etag != want => return ResponseTemplate::new(412),
                Some(_) => {}
            }
        }

        let new_etag = st.mint_etag();
        st.blobs.insert(
            key,
            BlobEntry {
                body: Bytes::from(req.body.clone()),
                etag: new_etag.clone(),
            },
        );

        ResponseTemplate::new(201)
            .insert_header("ETag", new_etag.as_str())
            .insert_header("Last-Modified", last_modified_now().as_str())
            .insert_header("x-ms-version", "2022-11-02")
            .insert_header("x-ms-request-server-encrypted", "true")
    }
}

struct DeleteResponder {
    state: Arc<Mutex<State>>,
}

impl Respond for DeleteResponder {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        let Some((container, blob)) = split_container_blob(&req.url) else {
            return ResponseTemplate::new(400).set_body_string("bad path");
        };
        let key = format!("{container}/{blob}");

        let mut st = self.state.lock().expect("state mutex poisoned");
        match st.blobs.get(&key) {
            None => ResponseTemplate::new(404),
            Some(entry) => {
                if let Some(want) = if_match_header(req) {
                    if entry.etag != want {
                        return ResponseTemplate::new(412);
                    }
                }
                st.blobs.remove(&key);
                ResponseTemplate::new(202).insert_header("x-ms-version", "2022-11-02")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// (end of file)
// ---------------------------------------------------------------------------
