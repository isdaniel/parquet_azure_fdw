#![forbid(unsafe_code)]
//! Pure-Rust parsers and validators for SERVER, USER MAPPING, and FOREIGN TABLE
//! options. Operates on `&[(&str, &str)]` so the logic is testable without a
//! running Postgres. Task 14 will adapt pgrx's `PgList<DefElem>` into a slice
//! and call these from the SQL validator entry point.

use crate::azure::{parse_auth_method, AuthMethod};
use crate::error::{FdwError, FdwResult};
use crate::parquet_io::Compression;
use pgrx::pg_sys;

const DEFAULT_ENDPOINT: &str = "blob.core.windows.net";

/// Partition-key column types supported in v1. See SP-3b spec §4 for the
/// rationale on the limited set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PgPartitionType {
    Int2,
    Int4,
    Int8,
    Text,
    Date,
}

impl PgPartitionType {
    pub fn parse(s: &str) -> FdwResult<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "int2" | "smallint" => Self::Int2,
            "int4" | "integer" | "int" => Self::Int4,
            "int8" | "bigint" => Self::Int8,
            "text" => Self::Text,
            "date" => Self::Date,
            other => {
                return Err(FdwError::InvalidOption(format!(
                    "partition_keys: type '{other}' not supported in v1 \
                     (supported: int2, int4, int8, text, date)"
                )));
            }
        })
    }
}

/// Allowed `endpoint` suffixes. Restricting to known Azure clouds prevents an attacker who can influence SERVER options from redirecting Managed Identity AAD bearer tokens to an arbitrary host. To extend (e.g. for a private Azurite, Azure Stack, or a future sovereign cloud) add the suffix here.
const ALLOWED_ENDPOINT_SUFFIXES: &[&str] = &[
    // Public cloud
    "blob.core.windows.net",
    "dfs.core.windows.net",
    // Sovereign / government clouds
    "blob.core.chinacloudapi.cn",
    "dfs.core.chinacloudapi.cn",
    "blob.core.usgovcloudapi.net",
    "dfs.core.usgovcloudapi.net",
    "blob.core.cloudapi.de",
    "dfs.core.cloudapi.de",
    // Azurite default (host:port form); validated separately below.
    "127.0.0.1:10000",
    "localhost:10000",
];

/// Storage account names: 3-24 lowercase alphanumeric chars (Azure constraint).
fn validate_account_name(name: &str) -> FdwResult<()> {
    if !(3..=24).contains(&name.len())
        || !name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
    {
        return Err(FdwError::InvalidOption(format!(
            "account_name must be 3-24 lowercase alphanumeric characters (got '{name}')"
        )));
    }
    Ok(())
}

/// Container names: 3-63 chars, lowercase alphanumeric + hyphen, must not
/// start/end with hyphen, no double hyphens (Azure naming rules).
pub fn validate_container_name(name: &str) -> FdwResult<()> {
    let ok = (3..=63).contains(&name.len())
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        && !name.starts_with('-')
        && !name.ends_with('-')
        && !name.contains("--");
    if !ok {
        return Err(FdwError::InvalidOption(format!(
            "container must be 3-63 chars of [a-z0-9-], no leading/trailing or doubled '-' (got '{name}')"
        )));
    }
    Ok(())
}

/// Validate a user-supplied blob name or list-prefix: reject empty, leading
/// `/`, and any `..` path segment. Used by every code path that takes a
/// blob/prefix string from outside the trust boundary (table options,
/// IMPORT FOREIGN SCHEMA's remote_schema, SQL helpers' source argument).
pub fn validate_blob_pattern(pattern: &str, field_name: &str) -> FdwResult<()> {
    if pattern.is_empty() {
        return Err(FdwError::InvalidOption(format!("{field_name} is empty")));
    }
    if pattern.starts_with('/') {
        return Err(FdwError::InvalidOption(format!(
            "{field_name} must be a blob name within the container, not an absolute path"
        )));
    }
    if pattern.split('/').any(|seg| seg == "..") {
        return Err(FdwError::InvalidOption(format!(
            "{field_name} must not contain '..' path segments"
        )));
    }
    Ok(())
}

/// `endpoint` must be one of the cloud suffixes we trust. Rejects e.g.
/// `attacker.com`, `blob.core.windows.net.attacker.com`, or anything with
/// a `/` or `@` (which could subvert URL parsing).
fn validate_endpoint(endpoint: &str) -> FdwResult<()> {
    if endpoint.contains('/') || endpoint.contains('@') || endpoint.contains('?') {
        return Err(FdwError::InvalidOption(format!(
            "endpoint must be a bare host(:port), not a URL (got '{endpoint}')"
        )));
    }
    if !ALLOWED_ENDPOINT_SUFFIXES
        .iter()
        .any(|allowed| endpoint.eq_ignore_ascii_case(allowed))
    {
        return Err(FdwError::InvalidOption(format!(
            "endpoint '{endpoint}' is not in the allowlist of known Azure cloud suffixes"
        )));
    }
    Ok(())
}

/// SAS container URLs: must be `https://<account>.<allowed-endpoint>/<container>...`
/// (Azurite `http://127.0.0.1:10000/...` also accepted). Rejects redirects to
/// link-local / private hosts that would coerce the client into talking to
/// IMDS (`169.254.169.254`) or another internal service.
fn validate_sas_url(sas: &str) -> FdwResult<()> {
    let url = url::Url::parse(sas)
        .map_err(|e| FdwError::InvalidOption(format!("sas_url is not a valid URL: {e}")))?;
    let scheme = url.scheme();
    let host = url
        .host_str()
        .ok_or_else(|| FdwError::InvalidOption("sas_url has no host".into()))?
        .to_ascii_lowercase();

    // Only https in production. http allowed only for Azurite on loopback.
    let is_loopback = matches!(host.as_str(), "127.0.0.1" | "::1" | "localhost");
    if scheme != "https" && !(scheme == "http" && is_loopback) {
        return Err(FdwError::InvalidOption(format!(
            "sas_url must use https (got scheme '{scheme}')"
        )));
    }

    // Block link-local / IMDS / RFC1918 hosts unless it's the Azurite loopback.
    if !is_loopback && is_private_or_link_local(&host) {
        return Err(FdwError::InvalidOption(format!(
            "sas_url host '{host}' is private/link-local — refusing to send credentials there"
        )));
    }

    // Host must end in one of our allowed suffixes (case-insensitive) or be
    // loopback. Suffix check requires a leading dot so `evil.windows.net`
    // doesn't slip through against `blob.core.windows.net`.
    let allowed = is_loopback
        || ALLOWED_ENDPOINT_SUFFIXES.iter().any(|sfx| {
            let s = sfx.to_ascii_lowercase();
            // Strip any `:port` from the allowed suffix for the host compare.
            let bare = s.split(':').next().unwrap_or(&s);
            host == *bare || host.ends_with(&format!(".{bare}"))
        });
    if !allowed {
        return Err(FdwError::InvalidOption(format!(
            "sas_url host '{host}' is not an Azure storage host"
        )));
    }
    Ok(())
}

/// True for IPv4/IPv6 addresses that should never receive Azure credentials:
/// link-local (169.254/16, fe80::/10), loopback (127/8, ::1), and RFC1918
/// private ranges (10/8, 172.16/12, 192.168/16, fc00::/7).
fn is_private_or_link_local(host: &str) -> bool {
    use std::net::IpAddr;
    let Ok(ip) = host.parse::<IpAddr>() else {
        return false;
    };
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            v4.is_loopback()
                || v4.is_link_local()
                || v4.is_private()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_multicast()
                // CGNAT / shared address space — not in is_private().
                || (o[0] == 100 && (64..128).contains(&o[1]))
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                // Link-local fe80::/10
                || (v6.segments()[0] & 0xffc0 == 0xfe80)
                // Unique local fc00::/7
                || (v6.segments()[0] & 0xfe00 == 0xfc00)
        }
    }
}

#[derive(Debug, Clone)]
pub struct ServerOptions {
    pub account_name: String,
    pub endpoint: String,
    pub auth_method: AuthMethod,
    pub enable_pushdown: bool,
}

#[derive(Debug, Default, Clone)]
pub struct UserMappingOptions {
    /// Full container SAS URL when `auth_method='sas_url'`. Not used for
    /// managed identity or AAD service-principal auth.
    pub sas_url: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TableOptions {
    pub container: String,
    pub filename: String,
    /// Codec for newly INSERTed blobs. UPDATE/DELETE rewrites preserve the
    /// source blob's compression codec — the table-level `compression` option
    /// does NOT change a blob's codec on rewrite.
    pub compression: Compression,
    /// Upper bound on parallel workers for SELECT against this table.
    /// `None` → use cluster default (`max_parallel_workers_per_gather`).
    /// `Some(0)` → disable parallel scan for this table entirely.
    /// `Some(n>0)` → cap; PG caps further at the cluster GUC.
    pub parallel_workers: Option<i32>,
    pub partition_columns: Vec<String>,
    pub partition_keys: Vec<(String, PgPartitionType)>,
    pub sorted: Vec<String>,
    pub files_in_order: bool,
}

/// Parse SERVER OPTIONS.
///
/// Required: `account_name`, `auth_method`. Optional: `endpoint`
/// (defaults to `blob.core.windows.net`).
pub fn parse_server_options_from_slice(kv: &[(&str, &str)]) -> FdwResult<ServerOptions> {
    let mut account_name: Option<String> = None;
    let mut endpoint = DEFAULT_ENDPOINT.to_string();
    let mut auth_method: Option<AuthMethod> = None;
    let mut enable_pushdown: bool = true;
    for (k, v) in kv {
        match *k {
            "account_name" => account_name = Some((*v).to_string()),
            "endpoint" => endpoint = (*v).to_string(),
            "auth_method" => auth_method = Some(parse_auth_method(v)?),
            "enable_pushdown" => {
                enable_pushdown = match v.to_ascii_lowercase().as_str() {
                    "true" | "on" | "1" => true,
                    "false" | "off" | "0" => false,
                    _ => {
                        return Err(FdwError::InvalidOption(
                            "enable_pushdown must be true|false".into(),
                        ))
                    }
                };
            }
            other => {
                return Err(FdwError::InvalidOption(format!(
                    "unknown server option '{other}'"
                )))
            }
        }
    }
    let server = ServerOptions {
        account_name: account_name.ok_or(FdwError::MissingOption("account_name"))?,
        endpoint,
        auth_method: auth_method.ok_or(FdwError::MissingOption("auth_method"))?,
        enable_pushdown,
    };
    validate_account_name(&server.account_name)?;
    validate_endpoint(&server.endpoint)?;
    Ok(server)
}

/// Parse USER MAPPING OPTIONS.
///
/// Only `sas_url` is accepted. `account_key` is rejected with remediation
/// pointing the user at `sas_url` (per the azure_storage_blob 1.0 constraint
/// documented on `parse_auth_method`).
pub fn parse_user_mapping_options_from_slice(kv: &[(&str, &str)]) -> FdwResult<UserMappingOptions> {
    let mut out = UserMappingOptions::default();
    for (k, v) in kv {
        match *k {
            "sas_url" => {
                validate_sas_url(v)?;
                out.sas_url = Some((*v).to_string());
            }
            "account_key" => {
                return Err(FdwError::InvalidOption(
                    "user mapping option 'account_key' is not supported; \
                     generate a container SAS client-side and set 'sas_url' instead"
                        .into(),
                ))
            }
            other => {
                return Err(FdwError::InvalidOption(format!(
                    "unknown user mapping option '{other}'"
                )))
            }
        }
    }
    Ok(out)
}

/// Parse FOREIGN TABLE OPTIONS.
///
/// Required: `container`, `filename`. Optional: `compression` (defaults to
/// snappy). `filename` is treated as a blob-name (or glob) within the
/// container and must be relative and free of `..` segments.
pub fn parse_table_options_from_slice(kv: &[(&str, &str)]) -> FdwResult<TableOptions> {
    let mut container: Option<String> = None;
    let mut filename: Option<String> = None;
    let mut compression = Compression::Snappy;
    let mut parallel_workers: Option<i32> = None;
    let mut partition_columns: Option<Vec<String>> = None;
    let mut partition_keys: Option<Vec<(String, PgPartitionType)>> = None;
    let mut sorted: Vec<String> = Vec::new();
    let mut files_in_order: bool = false;
    for (k, v) in kv {
        match *k {
            "container" => container = Some((*v).to_string()),
            "filename" => filename = Some((*v).to_string()),
            "compression" => compression = Compression::parse(v)?,
            "parallel_workers" => {
                let n: i32 = v.parse().map_err(|_| {
                    FdwError::InvalidOption(format!(
                        "parallel_workers must be a non-negative integer; got '{v}'"
                    ))
                })?;
                if n < 0 {
                    return Err(FdwError::InvalidOption(format!(
                        "parallel_workers must be non-negative; got {n}"
                    )));
                }
                parallel_workers = Some(n);
            }
            "partition_columns" => {
                partition_columns = Some(
                    v.split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect(),
                );
            }
            "partition_keys" => {
                let mut parsed: Vec<(String, PgPartitionType)> = Vec::new();
                for entry in v.split(',') {
                    let entry = entry.trim();
                    if entry.is_empty() {
                        continue;
                    }
                    let (name, ty_str) = entry.split_once(':').ok_or_else(|| {
                        FdwError::InvalidOption(format!(
                            "partition_keys entry '{entry}' must be 'name:type'"
                        ))
                    })?;
                    let ty = PgPartitionType::parse(ty_str.trim())?;
                    parsed.push((name.trim().to_string(), ty));
                }
                partition_keys = Some(parsed);
            }
            "sorted" => {
                sorted = v
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            "files_in_order" => {
                files_in_order =
                    matches!(v.to_ascii_lowercase().as_str(), "true" | "yes" | "1" | "on");
            }
            other => {
                return Err(FdwError::InvalidOption(format!(
                    "unknown table option '{other}'"
                )))
            }
        }
    }
    let filename = filename.ok_or(FdwError::MissingOption("filename"))?;
    validate_blob_pattern(&filename, "filename")?;
    let container = container.ok_or(FdwError::MissingOption("container"))?;
    validate_container_name(&container)?;
    let (partition_columns, partition_keys) = match (partition_columns, partition_keys) {
        (None, None) => (Vec::new(), Vec::new()),
        (Some(_), None) => {
            return Err(FdwError::InvalidOption(
                "partition_columns set but partition_keys missing".into(),
            ))
        }
        (None, Some(_)) => {
            return Err(FdwError::InvalidOption(
                "partition_keys set but partition_columns missing".into(),
            ))
        }
        (Some(cols), Some(keys)) => {
            if cols.len() != keys.len() {
                return Err(FdwError::InvalidOption(format!(
                    "partition_columns ({} entries) and partition_keys ({} entries) must have the same length",
                    cols.len(),
                    keys.len()
                )));
            }
            for (i, (col, (kname, _))) in cols.iter().zip(keys.iter()).enumerate() {
                if !col.eq_ignore_ascii_case(kname) {
                    return Err(FdwError::InvalidOption(format!(
                        "partition_columns/partition_keys disagree at index {i}: \
                         '{col}' vs '{kname}' (must be same name in same order)"
                    )));
                }
            }
            (cols, keys)
        }
    };
    match (sorted.is_empty(), files_in_order) {
        (false, false) => {
            return Err(FdwError::InvalidOption(
                "'sorted' is set but 'files_in_order' is missing or false; both must be set together"
                    .into(),
            ));
        }
        (true, true) => {
            return Err(FdwError::InvalidOption(
                "'files_in_order' is true but 'sorted' is empty; both must be set together".into(),
            ));
        }
        _ => {} // both unset OR both set — fine
    }
    Ok(TableOptions {
        container,
        filename,
        compression,
        parallel_workers,
        partition_columns,
        partition_keys,
        sorted,
        files_in_order,
    })
}

/// Cross-check SERVER auth_method against USER MAPPING options.
///
/// - `sas_url` auth requires the `sas_url` user-mapping option.
/// - `aad_sp` auth must NOT carry a `sas_url` (it's a config smell — pick one).
/// - `managed_identity` accepts either presence; we don't error on a stray
///   `sas_url` but `aad_sp` does to keep the SP path unambiguous.
pub fn validate_combo(server: &ServerOptions, um: &UserMappingOptions) -> FdwResult<()> {
    match server.auth_method {
        AuthMethod::SasUrl => {
            if um.sas_url.is_none() {
                return Err(FdwError::MissingOption("sas_url"));
            }
        }
        AuthMethod::AadServicePrincipal => {
            if um.sas_url.is_some() {
                return Err(FdwError::InvalidOption(
                    "auth_method='aad_sp' must not set user mapping option 'sas_url'".into(),
                ));
            }
        }
        AuthMethod::ManagedIdentity => {}
    }
    Ok(())
}

/// SQL-level FDW validator entry point. PostgreSQL hands us a `text[]` of
/// `"key=value"` strings plus the OID of the catalog the options were attached
/// to (`pg_foreign_data_wrapper`, `pg_foreign_server`, `pg_user_mapping`, or
/// `pg_foreign_table`). We dispatch to the matching parser; the parser does
/// the real work and surfaces missing/unknown options.
///
/// `pg_foreign_data_wrapper` carries no options for this FDW — we accept the
/// empty list and reject anything else as "unknown".
pub fn validate(options: Vec<Option<String>>, catalog: pg_sys::Oid) -> FdwResult<()> {
    let kv = parse_kv_list(&options)?;
    let kv_refs: Vec<(&str, &str)> = kv.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();

    if catalog == pg_sys::ForeignDataWrapperRelationId {
        if let Some((k, _)) = kv_refs.first() {
            return Err(FdwError::InvalidOption(format!(
                "unknown foreign-data-wrapper option '{k}'"
            )));
        }
        Ok(())
    } else if catalog == pg_sys::ForeignServerRelationId {
        parse_server_options_from_slice(&kv_refs).map(|_| ())
    } else if catalog == pg_sys::UserMappingRelationId {
        parse_user_mapping_options_from_slice(&kv_refs).map(|_| ())
    } else if catalog == pg_sys::ForeignTableRelationId {
        parse_table_options_from_slice(&kv_refs).map(|_| ())
    } else {
        // Unknown catalog — be permissive; PG shouldn't hand us anything else
        // for an FDW validator, but treating it as a hard error would block
        // future PG versions adding new catalogs.
        Ok(())
    }
}

/// Split each `"key=value"` entry into `(key, value)`. PG joins options with
/// `=`; values themselves may contain `=` so we only split on the first one.
fn parse_kv_list(options: &[Option<String>]) -> FdwResult<Vec<(String, String)>> {
    let mut out = Vec::with_capacity(options.len());
    for entry in options.iter().flatten() {
        let (k, v) = entry.split_once('=').ok_or_else(|| {
            FdwError::InvalidOption(format!(
                "malformed option '{}' (expected key=value)",
                crate::redact::redact(entry)
            ))
        })?;
        out.push((k.to_string(), v.to_string()));
    }
    Ok(out)
}
