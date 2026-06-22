#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]
//! `ImportForeignSchema_function` — auto-generate CREATE FOREIGN TABLE DDL
//! from a container of Parquet blobs.
//!
//! Grouping strategy: one foreign table per directory prefix. The table
//! name is the LAST directory component (or the container name when the
//! blob lives at the container root). See SP-2 design doc for details.

use std::collections::BTreeMap;

/// Group blobs by their full directory prefix. Returns a map keyed by the
/// **table name** (the LAST directory component of the prefix, or the
/// container name for root-level blobs) → list of full blob paths.
///
/// If two distinct directory prefixes would produce the same table name
/// (e.g. `a/raw/` and `b/raw/` both → table `raw`), the FIRST prefix wins
/// (BTreeMap iteration order = lexicographic by full prefix) and subsequent
/// colliding prefixes are skipped with a NOTICE. To import the skipped one,
/// narrow the `remote_schema` prefix to disambiguate.
pub fn group_blobs_by_directory(
    blobs: &[String],
    container: &str,
) -> BTreeMap<String, Vec<String>> {
    // First pass: group by FULL directory prefix.
    let mut by_dir: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for b in blobs {
        let dir = match b.rfind('/') {
            Some(idx) => b[..idx].to_string(),
            None => String::new(), // root-level
        };
        by_dir.entry(dir).or_default().push(b.clone());
    }

    // Second pass: derive table name from dir, detect collisions.
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut claimed_by: BTreeMap<String, String> = BTreeMap::new(); // table → first_dir
    for (dir, blobs_in_dir) in by_dir {
        let table = if dir.is_empty() {
            container.to_string()
        } else {
            match dir.rfind('/') {
                Some(idx) => dir[idx + 1..].to_string(),
                None => dir.clone(),
            }
        };
        if let Some(prev_dir) = claimed_by.get(&table) {
            emit_notice(&format!(
                "skipped directory '{dir}' — its table name '{table}' is already \
                 claimed by directory '{prev_dir}'; narrow the IMPORT FOREIGN SCHEMA \
                 remote_schema prefix to disambiguate"
            ));
            continue;
        }
        claimed_by.insert(table.clone(), dir);
        out.insert(table, blobs_in_dir);
    }
    out
}

/// PG identifier quoting: wrap in `"..."` and double any internal `"`.
fn quote_ident(s: &str) -> String {
    let mut q = String::with_capacity(s.len() + 2);
    q.push('"');
    for c in s.chars() {
        if c == '"' {
            q.push('"');
            q.push('"');
        } else {
            q.push(c);
        }
    }
    q.push('"');
    q
}

/// PG string-literal quoting: wrap in `'...'` and double any internal `'`.
fn quote_str(s: &str) -> String {
    let mut q = String::with_capacity(s.len() + 2);
    q.push('\'');
    for c in s.chars() {
        if c == '\'' {
            q.push('\'');
            q.push('\'');
        } else {
            q.push(c);
        }
    }
    q.push('\'');
    q
}

pub fn build_create_table_ddl(
    local_schema: &str,
    table_name: &str,
    server_name: &str,
    container: &str,
    filename_glob: &str,
    columns: &[(String, String)],
) -> String {
    let mut s = String::new();
    s.push_str("CREATE FOREIGN TABLE ");
    s.push_str(&quote_ident(local_schema));
    s.push('.');
    s.push_str(&quote_ident(table_name));
    s.push_str(" (\n");
    for (i, (col, ty)) in columns.iter().enumerate() {
        if i > 0 {
            s.push_str(",\n");
        }
        s.push_str("    ");
        s.push_str(&quote_ident(col));
        s.push(' ');
        s.push_str(ty);
    }
    s.push_str("\n) SERVER ");
    s.push_str(&quote_ident(server_name));
    s.push_str(" OPTIONS (");
    s.push_str("container ");
    s.push_str(&quote_str(container));
    s.push_str(", filename ");
    s.push_str(&quote_str(filename_glob));
    s.push_str(");");
    s
}

// ---------------------------------------------------------------------------
// FFI callback: PG-side IMPORT FOREIGN SCHEMA wiring.
// ---------------------------------------------------------------------------

use crate::azure::{build_credential, AzureBlobClient};
use crate::convert::arrow_to_pg::arrow_type_to_pg_typename;
use crate::error::{raise, FdwError, FdwResult};
use crate::fdw::options::{parse_server_options_from_slice, parse_user_mapping_options_from_slice};
use crate::runtime;
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use pgrx::pg_sys;
use std::ffi::{CStr, CString};

/// Build an `AzureBlobClient` from a server OID and container name.
///
/// Reads SERVER + USER MAPPING options, validates the combination, builds
/// credentials, and constructs the client. Used by both the IMPORT FOREIGN
/// SCHEMA callback and SP-2 Task 5's SQL helpers (`import_parquet_azure*`).
pub fn client_for_server(server_oid: pg_sys::Oid, container: &str) -> FdwResult<AzureBlobClient> {
    // Defense in depth: enforce container naming for every caller path.
    crate::fdw::options::validate_container_name(container)?;
    // SAFETY: PG catalog accessors take Oid; `server_oid` comes from the
    // caller's validated path. The actual unsafe is contained in
    // `read_server_and_um`.
    let (server_opts, um_opts) = unsafe { read_server_and_um(server_oid)? };
    crate::fdw::options::validate_combo(&server_opts, &um_opts)?;
    let cred = build_credential(
        &server_opts.auth_method,
        &server_opts.account_name,
        um_opts.sas_url.as_deref(),
    )?;
    AzureBlobClient::new(
        &server_opts.endpoint,
        &server_opts.account_name,
        cred,
        container,
    )
}

/// `ImportForeignSchema_function` — generate CREATE FOREIGN TABLE DDL for
/// every directory-prefix group under `stmt.remote_schema` in the server's
/// configured container.
///
/// Returns a `*mut List` of `char*` SQL strings; PG executes each one as
/// part of the IMPORT FOREIGN SCHEMA statement.
///
/// # Safety
///
/// PG passes a live `ImportForeignSchemaStmt*` and a valid server OID.
pub unsafe extern "C-unwind" fn import_foreign_schema(
    stmt: *mut pg_sys::ImportForeignSchemaStmt,
    server_oid: pg_sys::Oid,
) -> *mut pg_sys::List {
    // SAFETY: PG-supplied pointer is valid for the duration of the callback.
    let result = unsafe { do_import(stmt, server_oid) };
    match result {
        Ok(list) => list,
        Err(e) => raise(e),
    }
}

unsafe fn do_import(
    stmt: *mut pg_sys::ImportForeignSchemaStmt,
    server_oid: pg_sys::Oid,
) -> FdwResult<*mut pg_sys::List> {
    // SAFETY: PG guarantees `stmt` lives for the callback.
    let (remote_schema, local_schema, server_name, container) = unsafe {
        let server = pg_sys::GetForeignServer(server_oid);
        if server.is_null() {
            return Err(FdwError::InvalidOption("foreign server not found".into()));
        }
        let server_name = CStr::from_ptr((*server).servername)
            .to_string_lossy()
            .into_owned();

        let remote_schema = CStr::from_ptr((*stmt).remote_schema)
            .to_string_lossy()
            .into_owned();
        let local_schema = CStr::from_ptr((*stmt).local_schema)
            .to_string_lossy()
            .into_owned();

        // Container is taken from the IMPORT OPTIONS list (mandatory; we
        // require the user to specify it).
        let opts_kv = pg_list_to_kv((*stmt).options);
        let container = opts_kv
            .iter()
            .find(|(k, _)| k == "container")
            .map(|(_, v)| v.clone())
            .ok_or(FdwError::MissingOption(
                "IMPORT FOREIGN SCHEMA requires OPTIONS (container 'xxx')",
            ))?;
        crate::fdw::options::validate_container_name(&container)?;
        (remote_schema, local_schema, server_name, container)
    };

    // Build the Azure client from server/user-mapping options.
    let client = client_for_server(server_oid, &container)?;

    // SSRF guard: validate remote_schema before any LIST/HEAD/GET on Azure.
    crate::fdw::options::validate_blob_pattern(&remote_schema, "remote_schema")?;

    // List under the remote_schema prefix.
    let listed: Vec<(String, String)> =
        runtime::block_on(client.list_with_prefix_etags(&remote_schema))?;
    let blob_names: Vec<String> = listed.into_iter().map(|(n, _)| n).collect();
    let groups = group_blobs_by_directory(&blob_names, &container);

    // Apply LIMIT TO / EXCEPT.
    let groups = unsafe { apply_limit_or_except(groups, stmt) };

    // For each group, infer schema from the first blob, emit DDL.
    let mut ddls: Vec<String> = Vec::new();
    for (table_name, blobs_in_group) in &groups {
        if blobs_in_group.is_empty() {
            continue;
        }
        let sample = &blobs_in_group[0];
        let columns = match infer_columns(&client, sample) {
            Ok(c) => c,
            Err(e) => {
                // Emit a NOTICE and skip this group.
                emit_notice(&format!(
                    "skipped table \"{table_name}\" (schema inference failed: {e})"
                ));
                continue;
            }
        };
        // Filename glob = `<directory_prefix>/*.parquet` if directory exists,
        // else `*.parquet`.
        let filename_glob = build_filename_glob(sample);
        let ddl = build_create_table_ddl(
            &local_schema,
            table_name,
            &server_name,
            &container,
            &filename_glob,
            &columns,
        );
        ddls.push(ddl);
    }

    // Pack into a *List of palloc'd char*.
    let mut list: *mut pg_sys::List = std::ptr::null_mut();
    for d in &ddls {
        let cs = CString::new(d.as_str())
            .map_err(|_| FdwError::SchemaMismatch("generated DDL contains NUL byte".into()))?;
        // SAFETY: palloc-copy the C string into PG's memory context, then
        // append to the list.
        list = unsafe {
            let ptr = pg_sys::pstrdup(cs.as_ptr());
            pg_sys::lappend(list, ptr as *mut std::ffi::c_void)
        };
    }
    Ok(list)
}

fn build_filename_glob(sample: &str) -> String {
    match sample.rfind('/') {
        Some(idx) => format!("{}/*.parquet", &sample[..idx]),
        None => "*.parquet".to_string(),
    }
}

pub fn infer_columns(client: &AzureBlobClient, name: &str) -> FdwResult<Vec<(String, String)>> {
    let reader = client.open_blob(name);
    let builder = runtime::block_on(ParquetRecordBatchStreamBuilder::new(reader))?;
    let arrow_schema = builder.schema();
    let mut cols = Vec::with_capacity(arrow_schema.fields().len());
    for f in arrow_schema.fields() {
        let ty = arrow_type_to_pg_typename(f.data_type())?;
        cols.push((f.name().clone(), ty));
    }
    Ok(cols)
}

unsafe fn read_server_and_um(
    server_oid: pg_sys::Oid,
) -> FdwResult<(
    crate::fdw::options::ServerOptions,
    crate::fdw::options::UserMappingOptions,
)> {
    // SAFETY: PG catalog accessors return palloc'd values valid for the
    // current memory context.
    unsafe {
        let server = pg_sys::GetForeignServer(server_oid);
        if server.is_null() {
            return Err(FdwError::InvalidOption("foreign server not found".into()));
        }
        let um = pg_sys::GetUserMapping(pg_sys::GetUserId(), server_oid);
        let server_kv = pg_list_to_kv((*server).options);
        let um_kv = if um.is_null() {
            Vec::new()
        } else {
            pg_list_to_kv((*um).options)
        };
        let server_opts = parse_server_options_from_slice(
            &server_kv
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect::<Vec<_>>(),
        )?;
        let um_opts = parse_user_mapping_options_from_slice(
            &um_kv
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect::<Vec<_>>(),
        )?;
        Ok((server_opts, um_opts))
    }
}

unsafe fn pg_list_to_kv(list: *mut pg_sys::List) -> Vec<(String, String)> {
    if list.is_null() {
        return Vec::new();
    }
    // SAFETY: PG list iteration via pgrx PgList helper.
    let pg_list: pgrx::PgList<pg_sys::DefElem> = unsafe { pgrx::PgList::from_pg(list) };
    let mut out = Vec::with_capacity(pg_list.len());
    for def in pg_list.iter_ptr() {
        if def.is_null() {
            continue;
        }
        // SAFETY: defname is a palloc'd NUL-terminated string.
        let name = unsafe {
            CStr::from_ptr((*def).defname)
                .to_string_lossy()
                .into_owned()
        };
        // SAFETY: defGetString accepts a live DefElem pointer.
        let value_ptr = unsafe { pg_sys::defGetString(def) };
        let value = if value_ptr.is_null() {
            String::new()
        } else {
            // SAFETY: defGetString returns a palloc'd NUL-terminated string.
            unsafe { CStr::from_ptr(value_ptr).to_string_lossy().into_owned() }
        };
        out.push((name, value));
    }
    out
}

unsafe fn apply_limit_or_except(
    mut groups: BTreeMap<String, Vec<String>>,
    stmt: *mut pg_sys::ImportForeignSchemaStmt,
) -> BTreeMap<String, Vec<String>> {
    // SAFETY: PG guarantees stmt lives for the call.
    let list_type = unsafe { (*stmt).list_type };
    if list_type == pg_sys::ImportForeignSchemaType::FDW_IMPORT_SCHEMA_ALL {
        return groups;
    }
    // SAFETY: PG guarantees stmt lives for the call.
    let table_list = unsafe { (*stmt).table_list };
    let names: Vec<String> = if table_list.is_null() {
        Vec::new()
    } else {
        // SAFETY: table_list is a PG List of RangeVar*.
        let pg_list: pgrx::PgList<pg_sys::RangeVar> = unsafe { pgrx::PgList::from_pg(table_list) };
        pg_list
            .iter_ptr()
            .filter_map(|rv| {
                if rv.is_null() {
                    None
                } else {
                    // SAFETY: rv is non-null and points at a live RangeVar.
                    Some(unsafe { CStr::from_ptr((*rv).relname).to_string_lossy().into_owned() })
                }
            })
            .collect()
    };
    match list_type {
        pg_sys::ImportForeignSchemaType::FDW_IMPORT_SCHEMA_LIMIT_TO => {
            groups.retain(|k, _| names.contains(k));
        }
        pg_sys::ImportForeignSchemaType::FDW_IMPORT_SCHEMA_EXCEPT => {
            groups.retain(|k, _| !names.contains(k));
        }
        _ => {}
    }
    groups
}

#[cfg(not(any(test, feature = "pg_test")))]
fn emit_notice(msg: &str) {
    pgrx::notice!("{}", msg);
}

#[cfg(any(test, feature = "pg_test"))]
fn emit_notice(msg: &str) {
    // When the library is linked into Rust integration tests (tests/*.rs),
    // pgrx::notice! would emit unresolved references to PG runtime symbols
    // (errstart/errfinish/etc.). Fall back to stderr in test builds so the
    // library is link-clean for cargo test. In-crate `#[pg_test]` cases run
    // inside PG and will see this in their stderr capture.
    eprintln!("NOTICE: {msg}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_by_directory_two_tables() {
        let blobs = vec![
            "users/2026.parquet".to_string(),
            "users/2027.parquet".to_string(),
            "orders/2026.parquet".to_string(),
        ];
        let groups = group_blobs_by_directory(&blobs, "data");
        assert_eq!(groups.len(), 2);
        let users = groups.get("users").unwrap();
        assert_eq!(
            users,
            &vec![
                "users/2026.parquet".to_string(),
                "users/2027.parquet".to_string()
            ]
        );
        let orders = groups.get("orders").unwrap();
        assert_eq!(orders, &vec!["orders/2026.parquet".to_string()]);
    }

    #[test]
    fn group_by_directory_root_uses_container_name() {
        let blobs = vec!["root.parquet".to_string(), "other.parquet".to_string()];
        let groups = group_blobs_by_directory(&blobs, "mydata");
        assert_eq!(groups.len(), 1);
        let g = groups.get("mydata").unwrap();
        assert_eq!(g.len(), 2);
    }

    #[test]
    fn group_by_directory_nested_uses_last_component() {
        let blobs = vec![
            "events/raw/a.parquet".to_string(),
            "events/raw/b.parquet".to_string(),
            "events/processed/c.parquet".to_string(),
        ];
        let groups = group_blobs_by_directory(&blobs, "evt");
        assert_eq!(groups.len(), 2);
        assert!(groups.contains_key("raw"));
        assert!(groups.contains_key("processed"));
    }

    #[test]
    fn group_by_directory_basename_collision_skips_later() {
        let blobs = vec!["a/raw/x.parquet".to_string(), "b/raw/y.parquet".to_string()];
        let groups = group_blobs_by_directory(&blobs, "c");
        assert_eq!(groups.len(), 1, "second prefix must be skipped");
        let g = groups.get("raw").unwrap();
        // First prefix wins (alphabetical sort by BTreeMap keys means "a/raw" < "b/raw").
        assert_eq!(g, &vec!["a/raw/x.parquet".to_string()]);
    }

    #[test]
    fn build_create_table_ddl_renders_columns() {
        let cols = vec![
            ("id".to_string(), "INTEGER".to_string()),
            ("name".to_string(), "TEXT".to_string()),
        ];
        let ddl = build_create_table_ddl(
            "public",
            "users",
            "azure_srv",
            "data",
            "users/*.parquet",
            &cols,
        );
        assert!(ddl.contains(r#"CREATE FOREIGN TABLE "public"."users""#));
        assert!(ddl.contains(r#""id" INTEGER"#));
        assert!(ddl.contains(r#""name" TEXT"#));
        assert!(ddl.contains(r#"SERVER "azure_srv""#));
        assert!(ddl.contains(r#"OPTIONS (container 'data', filename 'users/*.parquet')"#));
    }

    #[test]
    fn build_create_table_ddl_quotes_identifiers() {
        // Names containing quotes must be escaped (PG identifier quoting).
        let cols = vec![("co\"l".to_string(), "TEXT".to_string())];
        let ddl = build_create_table_ddl("sch", "tab\"le", "srv", "c", "x", &cols);
        // Double-quote in identifier → escape to "" inside the quoted ident.
        assert!(ddl.contains(r#""tab""le""#));
        assert!(ddl.contains(r#""co""l" TEXT"#));
    }
}
