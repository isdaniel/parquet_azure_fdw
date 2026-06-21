# parquet_azure_fdw

PostgreSQL Foreign Data Wrapper for Parquet blobs in Azure Blob Storage. Written in Rust with [pgrx](https://github.com/pgcentralfoundation/pgrx) 0.18.1.

## Status

Supports `SELECT`, `INSERT`, `UPDATE`, `DELETE`, and `COPY` against Parquet blobs. Built and tested on PostgreSQL 14, 15, 16, 17, 18.

## Quick start

```sql
CREATE EXTENSION parquet_azure_fdw;

CREATE SERVER my_azure
  FOREIGN DATA WRAPPER parquet_azure_fdw
  OPTIONS (
    account_name 'mystorage',
    endpoint     'blob.core.windows.net',
    auth_method  'sas_url'
  );

CREATE USER MAPPING FOR CURRENT_USER SERVER my_azure
  OPTIONS (
    sas_url 'https://mystorage.blob.core.windows.net/data?sv=...&sig=...'
  );

CREATE FOREIGN TABLE events (
  id   bigint,
  name text
) SERVER my_azure OPTIONS (
  container 'data',
  filename  'events/2024/*.parquet'   -- single file or simple glob
);

SELECT count(*) FROM events;
```

## Usage

Define a foreign table; read it with `SELECT`, mutate it with `INSERT` / `COPY` / `UPDATE` / `DELETE`.

```sql
-- Read-target: literal name or single-`*` glob; expanded server-side.
CREATE FOREIGN TABLE events (id bigint, name text)
  SERVER my_azure
  OPTIONS (container 'data', filename 'events/2024/*.parquet');

-- Write-target: trailing `*` or `/` → UUID-stamped blob per statement.
CREATE FOREIGN TABLE events_write (id bigint, name text)
  SERVER my_azure
  OPTIONS (container 'data', filename 'events/2024/*', compression 'zstd');

SELECT id, name FROM events WHERE id > 100 LIMIT 10;

INSERT INTO events_write VALUES (1, 'alice'), (2, 'bob');   -- 1 blob per stmt
COPY   events_write (id, name) FROM STDIN WITH (FORMAT csv); -- 1 blob, bulk-load
UPDATE events SET name = 'ALICE' WHERE id = 1;
DELETE FROM events WHERE id < 0;
```

- **One blob per write statement.** Prefer `COPY` over a loop of `INSERT VALUES`. A literal `filename` overwrites; a trailing `*`/`/` generates a fresh UUID-stamped name.
- **UPDATE / DELETE are copy-on-write** with `If-Match` on the etag captured at SELECT time. A concurrent writer surfaces as `SQLSTATE 40001 serialization_failure` — retry the transaction. When a delete drops a blob to zero rows it is removed from the container.
- **No `key_columns` option.** Row identity is a synthetic ctid (`blob_id` in the high bits, row offset in the low bits).
- **`compression`** (table option) — `none` | `snappy` (default) | `gzip` | `zstd`.

Retry pattern for concurrent-writer races:

```sql
DO $$ BEGIN
  LOOP BEGIN
    UPDATE events SET name = upper(name) WHERE id = 1;
    EXIT;
  EXCEPTION WHEN serialization_failure THEN
    -- back off and retry
  END; END LOOP;
END $$;
```

## Authentication

| `auth_method` | What it does | Required user-mapping options |
|---|---|---|
| `managed_identity` | System-assigned MI via IMDS / App Service. | none |
| `aad_sp` | AAD service principal. Reads `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET` from the postgres process environment. | none |
| `sas_url` | Pre-signed container URL. | `sas_url` (full container SAS URL) |

`account_key` is **not** supported — `azure_storage_blob` 1.0 has no shared-key authorization policy. If you have an account key, generate a container SAS client-side (e.g. `az storage container generate-sas`) and use `auth_method='sas_url'`.

## Option validation & safety

To prevent SSRF and credential exfiltration the FDW validates options at `CREATE SERVER` / `CREATE USER MAPPING` time:

- `account_name` must be 3–24 lowercase alphanumeric characters.
- `container` must follow Azure container naming rules (3–63 chars of `[a-z0-9-]`, no leading/trailing/doubled `-`).
- `endpoint` must be one of the known Azure cloud suffixes: `blob.core.windows.net`, `dfs.core.windows.net`, `*.chinacloudapi.cn`, `*.usgovcloudapi.net`, `*.cloudapi.de`. Azurite loopback (`127.0.0.1:10000`) is also accepted.
- `sas_url` must be `https://` (or `http://127.0.0.1:…` for Azurite) and point at an Azure-suffix host. URLs targeting link-local (`169.254/16`, IMDS), loopback (other than Azurite), or RFC1918 private ranges are rejected — a misconfigured SAS won't redirect a Managed Identity bearer token to an attacker.
- `filename` is rejected if absolute or if it contains `..` path segments.

Error messages bubbled from the Azure SDK are run through a redactor that strips `sig=`, `signature=`, AAD `Bearer` tokens, and user-delegation key identifiers before reaching the Postgres log.

## Type support

| PostgreSQL type | Read | Write |
|---|:-:|:-:|
| `bool`, `int2`, `int4`, `int8` | ✅ | ✅ |
| `float4`, `float8` | ✅ | ✅ |
| `text`, `varchar` | ✅ | ✅ |
| `bytea` | ✅ | ✅ |
| `date` | ✅ | ✅ |
| `timestamp`, `timestamptz` | ✅ | ✅ |
| `numeric(p,s)` (`p ≤ 38` → `Decimal128`) | ✅ | ✅ |
| `jsonb` (parquet `Utf8` ↔ jsonb) | ✅ | ✅ |

Text columns assume the database encoding is UTF-8.

## Glob and write semantics

- `filename` accepts a literal blob name (`events/2024/jan.parquet`) or a single-`*` glob (`events/2024/*.parquet`). `?` and multi-`*` patterns are rejected.
- For **writes**, each `INSERT` / `COPY` statement produces **one new blob**. Literal `filename` → overwrite on every statement; trailing `*` or `/` → UUID-stamped name per statement. Prefer the prefix form for write-target foreign tables.
- One blob per write statement means an interactive `INSERT VALUES` loop creates one blob per statement — fine for batch loads, bad for high-frequency writes. Azure list operations degrade above ~100k blobs per container.

## Concurrency

- The scan side captures each blob's etag at LIST/HEAD time. The modify side stashes that scan-time etag into the per-row identifier and uses it as the `If-Match` precondition on both the GET and the PUT. Any concurrent writer between the SELECT and the UPDATE/DELETE causes the GET to fail with HTTP 412, surfaced as `SQLSTATE 40001 serialization_failure`.
- There is no FDW-internal retry; the application is expected to catch `serialization_failure` and replay the transaction.

### Staged writes and cleanup

`UPDATE` and `DELETE` use a two-phase write:

1. **Stage** — for each affected blob, write the rewritten parquet to `<original>.tmp.<uuid>.parquet` via `If-None-Match: *` (create-only).
2. **Swap** — `If-Match: <scan_etag>` PUTs the new bytes onto the original name, then deletes the staging blob.

The scan list filter hides any blob whose name contains `.tmp.`, so external readers never see staging blobs and **user blob names must not contain the `.tmp.` infix**. If a statement fails or the backend dies mid-rewrite, a per-backend `XactCallback` sweeps the still-registered staging blobs on abort. In the rare case both phase-2 commit AND the abort hook fail (transient network), orphaned `*.tmp.*` blobs remain hidden but consume storage — sweep manually:

```sh
az storage blob delete-batch --account-name <acct> -s <container> --pattern '*.tmp.*'
```

## Resource caps

To bound memory under malicious / accidental large containers:

- `MAX_LIST_RESULTS = 100 000` — `list_blobs` aborts past this; narrow the prefix.
- `MAX_BLOB_BYTES = 512 MiB` — symmetric cap on **read** (`get_with_etag` / `get_body_if_match`) and **write** (single-shot upload + `put_if_match`). The UPDATE/DELETE rewrite kernel materialises the full decoded blob plus an Arrow concat-and-rebuild buffer, so peak RSS per modify statement is roughly **3× the decoded blob size**; the 512 MiB cap keeps the worst case comfortably below the Postgres `MaxAllocSize` (~1 GiB). Users with larger inputs should split them at INSERT time.

## Limitations

- **Per-blob atomic, not statement-atomic.** A statement touching N blobs commits each blob independently. If blob #k of N fails (etag conflict, 5xx, etc.), blobs #1..k-1 are already committed; ROLLBACK does not undo them. Application-level retry on `serialization_failure` is the documented recovery.
- **Glob patterns.** Single trailing `*` only (`dir/prefix-*.parquet`); `?` and multi-`*` patterns are rejected at validation time.
- **Qual pushdown is a fixed whitelist.** See §Qual pushdown — predicates outside the whitelist are evaluated by Postgres (correct, just slower).
- **Reserved name infix `.tmp.`** Blob names containing `.tmp.` are hidden by the scan listing; they're reserved for UPDATE/DELETE staging.
- **Hive-style partition discovery** is not implemented.
- **`MERGE`** is not implemented.
- **Codec is not round-tripped on UPDATE/DELETE.** The rewrite uses the foreign-table `compression` option (default `snappy`). If the source blob was written with a different codec it is silently recompressed.
- **Type matrix:** see §Type support.


