# parquet_azure_fdw

PostgreSQL Foreign Data Wrapper for Parquet blobs in Azure Blob Storage. Written in Rust with [pgrx](https://github.com/pgcentralfoundation/pgrx).

## Status

Supports `SELECT`, `INSERT`, `UPDATE`, `DELETE`, and `COPY` against Parquet blobs, plus:

- **Qual pushdown** — WHERE clauses prune Parquet row groups (via column-chunk statistics) and filter rows Arrow-side before they reach Postgres.
- **Full glob** — `*` and `?` in any path segment.
- **`IMPORT FOREIGN SCHEMA`** — auto-generate foreign tables from a container of Parquet blobs.
- **Parallel scan** — SELECT spreads across Postgres parallel workers.
- **Hive partitioning** — `key=value/` path components as virtual columns, with partition pruning at list time.
- **Multi-file sorted merge** — a K-way merge over pre-sorted blobs that lets the planner skip its Sort node.

Built and tested on PostgreSQL 14, 15, 16, 17, 18.

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
  filename  'events/2024/*.parquet'
);

SELECT count(*) FROM events;
```

## Usage

Define a foreign table; read it with `SELECT`, mutate it with `INSERT` / `COPY` / `UPDATE` / `DELETE`.

```sql
-- Read-target: literal name or glob; expanded server-side.
CREATE FOREIGN TABLE events (id bigint, name text)
  SERVER my_azure
  OPTIONS (container 'data', filename 'events/2024/*.parquet');

-- Write-target: trailing `*` or `/` → UUID-stamped blob per statement.
CREATE FOREIGN TABLE events_write (id bigint, name text)
  SERVER my_azure
  OPTIONS (container 'data', filename 'events/2024/*', compression 'zstd');

SELECT id, name FROM events WHERE id > 100 LIMIT 10;

INSERT INTO events_write VALUES (1, 'alice'), (2, 'bob');    -- 1 blob per stmt
COPY   events_write (id, name) FROM STDIN WITH (FORMAT csv);  -- 1 blob, bulk-load
UPDATE events SET name = 'ALICE' WHERE id = 1;
DELETE FROM events WHERE id < 0;
```

- **One blob per write statement.** Prefer `COPY` over a loop of `INSERT VALUES`. A literal `filename` overwrites; a trailing `*`/`/` generates a fresh UUID-stamped name.
- **UPDATE / DELETE are copy-on-write** with `If-Match` on the etag captured at SELECT time. A concurrent writer surfaces as `SQLSTATE 40001 serialization_failure` — retry the transaction. When a delete drops a blob to zero rows it is removed from the container. The rewrite **preserves the source blob's compression codec** (see [Compression](#compression)).
- **No `key_columns` option.** Row identity is a synthetic ctid (`blob_id` in the high bits, row offset in the low bits).

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

## Table options

| Option | Applies to | Meaning |
|---|---|---|
| `container` | required | Azure container name. |
| `filename` | required | Blob name or glob (read), or write-target (literal / prefix). |
| `compression` | write | Codec for **new INSERTs**: `none` \| `snappy` (default) \| `gzip` \| `zstd`. |
| `parallel_workers` | read | Cap on parallel workers for SELECT. `0` disables parallel scan for this table. |
| `partition_columns` | read/write | Comma list of Hive partition column names (see [Hive partitioning](#hive-partitioning)). |
| `partition_keys` | read/write | Comma list of `name:type` for each partition column (same names + order). |
| `sorted` | read | Comma list of sort columns for [sorted merge](#multi-file-sorted-merge). |
| `files_in_order` | read | `true` asserts each blob is individually sorted on `sorted` (paired with `sorted`). |

Server option `enable_pushdown` (default `true`) toggles qual pushdown for all tables on the server.

## Globbing

`filename` accepts a literal blob name (`events/2024/jan.parquet`) or a glob:

- `*` matches any run of characters **within one path segment** (not across `/`).
- `?` matches a single character within a segment.
- `events/*/access.log`, `v?/data.parquet`, `data/*.parquet` are all valid.
- Recursive `**`, absolute paths (`/…`), and `..` traversal are rejected at validation time.

For **writes**, each `INSERT` / `COPY` statement produces **one new blob**: a literal `filename` overwrites on every statement; a trailing `*` or `/` produces a UUID-stamped name per statement.

## Qual pushdown

When `enable_pushdown` is on (default), the FDW pushes a fixed whitelist of predicates down into the Parquet scan. Pushdown is **advisory** — Postgres always re-evaluates the original WHERE above the scan, so it never changes results, only speed.

- **Row-group pruning** — for `=`, `<>`, `<`, `<=`, `>`, `>=`, `IS NULL`, `IS NOT NULL`, the scan consults each row group's column-chunk `min`/`max`/`null_count` statistics and skips groups that provably cannot match.
- **Row-level filtering** — surviving row groups are filtered Arrow-side before decode reaches the convert layer.
- **`LIKE 'prefix%'`** is translated to a `col >= 'prefix' AND col < <next>` range; other LIKE shapes fall through to Postgres.

Pushable types: `int2/4/8`, `float4/8`, `text`, `date`, `timestamp` (without time zone), and `numeric` with scale ≤ 18. **Not** pushed (Postgres evaluates them): `timestamptz`, wide-scale decimals, function calls, and predicates on a column with a non-default, non-`C` collation. Missing column-chunk statistics always default to "keep".

## IMPORT FOREIGN SCHEMA

Introspect a container and generate one foreign table per directory prefix:

```sql
IMPORT FOREIGN SCHEMA "sales/2026/"
  FROM SERVER my_azure
  INTO public
  OPTIONS (container 'data');
```

Blobs are grouped by their directory prefix; the table name is the last path component (`data/users/*.parquet` → table `users`). The first blob in each group is sampled to infer the column list, and the emitted DDL preserves the Parquet schema's column order. A group whose sample blob fails schema inference is skipped with a `NOTICE`.

Two SQL helpers wrap the same machinery for callers who can't issue `IMPORT FOREIGN SCHEMA` directly:

```sql
-- Import every directory under a prefix.
SELECT import_parquet_azure(
  server_name   => 'my_azure',
  container     => 'data',
  remote_prefix => 'sales/2026',
  target_schema => 'public');

-- Register one foreign table over an explicit blob ('<container>/<blob>').
SELECT import_parquet_azure_explicit(
  server_name   => 'my_azure',
  target_schema => 'public',
  table_name    => 'snapshot',
  sources       => ARRAY['data/snapshot.parquet']);
```

## Parallel scan

A `SELECT` over a foreign table runs in parallel when Postgres chooses a parallel plan. Workers share a DSM-backed cursor over the blob list and each pulls the next whole blob to scan. Set `parallel_workers '0'` on a table to force the sequential path. `UPDATE` / `DELETE` / `INSERT` always run sequentially (the lost-update guard is leader-only).

## Hive partitioning

Declare `key=value/` path components as virtual columns. Partition columns live in the foreign-table schema but **not** in the Parquet files — they are synthesized per row from the blob path.

```sql
CREATE FOREIGN TABLE events (
  year   int4,    -- partition column
  region text,    -- partition column
  id     int4,    -- storage column (in the parquet file)
  name   text     -- storage column
) SERVER my_azure OPTIONS (
  container         'data',
  filename          'events/*.parquet',
  partition_columns 'year,region',
  partition_keys    'year:int4,region:text'
);
-- reads from data/events/year=2026/region=us/*.parquet, etc.
```

- `partition_columns` and `partition_keys` must name the same columns in the same order. Supported key types: `int2`, `int4`, `int8`, `text`, `date`.
- **Partition pruning** — `WHERE year = 2026` is evaluated against the blob path at *list* time, so non-matching blobs are never opened.
- **INSERT routing** — rows are grouped by their partition tuple and written to `…/year=2026/region=us/<uuid>.parquet`.
- `UPDATE` of a partition column is rejected (relocating a row across partitions = `DELETE` + `INSERT`). A blob whose path is malformed or whose value fails the declared cast is skipped with a `NOTICE`.

## Multi-file sorted merge

When each blob is individually sorted on a key, a K-way merge returns rows in globally-sorted order and lets the planner skip its own Sort node:

```sql
CREATE FOREIGN TABLE logs (ts timestamp, id bigint, msg text)
  SERVER my_azure OPTIONS (
    container      'logs',
    filename       'archive/*.parquet',
    sorted         'ts,id',
    files_in_order 'true'
  );
```

- `sorted` + `files_in_order` must be set together. Ascending, NULLS-LAST ordering only.
- Sort columns must be **storage** columns (not partition columns).
- At most 256 blobs may be merged at once — narrow the glob or a partition filter past that.
- `files_in_order` is an **assertion**: the merge verifies at iteration time that each blob really is sorted and raises a clear error naming the offending blob if not.
- The GUC `parquet_fdw.enable_multifile` (default `on`) toggles the merge per session; with it `off`, a sorted table still returns correct rows via the sequential path.

## Authentication

| `auth_method` | What it does | Required user-mapping options |
|---|---|---|
| `managed_identity` | System-assigned MI via IMDS / App Service. | none |
| `aad_sp` | AAD service principal. Reads `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET` from the postgres process environment. | none |
| `sas_url` | Pre-signed container URL. | `sas_url` (full container SAS URL) |

`account_key` is **not** supported — `azure_storage_blob` 1.0 has no shared-key authorization policy. If you have an account key, generate a container SAS client-side (e.g. `az storage container generate-sas`) and use `auth_method='sas_url'`.

## Option validation & safety

To prevent SSRF and credential exfiltration the FDW validates options at `CREATE SERVER` / `CREATE USER MAPPING` / `CREATE FOREIGN TABLE` time:

- `account_name` must be 3–24 lowercase alphanumeric characters.
- `container` must follow Azure container naming rules (3–63 chars of `[a-z0-9-]`, no leading/trailing/doubled `-`).
- `endpoint` must be one of the known Azure cloud suffixes: `blob.core.windows.net`, `dfs.core.windows.net`, `*.chinacloudapi.cn`, `*.usgovcloudapi.net`, `*.cloudapi.de`. Azurite loopback (`127.0.0.1:10000`) is also accepted.
- `sas_url` must be `https://` (or `http://127.0.0.1:…` for Azurite) and point at an Azure-suffix host. URLs targeting link-local (`169.254/16`, IMDS), loopback (other than Azurite), or RFC1918 private ranges are rejected — a misconfigured SAS won't redirect a Managed Identity bearer token to an attacker.
- `filename`, partition values, and IMPORT prefixes are rejected if absolute or if they contain `..` path segments.

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

Text columns assume the database encoding is UTF-8. (Qual pushdown supports a narrower set than read/write — see [Qual pushdown](#qual-pushdown).)

## Compression

The `compression` table option (`none` | `snappy` (default) | `gzip` | `zstd`) sets the codec for **new INSERTed blobs**. `UPDATE` / `DELETE` rewrites **preserve the source blob's codec** — the table option does not recompress an existing blob to a different codec.

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
- `MAX_BLOB_BYTES = 512 MiB` — symmetric cap on **read** (`get_with_etag` / `get_body_if_match`) and **write** (single-shot upload + `put_if_match`). The `UPDATE`/`DELETE` rewrite kernel **streams** the source blob batch-by-batch, so peak RSS is roughly one decoded batch (not the whole blob); the 512 MiB cap is still enforced cumulatively on the decoded size. Users with larger inputs should split them at INSERT time.

## Limitations

- **Per-blob atomic, not statement-atomic.** A statement touching N blobs commits each blob independently. If blob #k of N fails (etag conflict, 5xx, etc.), blobs #1..k-1 are already committed; ROLLBACK does not undo them. Application-level retry on `serialization_failure` is the documented recovery.
- **Sorted merge:** ascending + NULLS-LAST only; storage (non-partition) sort columns only; ≤ 256 blobs merged at once. With the merge active, projection pushdown and row-group pruning are skipped for that scan (rows are still correct).
- **`MERGE`** is not implemented.
- **Reserved name infix `.tmp.`** — blob names containing `.tmp.` are hidden by the scan listing; they're reserved for UPDATE/DELETE staging.
- **Type matrix:** see [Type support](#type-support); qual pushdown covers a narrower set.

## Building & testing

```sh
make build                 # debug build for the default PG (pg14)
make test                  # cargo pgrx test on the default PG
make test-all              # cargo pgrx test on every supported PG (14–18)
make test-unit             # cargo check + clippy, no Postgres / no Docker
make before-git-push       # fmt --check + clippy -D warnings + pgrx test (PG14)
make before-git-push-all   # the above on every PG version
```

The test harness uses an in-process [wiremock](https://github.com/LukeMathWalker/wiremock-rs) fake of the Azure Blob REST API, so `cargo test` and `cargo pgrx test` need no Docker, no Azurite, and no Azure credentials. The opt-in `make test-live` smoke test runs against real Azurite / Azure when `AZURE_TEST_SAS_URL` is set.
