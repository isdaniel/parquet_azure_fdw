// Inner modules opt into `forbid(unsafe_code)` individually. The
// `partition_datum` sibling carries an unsafe carve-out for the varlena
// detoast path (mirroring the INSERT slot decoder), so we do NOT set
// `#![forbid(unsafe_code)]` at this module level.

pub mod arrow_to_pg;
pub mod partition_datum;
pub mod pg_to_arrow;

pub use partition_datum::datum_to_partition_string;
pub use pg_to_arrow::{pg_attrs_to_arrow_schema, pg_oid_to_arrow_type, RecordBatchBuilders};
