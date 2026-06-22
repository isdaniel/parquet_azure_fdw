#![forbid(unsafe_code)]
//! Hand-curated table of PG operator OIDs we know how to push down.
//! OIDs are part of PG's stable ABI (initdb-fixed in `pg_operator`).
//! `tests::oid_table_matches_pg_catalog` (#[pg_test]) asserts each (opno,
//! oprname) pair still holds against the running PG cluster.

use crate::fdw::pushdown::PushedOp;
use arrow::datatypes::DataType;
use pgrx::pg_sys;

/// Postgres operator OID → (op, lhs type, rhs type).
/// Source: `SELECT oid, oprname, oprleft::regtype, oprright::regtype FROM pg_operator`.
pub const PUSHABLE_OPS: &[(u32, PushedOp, DataType, DataType)] = &[
    // int2
    (94, PushedOp::Eq, DataType::Int16, DataType::Int16), // int2 =
    (519, PushedOp::Ne, DataType::Int16, DataType::Int16), // int2 <>
    (95, PushedOp::Lt, DataType::Int16, DataType::Int16), // int2 <
    (522, PushedOp::Le, DataType::Int16, DataType::Int16), // int2 <=
    (520, PushedOp::Gt, DataType::Int16, DataType::Int16), // int2 >
    (524, PushedOp::Ge, DataType::Int16, DataType::Int16), // int2 >=
    // int4
    (96, PushedOp::Eq, DataType::Int32, DataType::Int32),
    (518, PushedOp::Ne, DataType::Int32, DataType::Int32),
    (97, PushedOp::Lt, DataType::Int32, DataType::Int32),
    (523, PushedOp::Le, DataType::Int32, DataType::Int32),
    (521, PushedOp::Gt, DataType::Int32, DataType::Int32),
    (525, PushedOp::Ge, DataType::Int32, DataType::Int32),
    // int8
    (410, PushedOp::Eq, DataType::Int64, DataType::Int64),
    (411, PushedOp::Ne, DataType::Int64, DataType::Int64),
    (412, PushedOp::Lt, DataType::Int64, DataType::Int64),
    (414, PushedOp::Le, DataType::Int64, DataType::Int64),
    (413, PushedOp::Gt, DataType::Int64, DataType::Int64),
    (415, PushedOp::Ge, DataType::Int64, DataType::Int64),
    // float4
    (620, PushedOp::Eq, DataType::Float32, DataType::Float32),
    (621, PushedOp::Ne, DataType::Float32, DataType::Float32),
    (622, PushedOp::Lt, DataType::Float32, DataType::Float32),
    (624, PushedOp::Le, DataType::Float32, DataType::Float32),
    (623, PushedOp::Gt, DataType::Float32, DataType::Float32),
    (625, PushedOp::Ge, DataType::Float32, DataType::Float32),
    // float8
    (670, PushedOp::Eq, DataType::Float64, DataType::Float64),
    (671, PushedOp::Ne, DataType::Float64, DataType::Float64),
    (672, PushedOp::Lt, DataType::Float64, DataType::Float64),
    (673, PushedOp::Le, DataType::Float64, DataType::Float64),
    (674, PushedOp::Gt, DataType::Float64, DataType::Float64),
    (675, PushedOp::Ge, DataType::Float64, DataType::Float64),
    // text
    (98, PushedOp::Eq, DataType::Utf8, DataType::Utf8),
    (531, PushedOp::Ne, DataType::Utf8, DataType::Utf8),
    (664, PushedOp::Lt, DataType::Utf8, DataType::Utf8),
    (665, PushedOp::Le, DataType::Utf8, DataType::Utf8),
    (666, PushedOp::Gt, DataType::Utf8, DataType::Utf8),
    (667, PushedOp::Ge, DataType::Utf8, DataType::Utf8),
    // date
    (1093, PushedOp::Eq, DataType::Date32, DataType::Date32),
    (1094, PushedOp::Ne, DataType::Date32, DataType::Date32),
    (1095, PushedOp::Lt, DataType::Date32, DataType::Date32),
    (1096, PushedOp::Le, DataType::Date32, DataType::Date32),
    (1097, PushedOp::Gt, DataType::Date32, DataType::Date32),
    (1098, PushedOp::Ge, DataType::Date32, DataType::Date32),
];

pub fn lookup_op(opno: pg_sys::Oid) -> Option<(PushedOp, &'static DataType, &'static DataType)> {
    let v = opno.to_u32();
    PUSHABLE_OPS
        .iter()
        .find(|(o, _, _, _)| *o == v)
        .map(|(_, op, l, r)| (*op, l, r))
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use super::*;
    use pgrx::prelude::*;

    /// Each (opno, expected_name) pair must round-trip through pg_operator.
    /// Catches OID drift across PG14–18.
    #[pg_test]
    fn oid_table_matches_pg_catalog() {
        let expected = [
            (94u32, "="),
            (519, "<>"),
            (95, "<"),
            (522, "<="),
            (520, ">"),
            (524, ">="),
            (96, "="),
            (518, "<>"),
            (97, "<"),
            (523, "<="),
            (521, ">"),
            (525, ">="),
            (410, "="),
            (411, "<>"),
            (412, "<"),
            (414, "<="),
            (413, ">"),
            (415, ">="),
            (620, "="),
            (621, "<>"),
            (622, "<"),
            (624, "<="),
            (623, ">"),
            (625, ">="),
            (670, "="),
            (671, "<>"),
            (672, "<"),
            (673, "<="),
            (674, ">"),
            (675, ">="),
            (98, "="),
            (531, "<>"),
            (664, "<"),
            (665, "<="),
            (666, ">"),
            (667, ">="),
            (1093, "="),
            (1094, "<>"),
            (1095, "<"),
            (1096, "<="),
            (1097, ">"),
            (1098, ">="),
            // text LIKE (~~) — not in PUSHABLE_OPS; consumed by
            // pushdown_walk::TEXT_LIKE_OPNO to translate `col LIKE 'pfx%'`
            // to a range qual. Included here to guard against OID drift.
            (1209, "~~"),
        ];
        for (oid, name) in expected {
            let oid_val = pg_sys::Oid::from(oid);
            let got: Option<String> = Spi::get_one_with_args(
                "SELECT oprname::text FROM pg_operator WHERE oid = $1",
                &[oid_val.into()],
            )
            .expect("Spi");
            assert_eq!(
                got.as_deref(),
                Some(name),
                "OID {oid} expected oprname '{name}'"
            );
        }
    }
}
