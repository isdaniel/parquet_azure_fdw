#![forbid(unsafe_code)]
//! Hive partition support: path parsing + tuple keying for INSERT routing.

use crate::error::{FdwError, FdwResult};
use crate::fdw::options::PgPartitionType;
use std::collections::BTreeMap;

pub fn partition_values_from_path(
    blob_name: &str,
    declared: &[(String, PgPartitionType)],
) -> FdwResult<BTreeMap<String, String>> {
    let mut found: BTreeMap<String, String> = BTreeMap::new();
    let declared_names: std::collections::HashSet<&str> =
        declared.iter().map(|(n, _)| n.as_str()).collect();

    for segment in blob_name.split('/') {
        let Some((key, value)) = segment.split_once('=') else {
            continue;
        };
        // Only consider segments whose key matches a declared partition.
        if !declared_names.contains(key) {
            continue;
        }
        if value.is_empty() {
            return Err(FdwError::SchemaMismatch(format!(
                "blob '{blob_name}' has empty value for partition key '{key}'"
            )));
        }
        if value.contains('%') {
            return Err(FdwError::SchemaMismatch(format!(
                "blob '{blob_name}' partition '{key}' value contains percent — \
                 URL-encoded partition values are not supported in v1"
            )));
        }
        if found.insert(key.to_string(), value.to_string()).is_some() {
            return Err(FdwError::SchemaMismatch(format!(
                "blob '{blob_name}' has duplicate partition segment '{key}='"
            )));
        }
    }

    // All declared keys must be present.
    for (name, _) in declared {
        if !found.contains_key(name) {
            return Err(FdwError::SchemaMismatch(format!(
                "blob '{blob_name}' missing partition key '{name}'"
            )));
        }
    }

    Ok(found)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PartitionTupleKey {
    pub values: Vec<String>,
}

use crate::fdw::pushdown::{PushedExpr, PushedOp, PushedQual, ScalarValueRepr};

/// Split top-level pushed expressions into partition-only and storage-only.
/// Expressions whose leaves reference BOTH partition and storage cols are
/// dropped from both — PG re-evaluates above the scan.
pub fn split_quals_by_target(
    exprs: Vec<PushedExpr>,
    partition_attnums: &[usize],
) -> (Vec<PushedExpr>, Vec<PushedExpr>) {
    let mut partition = Vec::new();
    let mut storage = Vec::new();
    for e in exprs {
        match classify_target(&e, partition_attnums) {
            Target::Partition => partition.push(e),
            Target::Storage => storage.push(e),
            Target::Mixed => { /* drop */ }
        }
    }
    (partition, storage)
}

enum Target {
    Partition,
    Storage,
    Mixed,
}

fn classify_target(e: &PushedExpr, partition_attnums: &[usize]) -> Target {
    match e {
        PushedExpr::Leaf(q) => {
            if partition_attnums.contains(&q.col) {
                Target::Partition
            } else {
                Target::Storage
            }
        }
        PushedExpr::And(xs) | PushedExpr::Or(xs) => {
            let mut seen_p = false;
            let mut seen_s = false;
            for x in xs {
                match classify_target(x, partition_attnums) {
                    Target::Partition => seen_p = true,
                    Target::Storage => seen_s = true,
                    Target::Mixed => return Target::Mixed,
                }
            }
            match (seen_p, seen_s) {
                (true, true) => Target::Mixed,
                (true, false) => Target::Partition,
                (false, true) => Target::Storage,
                (false, false) => Target::Storage, // empty BoolExpr — irrelevant
            }
        }
        PushedExpr::Not(x) => classify_target(x, partition_attnums),
    }
}

/// Evaluate partition-only quals against a blob's parsed partition values.
/// Returns true if the blob should be kept (all top-level expressions
/// evaluate to TRUE under the parsed values).
pub fn evaluate_partition_quals_against_blob(
    exprs: &[PushedExpr],
    partition_attnums: &[usize],
    parsed: &BTreeMap<String, String>,
    declared: &[(String, PgPartitionType)],
) -> bool {
    if exprs.is_empty() {
        return true;
    }
    for e in exprs {
        if !eval_one_partition_expr(e, partition_attnums, parsed, declared) {
            return false;
        }
    }
    true
}

fn eval_one_partition_expr(
    e: &PushedExpr,
    partition_attnums: &[usize],
    parsed: &BTreeMap<String, String>,
    declared: &[(String, PgPartitionType)],
) -> bool {
    match e {
        PushedExpr::Leaf(q) => eval_partition_leaf(q, partition_attnums, parsed, declared),
        PushedExpr::And(xs) => xs
            .iter()
            .all(|x| eval_one_partition_expr(x, partition_attnums, parsed, declared)),
        PushedExpr::Or(xs) => xs
            .iter()
            .any(|x| eval_one_partition_expr(x, partition_attnums, parsed, declared)),
        PushedExpr::Not(x) => !eval_one_partition_expr(x, partition_attnums, parsed, declared),
    }
}

fn eval_partition_leaf(
    q: &PushedQual,
    partition_attnums: &[usize],
    parsed: &BTreeMap<String, String>,
    declared: &[(String, PgPartitionType)],
) -> bool {
    // Find which declared partition this attno is.
    let pos = match partition_attnums.iter().position(|&a| a == q.col) {
        Some(p) => p,
        None => return true, // not a partition qual — caller bug; keep blob
    };
    let (name, _ty) = &declared[pos];
    let raw = match parsed.get(name) {
        Some(v) => v,
        None => return true, // missing — caller already skipped via NOTICE upstream
    };
    // String-level comparison for everything (sound for Text + Int by lex
    // when values are normalized; for v1 we cast on both sides).
    let lit = match &q.value {
        ScalarValueRepr::I32(v) => v.to_string(),
        ScalarValueRepr::I16(v) => v.to_string(),
        ScalarValueRepr::I64(v) => v.to_string(),
        ScalarValueRepr::Utf8(s) => s.clone(),
        ScalarValueRepr::Date32(d) => {
            // d is days since UNIX epoch; format as YYYY-MM-DD for string compare.
            let unix = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
            let date = unix + chrono::Duration::days(*d as i64);
            date.format("%Y-%m-%d").to_string()
        }
        _ => return true, // unsupported literal type — keep (sound default)
    };
    match q.op {
        PushedOp::Eq => raw == &lit,
        PushedOp::Ne => raw != &lit,
        PushedOp::Lt | PushedOp::Le | PushedOp::Gt | PushedOp::Ge => {
            // For correctness across types, parse both sides as the same kind
            // and compare numerically when possible; otherwise lex compare.
            let raw_n: Result<i64, _> = raw.parse();
            let lit_n: Result<i64, _> = lit.parse();
            match (raw_n, lit_n) {
                (Ok(r), Ok(l)) => match q.op {
                    PushedOp::Lt => r < l,
                    PushedOp::Le => r <= l,
                    PushedOp::Gt => r > l,
                    PushedOp::Ge => r >= l,
                    _ => unreachable!(),
                },
                _ => match q.op {
                    PushedOp::Lt => raw.as_str() < lit.as_str(),
                    PushedOp::Le => raw.as_str() <= lit.as_str(),
                    PushedOp::Gt => raw.as_str() > lit.as_str(),
                    PushedOp::Ge => raw.as_str() >= lit.as_str(),
                    _ => unreachable!(),
                },
            }
        }
        PushedOp::IsNull | PushedOp::IsNotNull => {
            // Partition values are never NULL (path segment is present or
            // blob is skipped).
            matches!(q.op, PushedOp::IsNotNull)
        }
    }
}

#[cfg(test)]
mod split_tests {
    use super::*;
    use crate::fdw::pushdown::{PushedExpr, PushedOp, PushedQual, ScalarValueRepr};

    fn leaf(col: usize, op: PushedOp, value: ScalarValueRepr) -> PushedExpr {
        PushedExpr::Leaf(PushedQual { col, op, value })
    }

    #[test]
    fn split_pure_partition() {
        let exprs = vec![leaf(0, PushedOp::Eq, ScalarValueRepr::I32(2026))];
        let (p, s) = split_quals_by_target(exprs, &[0]);
        assert_eq!(p.len(), 1);
        assert!(s.is_empty());
    }

    #[test]
    fn split_pure_storage() {
        let exprs = vec![leaf(3, PushedOp::Gt, ScalarValueRepr::I32(100))];
        let (p, s) = split_quals_by_target(exprs, &[0, 1]);
        assert!(p.is_empty());
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn split_mixed_within_one_expr_drops() {
        let mixed = PushedExpr::Or(vec![
            leaf(0, PushedOp::Eq, ScalarValueRepr::I32(2026)),
            leaf(3, PushedOp::Gt, ScalarValueRepr::I32(100)),
        ]);
        let (p, s) = split_quals_by_target(vec![mixed], &[0]);
        assert!(p.is_empty());
        assert!(s.is_empty());
    }

    #[test]
    fn evaluate_partition_keeps_matching_blob() {
        let exprs = vec![leaf(0, PushedOp::Eq, ScalarValueRepr::I32(2026))];
        let declared = vec![
            ("year".to_string(), PgPartitionType::Int4),
            ("region".to_string(), PgPartitionType::Text),
        ];
        let mut parsed = BTreeMap::new();
        parsed.insert("year".to_string(), "2026".to_string());
        parsed.insert("region".to_string(), "us".to_string());
        assert!(evaluate_partition_quals_against_blob(
            &exprs,
            &[0],
            &parsed,
            &declared
        ));
    }

    #[test]
    fn evaluate_partition_drops_non_matching_blob() {
        let exprs = vec![leaf(0, PushedOp::Eq, ScalarValueRepr::I32(2026))];
        let declared = vec![("year".to_string(), PgPartitionType::Int4)];
        let mut parsed = BTreeMap::new();
        parsed.insert("year".to_string(), "2027".to_string());
        assert!(!evaluate_partition_quals_against_blob(
            &exprs,
            &[0],
            &parsed,
            &declared
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn declared_year_region() -> Vec<(String, PgPartitionType)> {
        vec![
            ("year".to_string(), PgPartitionType::Int4),
            ("region".to_string(), PgPartitionType::Text),
        ]
    }

    #[test]
    fn parses_both_keys_in_order() {
        let m = partition_values_from_path(
            "events/year=2026/region=us/a.parquet",
            &declared_year_region(),
        )
        .unwrap();
        assert_eq!(m.get("year"), Some(&"2026".to_string()));
        assert_eq!(m.get("region"), Some(&"us".to_string()));
    }

    #[test]
    fn parses_keys_in_any_path_order() {
        // Declared order says year then region, but the path has region first.
        // We look up by name, so this should still parse correctly.
        let m = partition_values_from_path(
            "events/region=us/year=2026/a.parquet",
            &declared_year_region(),
        )
        .unwrap();
        assert_eq!(m.get("year"), Some(&"2026".to_string()));
        assert_eq!(m.get("region"), Some(&"us".to_string()));
    }

    #[test]
    fn missing_key_errors() {
        let err = partition_values_from_path("events/year=2026/a.parquet", &declared_year_region())
            .expect_err("missing region must error");
        assert!(format!("{err}").to_lowercase().contains("region"));
    }

    #[test]
    fn duplicate_key_errors() {
        let err = partition_values_from_path(
            "events/year=2026/year=2027/region=us/a.parquet",
            &declared_year_region(),
        )
        .expect_err("duplicate year must error");
        assert!(format!("{err}").to_lowercase().contains("year"));
    }

    #[test]
    fn extra_segments_ignored() {
        // Path may contain extra segments (e.g. a date directory) — fine, as
        // long as all declared keys are present.
        let m = partition_values_from_path(
            "extras/yr/year=2026/region=us/a.parquet",
            &declared_year_region(),
        )
        .unwrap();
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn empty_value_errors() {
        let err =
            partition_values_from_path("events/year=/region=us/a.parquet", &declared_year_region())
                .expect_err("empty value must error");
        assert!(format!("{err}").to_lowercase().contains("empty"));
    }

    #[test]
    fn percent_encoded_value_errors() {
        // SP-3b v1 rejects URL-encoded values explicitly — keep parsing simple.
        let err = partition_values_from_path(
            "events/year=2026/region=us%2Dwest/a.parquet",
            &declared_year_region(),
        )
        .expect_err("percent-encoded must reject");
        assert!(format!("{err}").to_lowercase().contains("percent"));
    }
}
