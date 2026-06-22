#![forbid(unsafe_code)]
//! Qual pushdown translation.
//!
//! This module owns the (pure-Rust) policy that decides which Postgres quals
//! are eligible to be pushed down into the parquet scan, plus the types Task 12
//! (scan callbacks) consumes when wiring up `BeginForeignScan` /
//! `IterateForeignScan`.
//!
//! # v1 limitation
//!
//! The `is_pushable` whitelist and the `PushedOp` / `PushedQual` /
//! `ScalarValueRepr` / `PushedExpr` types are in place, and
//! [`build_row_filter`] now translates a `&[PushedExpr]` into a real parquet
//! [`RowFilter`] using arrow compute kernels. The remaining stub is the
//! `RestrictInfo` walker that extracts `PushedExpr`s from PG expression trees —
//! that work needs `unsafe` pg_sys catalog access and will live behind the
//! `fdw/` unsafe carve-out, not here — this file stays
//! `#![forbid(unsafe_code)]`.

use arrow::array::{ArrayRef, BooleanArray, RecordBatch};
use arrow::compute::kernels::{boolean, cmp};
use arrow::datatypes::{DataType, Schema};
use arrow::error::ArrowError;
use parquet::arrow::arrow_reader::{ArrowPredicate, ArrowPredicateFn, RowFilter};
use parquet::arrow::ProjectionMask;
use parquet::schema::types::SchemaDescriptor;
use std::sync::Arc;

/// Comparison / null-test operators eligible for pushdown.
///
/// See [`is_pushable`] for the operator-name → variant mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PushedOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    IsNull,
    IsNotNull,
}

impl PushedOp {
    /// Logical negation: `Eq ↔ Ne`, `Lt ↔ Ge`, `Le ↔ Gt`, `IsNull ↔ IsNotNull`.
    pub fn inverse(self) -> Option<Self> {
        Some(match self {
            Self::Eq => Self::Ne,
            Self::Ne => Self::Eq,
            Self::Lt => Self::Ge,
            Self::Ge => Self::Lt,
            Self::Le => Self::Gt,
            Self::Gt => Self::Le,
            Self::IsNull => Self::IsNotNull,
            Self::IsNotNull => Self::IsNull,
        })
    }
}

/// A scalar literal extracted from a PG qual, lowered to a representation that
/// does not borrow from PG memory contexts.
///
/// We deliberately keep this enum small and self-contained rather than reusing
/// `arrow::array::Scalar` (which is generic over arrays) so that pushed quals
/// can be cheaply moved between threads / cloned into the scan state.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarValueRepr {
    /// Used for `IS NULL` / `IS NOT NULL` operators (no payload).
    Null,
    Bool(bool),
    I16(i16),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Utf8(String),
    /// Days since UNIX epoch, matching `DataType::Date32`.
    Date32(i32),
    /// Microseconds since UNIX epoch (timezone resolution is carried by the
    /// arrow schema, not here).
    TimestampMicros(i64),
    /// Decimal128 as `(unscaled, precision, scale)`.
    Decimal128(i128, u8, i8),
}

/// A single pushed-down predicate, addressed by column index into the
/// foreign table's arrow schema.
#[derive(Debug, Clone, PartialEq)]
pub struct PushedQual {
    pub col: usize,
    pub op: PushedOp,
    pub value: ScalarValueRepr,
}

/// Boolean composition of leaf `PushedQual`s. Mirrors PG `BoolExpr` shape.
#[derive(Debug, Clone, PartialEq)]
pub enum PushedExpr {
    Leaf(PushedQual),
    And(Vec<PushedExpr>),
    Or(Vec<PushedExpr>),
    Not(Box<PushedExpr>),
}

/// Return `true` iff `(op, ty)` is on the pushdown whitelist.
///
/// # Operator × DataType matrix
///
/// ```text
///                       Eq Ne Lt Le Gt Ge NULL
/// Boolean               Y  Y  Y  Y  Y  Y  Y
/// Int16/32/64           Y  Y  Y  Y  Y  Y  Y
/// Float32/64            Y  Y  Y  Y  Y  Y  Y
/// Utf8                  Y  Y  Y  Y  Y  Y  Y
/// Date32                Y  Y  Y  Y  Y  Y  Y
/// Timestamp(_, None)    Y  Y  Y  Y  Y  Y  Y    (TIMESTAMPTZ rejected)
/// Decimal128(p, s≤18)   Y  Y  Y  Y  Y  Y  Y    (scale > 18 rejected)
/// ```
///
/// Operator names are the textual form PG hands us (`"="`, `"<="`, …) plus
/// the synthetic `"IS NULL"` / `"IS NOT NULL"` tokens for null tests.
///
/// Soundness notes:
/// - `TIMESTAMPTZ` (`Timestamp(_, Some(_))`) is rejected because the timezone
///   resolution belongs to the session, not the stored value, and our
///   `ScalarValueRepr::TimestampMicros` does not carry a timezone.
/// - Decimal128 with `scale > 18` is rejected because our stats-comparison
///   path does not yet handle wide-scale arithmetic; the comparison may
///   produce wrong results at the boundary.
/// - Collation-bearing `Utf8` quals are handled by the WALKER (which can see
///   `Var.varcollid`); this function only knows the `DataType` and must
///   conservatively accept Utf8.
pub fn is_pushable(op: &str, ty: &DataType) -> bool {
    let op_ok = matches!(
        op,
        "=" | "<>" | "<" | "<=" | ">" | ">=" | "IS NULL" | "IS NOT NULL"
    );
    let ty_ok = match ty {
        DataType::Boolean
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::Float32
        | DataType::Float64
        | DataType::Utf8
        | DataType::Date32 => true,
        DataType::Timestamp(_, tz) => tz.is_none(),
        DataType::Decimal128(_, scale) => *scale <= 18 && *scale >= 0,
        _ => false,
    };
    op_ok && ty_ok
}

/// Build a parquet [`RowFilter`] from a slice of [`PushedExpr`]s.
///
/// Each top-level expression becomes one [`ArrowPredicate`]; the row filter
/// applies them in order and short-circuits via parquet's own row-filter
/// pipeline (a row is kept iff every predicate returns `true`). Returns `None`
/// when the input is empty so callers can avoid attaching a no-op filter.
pub fn build_row_filter(
    exprs: &[PushedExpr],
    arrow_schema: &Schema,
    parquet_schema: &SchemaDescriptor,
) -> Option<RowFilter> {
    if exprs.is_empty() {
        return None;
    }
    let mut predicates: Vec<Box<dyn ArrowPredicate>> = Vec::with_capacity(exprs.len());
    for e in exprs {
        let cols = collect_cols(e);
        let mask = ProjectionMask::roots(parquet_schema, cols.iter().copied());
        let cols_owned = cols.clone();
        let arrow_schema = arrow_schema.clone();
        let e_owned = e.clone();
        let pred = ArrowPredicateFn::new(mask, move |batch: RecordBatch| {
            let local_idx =
                |global: usize| -> Option<usize> { cols_owned.iter().position(|c| *c == global) };
            eval(&e_owned, &batch, &arrow_schema, &local_idx)
        });
        predicates.push(Box::new(pred));
    }
    Some(RowFilter::new(predicates))
}

fn collect_cols(e: &PushedExpr) -> Vec<usize> {
    fn walk(e: &PushedExpr, out: &mut Vec<usize>) {
        match e {
            PushedExpr::Leaf(q) => {
                if !out.contains(&q.col) {
                    out.push(q.col);
                }
            }
            PushedExpr::And(xs) | PushedExpr::Or(xs) => xs.iter().for_each(|c| walk(c, out)),
            PushedExpr::Not(x) => walk(x, out),
        }
    }
    let mut out = Vec::new();
    walk(e, &mut out);
    out
}

fn eval(
    e: &PushedExpr,
    batch: &RecordBatch,
    schema: &Schema,
    local_idx: &dyn Fn(usize) -> Option<usize>,
) -> Result<BooleanArray, ArrowError> {
    match e {
        PushedExpr::Leaf(q) => eval_leaf(q, batch, schema, local_idx),
        PushedExpr::And(xs) => {
            let mut acc: Option<BooleanArray> = None;
            for x in xs {
                let b = eval(x, batch, schema, local_idx)?;
                acc = Some(match acc {
                    None => b,
                    Some(a) => boolean::and(&a, &b)?,
                });
            }
            Ok(acc.unwrap_or_else(|| BooleanArray::from(vec![true; batch.num_rows()])))
        }
        PushedExpr::Or(xs) => {
            let mut acc: Option<BooleanArray> = None;
            for x in xs {
                let b = eval(x, batch, schema, local_idx)?;
                acc = Some(match acc {
                    None => b,
                    Some(a) => boolean::or(&a, &b)?,
                });
            }
            Ok(acc.unwrap_or_else(|| BooleanArray::from(vec![false; batch.num_rows()])))
        }
        PushedExpr::Not(x) => {
            let b = eval(x, batch, schema, local_idx)?;
            boolean::not(&b)
        }
    }
}

fn eval_leaf(
    q: &PushedQual,
    batch: &RecordBatch,
    _schema: &Schema,
    local_idx: &dyn Fn(usize) -> Option<usize>,
) -> Result<BooleanArray, ArrowError> {
    let idx = local_idx(q.col).ok_or_else(|| {
        ArrowError::ComputeError(format!("column {} missing in predicate batch", q.col))
    })?;
    let col: &ArrayRef = batch.column(idx);
    match q.op {
        PushedOp::IsNull => arrow::compute::is_null(col.as_ref()),
        PushedOp::IsNotNull => arrow::compute::is_not_null(col.as_ref()),
        op => {
            // For comparison ops, wrap the scalar in a single-element typed
            // array and use `arrow::array::Scalar` to broadcast it against
            // the column. `&dyn Datum` is implemented for both `ArrayRef` and
            // `Scalar<T>` in arrow 59.
            let scalar_arr = scalar_to_array(&q.value, col.data_type())?;
            let scalar = arrow::array::Scalar::new(scalar_arr);
            let lhs: &dyn arrow::array::Datum = &col.as_ref();
            let rhs: &dyn arrow::array::Datum = &scalar;
            match op {
                PushedOp::Eq => cmp::eq(lhs, rhs),
                PushedOp::Ne => cmp::neq(lhs, rhs),
                PushedOp::Lt => cmp::lt(lhs, rhs),
                PushedOp::Le => cmp::lt_eq(lhs, rhs),
                PushedOp::Gt => cmp::gt(lhs, rhs),
                PushedOp::Ge => cmp::gt_eq(lhs, rhs),
                PushedOp::IsNull | PushedOp::IsNotNull => unreachable!(),
            }
        }
    }
}

fn scalar_to_array(v: &ScalarValueRepr, _ty: &DataType) -> Result<ArrayRef, ArrowError> {
    use arrow::array::*;
    Ok(match v {
        ScalarValueRepr::Bool(b) => Arc::new(BooleanArray::from(vec![*b])) as ArrayRef,
        ScalarValueRepr::I16(x) => Arc::new(Int16Array::from(vec![*x])),
        ScalarValueRepr::I32(x) => Arc::new(Int32Array::from(vec![*x])),
        ScalarValueRepr::I64(x) => Arc::new(Int64Array::from(vec![*x])),
        ScalarValueRepr::F32(x) => Arc::new(Float32Array::from(vec![*x])),
        ScalarValueRepr::F64(x) => Arc::new(Float64Array::from(vec![*x])),
        ScalarValueRepr::Utf8(s) => Arc::new(StringArray::from(vec![s.as_str()])),
        ScalarValueRepr::Date32(d) => Arc::new(Date32Array::from(vec![*d])),
        ScalarValueRepr::TimestampMicros(t) => Arc::new(TimestampMicrosecondArray::from(vec![*t])),
        ScalarValueRepr::Decimal128(val, p, s) => {
            Arc::new(Decimal128Array::from(vec![*val]).with_precision_and_scale(*p, *s)?)
        }
        ScalarValueRepr::Null => {
            return Err(ArrowError::ComputeError("Null scalar in cmp".into()));
        }
    })
}

// ---------- row-group pruning ---------------------------------------------

/// Verdict for a single PushedExpr applied against a single row group's
/// column-chunk statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RgVerdict {
    /// Predicate cannot match any row in this group.
    CannotMatch,
    /// Predicate could match at least one row (or is indeterminate).
    CanMatch,
    /// Stats absent / unsupported — caller treats as `CanMatch` (sound).
    Indeterminate,
}

/// Prune row groups by column-chunk statistics.
///
/// `None` → no pruning was determinable; the caller should scan every group
///         (the `Some(qf)` arm in `scan.rs` skips `with_row_groups` entirely).
/// `Some(v)` → pass `v` to `ParquetRecordBatchStreamBuilder::with_row_groups`.
///             `v` may be empty (the whole blob is pruned).
pub fn prune_row_groups(
    meta: &parquet::file::metadata::ParquetMetaData,
    exprs: &[PushedExpr],
    arrow_schema: &Schema,
) -> Option<Vec<usize>> {
    if exprs.is_empty() {
        return None;
    }
    let n = meta.num_row_groups();
    let mut any_determinable = false;
    let mut keep = Vec::with_capacity(n);
    for rg_idx in 0..n {
        let rg = meta.row_group(rg_idx);
        let mut survives = true;
        for e in exprs {
            match evaluate_against_rg(e, rg, arrow_schema) {
                RgVerdict::CannotMatch => {
                    survives = false;
                    any_determinable = true;
                    break;
                }
                RgVerdict::CanMatch => any_determinable = true,
                RgVerdict::Indeterminate => {}
            }
        }
        if survives {
            keep.push(rg_idx);
        }
    }
    if any_determinable {
        Some(keep)
    } else {
        None
    }
}

fn evaluate_against_rg(
    e: &PushedExpr,
    rg: &parquet::file::metadata::RowGroupMetaData,
    schema: &Schema,
) -> RgVerdict {
    match e {
        PushedExpr::Leaf(q) => evaluate_leaf_against_rg(q, rg, schema),
        PushedExpr::And(xs) => {
            // AND: any child CannotMatch → CannotMatch; else CanMatch
            // (Indeterminate is non-strengthening — defaults to keep).
            let mut all_can = true;
            for x in xs {
                match evaluate_against_rg(x, rg, schema) {
                    RgVerdict::CannotMatch => return RgVerdict::CannotMatch,
                    RgVerdict::Indeterminate => all_can = false,
                    RgVerdict::CanMatch => {}
                }
            }
            if all_can {
                RgVerdict::CanMatch
            } else {
                RgVerdict::Indeterminate
            }
        }
        PushedExpr::Or(xs) => {
            // OR: any child CanMatch → CanMatch; all CannotMatch → CannotMatch;
            // else Indeterminate.
            let mut all_cannot = true;
            let mut any_can = false;
            for x in xs {
                match evaluate_against_rg(x, rg, schema) {
                    RgVerdict::CanMatch => any_can = true,
                    RgVerdict::CannotMatch => {}
                    RgVerdict::Indeterminate => all_cannot = false,
                }
            }
            if any_can {
                RgVerdict::CanMatch
            } else if all_cannot {
                RgVerdict::CannotMatch
            } else {
                RgVerdict::Indeterminate
            }
        }
        // Not is rare (walker uses De Morgan); always conservative.
        PushedExpr::Not(_) => RgVerdict::Indeterminate,
    }
}

fn evaluate_leaf_against_rg(
    q: &PushedQual,
    rg: &parquet::file::metadata::RowGroupMetaData,
    schema: &Schema,
) -> RgVerdict {
    // Resolve column. If schema doesn't know about this column index, treat as
    // Indeterminate (caller is responsible for col-index correctness, but
    // defense-in-depth keeps us sound).
    if q.col >= schema.fields().len() {
        return RgVerdict::Indeterminate;
    }
    if q.col >= rg.num_columns() {
        return RgVerdict::Indeterminate;
    }
    let col = rg.column(q.col);
    let stats = match col.statistics() {
        Some(s) => s,
        None => return RgVerdict::Indeterminate,
    };

    // Null-tests are decidable purely from null_count.
    match q.op {
        PushedOp::IsNull => {
            return match stats.null_count_opt() {
                Some(0) => RgVerdict::CannotMatch,
                Some(_) => RgVerdict::CanMatch,
                None => RgVerdict::Indeterminate,
            };
        }
        PushedOp::IsNotNull => {
            let nc = stats.null_count_opt();
            let total = col.num_values() as u64;
            return match nc {
                Some(n) if n >= total => RgVerdict::CannotMatch,
                Some(_) => RgVerdict::CanMatch,
                None => RgVerdict::Indeterminate,
            };
        }
        _ => {}
    }

    // Comparison ops need typed min/max. Match on the qual's literal type and
    // pull the typed min/max from the variant-specific Statistics. Mixed-type
    // mismatches (column type doesn't match literal) → Indeterminate.
    use parquet::data_type::AsBytes;
    use parquet::file::statistics::Statistics as PqStats;
    match (&q.value, stats) {
        (ScalarValueRepr::I32(v), PqStats::Int32(s)) => {
            cmp_verdict(q.op, *v, s.min_opt().copied(), s.max_opt().copied())
        }
        (ScalarValueRepr::I64(v), PqStats::Int64(s)) => {
            cmp_verdict(q.op, *v, s.min_opt().copied(), s.max_opt().copied())
        }
        (ScalarValueRepr::I16(v), PqStats::Int32(s)) => {
            cmp_verdict(q.op, *v as i32, s.min_opt().copied(), s.max_opt().copied())
        }
        (ScalarValueRepr::F32(v), PqStats::Float(s)) => {
            cmp_verdict(q.op, *v, s.min_opt().copied(), s.max_opt().copied())
        }
        (ScalarValueRepr::F64(v), PqStats::Double(s)) => {
            cmp_verdict(q.op, *v, s.min_opt().copied(), s.max_opt().copied())
        }
        (ScalarValueRepr::Date32(v), PqStats::Int32(s)) => {
            cmp_verdict(q.op, *v, s.min_opt().copied(), s.max_opt().copied())
        }
        (ScalarValueRepr::TimestampMicros(v), PqStats::Int64(s)) => {
            cmp_verdict(q.op, *v, s.min_opt().copied(), s.max_opt().copied())
        }
        (ScalarValueRepr::Utf8(v), PqStats::ByteArray(s)) => {
            let lit = v.as_bytes();
            let mn = s.min_opt().map(|b| b.as_bytes().to_vec());
            let mx = s.max_opt().map(|b| b.as_bytes().to_vec());
            cmp_verdict_bytes(q.op, lit, mn.as_deref(), mx.as_deref())
        }
        (ScalarValueRepr::Bool(v), PqStats::Boolean(s)) => {
            // BooleanArray stats: min/max are bool. For Eq/Ne we can decide.
            match (s.min_opt(), s.max_opt()) {
                (Some(mn), Some(mx)) => match q.op {
                    PushedOp::Eq => {
                        if *v == *mn && *v == *mx {
                            RgVerdict::CanMatch
                        } else if *v != *mn && *v != *mx {
                            RgVerdict::CannotMatch
                        } else {
                            RgVerdict::CanMatch
                        }
                    }
                    PushedOp::Ne => {
                        if *v == *mn && *v == *mx {
                            RgVerdict::CannotMatch
                        } else {
                            RgVerdict::CanMatch
                        }
                    }
                    _ => RgVerdict::Indeterminate,
                },
                _ => RgVerdict::Indeterminate,
            }
        }
        // Decimal128 stats live in PqStats::FixedLenByteArray (BE-encoded i128).
        // For v1 keep this Indeterminate — extending it is straightforward but
        // out of scope; PG still re-evaluates above the scan.
        (ScalarValueRepr::Decimal128(..), _) => RgVerdict::Indeterminate,
        // Any other type-mismatch combination is Indeterminate (sound).
        _ => RgVerdict::Indeterminate,
    }
}

fn cmp_verdict<T: PartialOrd + Copy>(
    op: PushedOp,
    lit: T,
    mn: Option<T>,
    mx: Option<T>,
) -> RgVerdict {
    let (mn, mx) = match (mn, mx) {
        (Some(a), Some(b)) => (a, b),
        _ => return RgVerdict::Indeterminate,
    };
    match op {
        PushedOp::Eq => {
            if lit < mn || lit > mx {
                RgVerdict::CannotMatch
            } else {
                RgVerdict::CanMatch
            }
        }
        PushedOp::Ne => {
            if mn == mx && lit == mn {
                RgVerdict::CannotMatch
            } else {
                RgVerdict::CanMatch
            }
        }
        PushedOp::Lt => {
            if mn >= lit {
                RgVerdict::CannotMatch
            } else {
                RgVerdict::CanMatch
            }
        }
        PushedOp::Le => {
            if mn > lit {
                RgVerdict::CannotMatch
            } else {
                RgVerdict::CanMatch
            }
        }
        PushedOp::Gt => {
            if mx <= lit {
                RgVerdict::CannotMatch
            } else {
                RgVerdict::CanMatch
            }
        }
        PushedOp::Ge => {
            if mx < lit {
                RgVerdict::CannotMatch
            } else {
                RgVerdict::CanMatch
            }
        }
        PushedOp::IsNull | PushedOp::IsNotNull => unreachable!(),
    }
}

fn cmp_verdict_bytes(op: PushedOp, lit: &[u8], mn: Option<&[u8]>, mx: Option<&[u8]>) -> RgVerdict {
    let (mn, mx) = match (mn, mx) {
        (Some(a), Some(b)) => (a, b),
        _ => return RgVerdict::Indeterminate,
    };
    match op {
        PushedOp::Eq => {
            if lit < mn || lit > mx {
                RgVerdict::CannotMatch
            } else {
                RgVerdict::CanMatch
            }
        }
        PushedOp::Ne => {
            if mn == mx && lit == mn {
                RgVerdict::CannotMatch
            } else {
                RgVerdict::CanMatch
            }
        }
        PushedOp::Lt => {
            if mn >= lit {
                RgVerdict::CannotMatch
            } else {
                RgVerdict::CanMatch
            }
        }
        PushedOp::Le => {
            if mn > lit {
                RgVerdict::CannotMatch
            } else {
                RgVerdict::CanMatch
            }
        }
        PushedOp::Gt => {
            if mx <= lit {
                RgVerdict::CannotMatch
            } else {
                RgVerdict::CanMatch
            }
        }
        PushedOp::Ge => {
            if mx < lit {
                RgVerdict::CannotMatch
            } else {
                RgVerdict::CanMatch
            }
        }
        PushedOp::IsNull | PushedOp::IsNotNull => unreachable!(),
    }
}

/// Compute the lexicographic upper bound for a `LIKE 'prefix%'` translation.
///
/// Returns `Some(upper)` where every byte string lexicographically less than
/// `upper` and >= `prefix` is a valid candidate prefix-match. Returns `None`
/// if `prefix` is empty or every trailing byte is `0xFF` (no representable
/// upper bound).
///
/// Example: `next_lex_upper("ab")` returns `Some("ac".into())`. The walker
/// emits `col >= "ab" AND col < "ac"`.
pub fn next_lex_upper(prefix: &str) -> Option<String> {
    let mut bytes = prefix.as_bytes().to_vec();
    while let Some(last) = bytes.last_mut() {
        if *last < 0xFF {
            *last += 1;
            // The result may not be valid UTF-8 (we incremented a byte) — but
            // parquet ByteArray stats are also raw bytes, and the row-filter
            // path compares byte-wise as well. We hand back a `String` only
            // because `ScalarValueRepr::Utf8` wraps `String`; the bytes are
            // still byte-compared downstream.
            //
            // If the increment leaves invalid UTF-8, fall back to dropping.
            return String::from_utf8(bytes).ok();
        }
        bytes.pop();
    }
    None
}

/// Translate a `PushedExpr` whose leaves reference foreign-table attnums into
/// one whose leaves reference parquet column indices, via the
/// `storage_attno_to_parquet_idx` map. Returns `None` if any leaf points at a
/// partition column (caller bug — split_quals_by_target should have caught it).
pub fn translate_qual_to_parquet_idx(
    expr: PushedExpr,
    map: &[Option<usize>],
) -> Option<PushedExpr> {
    match expr {
        PushedExpr::Leaf(mut q) => {
            let p = map.get(q.col).and_then(|x| x.as_ref())?;
            q.col = *p;
            Some(PushedExpr::Leaf(q))
        }
        PushedExpr::And(xs) => {
            let translated: Option<Vec<_>> = xs
                .into_iter()
                .map(|x| translate_qual_to_parquet_idx(x, map))
                .collect();
            translated.map(PushedExpr::And)
        }
        PushedExpr::Or(xs) => {
            let translated: Option<Vec<_>> = xs
                .into_iter()
                .map(|x| translate_qual_to_parquet_idx(x, map))
                .collect();
            translated.map(PushedExpr::Or)
        }
        PushedExpr::Not(x) => {
            translate_qual_to_parquet_idx(*x, map).map(|t| PushedExpr::Not(Box::new(t)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pushed_op_inverse_round_trips() {
        for op in [
            PushedOp::Eq,
            PushedOp::Ne,
            PushedOp::Lt,
            PushedOp::Le,
            PushedOp::Gt,
            PushedOp::Ge,
            PushedOp::IsNull,
            PushedOp::IsNotNull,
        ] {
            let inv = op.inverse().expect("every PushedOp has an inverse");
            assert_eq!(
                inv.inverse(),
                Some(op),
                "inverse must be involutive for {op:?}"
            );
        }
    }
    #[test]
    fn pushed_expr_constructs() {
        let leaf = PushedExpr::Leaf(PushedQual {
            col: 0,
            op: PushedOp::Eq,
            value: ScalarValueRepr::I32(5),
        });
        let _ = PushedExpr::And(vec![leaf.clone()]);
        let _ = PushedExpr::Or(vec![leaf.clone()]);
        let _ = PushedExpr::Not(Box::new(leaf));
    }

    #[test]
    fn build_row_filter_simple_eq_returns_some() {
        use arrow::datatypes::{Field, Schema};
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::WriterProperties;
        use parquet::file::reader::FileReader;
        use parquet::file::serialized_reader::SerializedFileReader;

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        // Build a tiny parquet to get a SchemaDescriptor.
        let mut buf: Vec<u8> = Vec::new();
        {
            let w = ArrowWriter::try_new(
                &mut buf,
                schema.clone(),
                Some(WriterProperties::builder().build()),
            )
            .unwrap();
            w.close().unwrap();
        }
        let bytes = bytes::Bytes::from(buf);
        let reader = SerializedFileReader::new(bytes).unwrap();
        let parquet_schema = reader.metadata().file_metadata().schema_descr_ptr();

        let exprs = vec![PushedExpr::Leaf(PushedQual {
            col: 0,
            op: PushedOp::Eq,
            value: ScalarValueRepr::I32(5),
        })];
        assert!(build_row_filter(&exprs, &schema, &parquet_schema).is_some());
        assert!(build_row_filter(&[], &schema, &parquet_schema).is_none());
    }

    #[test]
    fn translate_storage_qual_maps_attno_to_parquet_idx() {
        let map = vec![None, None, Some(0), Some(1)]; // 2 partition, 2 storage
        let e = PushedExpr::Leaf(PushedQual {
            col: 2,
            op: PushedOp::Eq,
            value: ScalarValueRepr::I32(5),
        });
        let t = translate_qual_to_parquet_idx(e, &map).unwrap();
        if let PushedExpr::Leaf(q) = t {
            assert_eq!(q.col, 0);
        } else {
            panic!("expected Leaf");
        }
    }

    #[test]
    fn translate_partition_qual_returns_none() {
        let map = vec![None, Some(0)];
        let e = PushedExpr::Leaf(PushedQual {
            col: 0, // partition col
            op: PushedOp::Eq,
            value: ScalarValueRepr::I32(5),
        });
        assert!(translate_qual_to_parquet_idx(e, &map).is_none());
    }
}
