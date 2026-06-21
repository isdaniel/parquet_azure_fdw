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

/// Return `true` iff `(op, ty)` is on the pushdown whitelist defined in the
/// design spec §7.3.
///
/// Operator names are the textual form PG hands us (`"="`, `"<="`, …) plus the
/// synthetic `"IS NULL"` / `"IS NOT NULL"` tokens for null tests.
pub fn is_pushable(op: &str, ty: &DataType) -> bool {
    let op_ok = matches!(
        op,
        "=" | "<>" | "<" | "<=" | ">" | ">=" | "IS NULL" | "IS NOT NULL"
    );
    let ty_ok = matches!(
        ty,
        DataType::Boolean
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Float32
            | DataType::Float64
            | DataType::Utf8
            | DataType::Date32
            | DataType::Timestamp(_, _)
            | DataType::Decimal128(_, _)
    );
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
}
