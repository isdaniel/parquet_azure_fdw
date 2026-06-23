#![forbid(unsafe_code)]
//! Postgres → Arrow conversion: type mapping, schema construction, and
//! per-row builder appends for the §7.4 type matrix.
//!
//! Used by the INSERT/COPY path (Task 13) to accumulate a `RecordBatch` from
//! a stream of `TupleTableSlot` rows before flushing to parquet.
//!
//! The PG-side type matrix mirrors `arrow_to_pg.rs` so a write-then-read
//! roundtrip is type-stable.

/// Re-export so callers may use either `convert::datum_to_partition_string`
/// or the historical `convert::pg_to_arrow::datum_to_partition_string` path.
/// The implementation lives in the sibling `partition_datum` module because
/// it requires `unsafe` (varlena detoast) and this file is
/// `#![forbid(unsafe_code)]`.
pub use super::partition_datum::datum_to_partition_string;

use crate::error::{FdwError, FdwResult};
use arrow::array::{
    ArrayBuilder, ArrayRef, BooleanBuilder, Float32Builder, Float64Builder, Int16Builder,
    Int32Builder, Int64Builder, StringBuilder,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use arrow::record_batch::RecordBatch;
use pgrx::pg_sys;
use std::sync::Arc;

/// Map a Postgres type OID to the canonical Arrow `DataType` used by this
/// extension's write path. Mirrors the §7.4 matrix used by
/// `arrow_to_pg::arrow_value_to_datum` so a write-then-read roundtrip is
/// type-stable.
pub fn pg_oid_to_arrow_type(pg_oid: pg_sys::Oid) -> FdwResult<DataType> {
    let dt = match pg_oid {
        oid if oid == pg_sys::BOOLOID => DataType::Boolean,
        oid if oid == pg_sys::INT2OID => DataType::Int16,
        oid if oid == pg_sys::INT4OID => DataType::Int32,
        oid if oid == pg_sys::INT8OID => DataType::Int64,
        oid if oid == pg_sys::FLOAT4OID => DataType::Float32,
        oid if oid == pg_sys::FLOAT8OID => DataType::Float64,
        oid if oid == pg_sys::TEXTOID || oid == pg_sys::VARCHAROID => DataType::Utf8,
        oid if oid == pg_sys::BYTEAOID => DataType::Binary,
        oid if oid == pg_sys::DATEOID => DataType::Date32,
        oid if oid == pg_sys::TIMESTAMPOID => DataType::Timestamp(TimeUnit::Microsecond, None),
        oid if oid == pg_sys::TIMESTAMPTZOID => {
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
        }
        // Decimal128(38, 9) is the canonical numeric mapping on the v1 write
        // path: the read path accepts any Decimal128(p,s), so we pin one
        // precision/scale instead of plumbing per-table options. JSONB rides
        // the Utf8 builder (canonical text on disk).
        oid if oid == pg_sys::NUMERICOID => DataType::Decimal128(38, 9),
        oid if oid == pg_sys::JSONBOID => DataType::Utf8,
        other => {
            return Err(FdwError::UnsupportedType {
                pg_type: format!("oid={}", other.to_u32()),
                arrow_type: "<none>".to_string(),
            });
        }
    };
    Ok(dt)
}

/// Build an Arrow `Schema` from a slice of `FormData_pg_attribute`. Dropped
/// (system) columns should already be filtered by the caller.
pub fn pg_attrs_to_arrow_schema(attrs: &[pg_sys::FormData_pg_attribute]) -> FdwResult<SchemaRef> {
    let mut fields = Vec::with_capacity(attrs.len());
    for att in attrs {
        let dt = pg_oid_to_arrow_type(att.atttypid)?;
        // attname is a NameData (C string); fall back to a positional name if
        // it can't be decoded cleanly.
        let name = att_name(att).unwrap_or_else(|| format!("col{}", fields.len()));
        let nullable = !att.attnotnull;
        fields.push(Field::new(name, dt, nullable));
    }
    Ok(Arc::new(Schema::new(fields)))
}

fn att_name(att: &pg_sys::FormData_pg_attribute) -> Option<String> {
    // NameData.data is a fixed-size [c_char; NAMEDATALEN]. Read until NUL.
    let raw = &att.attname.data;
    let bytes: Vec<u8> = raw
        .iter()
        .take_while(|c| **c as u8 != 0)
        .map(|c| *c as u8)
        .collect();
    String::from_utf8(bytes).ok()
}

/// Column-aligned set of Arrow array builders for one in-progress
/// `RecordBatch`. Used by the INSERT/COPY path: callers `append_*` one row at
/// a time, then `finish()` to produce the batch.
///
/// Per-column appends route through typed helpers (`append_bool`,
/// `append_i64`, …) and through the cross-column dispatcher
/// `crate::fdw::modify::insert::append_one`, which is what the scan/modify
/// glue calls per slot column. There is no `append_slot(*mut
/// TupleTableSlot)` wrapper here — the live dispatcher handles that work
/// (and is exercised end-to-end by the §17 INSERT/COPY pg_test suite).
pub struct RecordBatchBuilders {
    schema: SchemaRef,
    builders: Vec<Box<dyn ArrayBuilder>>,
}

impl RecordBatchBuilders {
    /// Allocate one builder per field in `schema`, sized for `capacity` rows.
    pub fn new(schema: SchemaRef, capacity: usize) -> FdwResult<Self> {
        let mut builders: Vec<Box<dyn ArrayBuilder>> = Vec::with_capacity(schema.fields().len());
        for field in schema.fields() {
            builders.push(make_builder(field.data_type(), capacity)?);
        }
        Ok(Self { schema, builders })
    }

    /// Number of rows currently buffered.
    ///
    /// Derived from the first builder's length — callers append left-to-right
    /// so column 0 is the most conservative count. `finish()` validates that
    /// every column ends up at the same length.
    pub fn len(&self) -> usize {
        self.builders.first().map(|b| b.len()).unwrap_or(0)
    }

    /// Whether no rows have been appended yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append a typed value to column `col`. `None` records a NULL.
    pub fn append_bool(&mut self, col: usize, value: Option<bool>) -> FdwResult<()> {
        let b = self.typed_builder::<BooleanBuilder>(col)?;
        match value {
            Some(v) => b.append_value(v),
            None => b.append_null(),
        }
        Ok(())
    }

    pub fn append_i16(&mut self, col: usize, value: Option<i16>) -> FdwResult<()> {
        let b = self.typed_builder::<Int16Builder>(col)?;
        match value {
            Some(v) => b.append_value(v),
            None => b.append_null(),
        }
        Ok(())
    }

    pub fn append_i32(&mut self, col: usize, value: Option<i32>) -> FdwResult<()> {
        let b = self.typed_builder::<Int32Builder>(col)?;
        match value {
            Some(v) => b.append_value(v),
            None => b.append_null(),
        }
        Ok(())
    }

    pub fn append_i64(&mut self, col: usize, value: Option<i64>) -> FdwResult<()> {
        let b = self.typed_builder::<Int64Builder>(col)?;
        match value {
            Some(v) => b.append_value(v),
            None => b.append_null(),
        }
        Ok(())
    }

    pub fn append_f32(&mut self, col: usize, value: Option<f32>) -> FdwResult<()> {
        let b = self.typed_builder::<Float32Builder>(col)?;
        match value {
            Some(v) => b.append_value(v),
            None => b.append_null(),
        }
        Ok(())
    }

    pub fn append_f64(&mut self, col: usize, value: Option<f64>) -> FdwResult<()> {
        let b = self.typed_builder::<Float64Builder>(col)?;
        match value {
            Some(v) => b.append_value(v),
            None => b.append_null(),
        }
        Ok(())
    }

    /// Append a NULL to column `col` without needing to know its type.
    ///
    /// Walks the trait-object builder and dispatches via `as_any_mut` to call
    /// the typed `.append_null()`. Used by the modify path when the slot's
    /// `tts_isnull[i]` is true.
    pub fn append_null(&mut self, col: usize) -> FdwResult<()> {
        let b = self.builders.get_mut(col).ok_or_else(|| {
            FdwError::SchemaMismatch(format!(
                "column index {col} out of range (have {})",
                self.schema.fields().len()
            ))
        })?;
        let any = b.as_any_mut();
        if let Some(x) = any.downcast_mut::<BooleanBuilder>() {
            x.append_null();
        } else if let Some(x) = any.downcast_mut::<Int16Builder>() {
            x.append_null();
        } else if let Some(x) = any.downcast_mut::<Int32Builder>() {
            x.append_null();
        } else if let Some(x) = any.downcast_mut::<Int64Builder>() {
            x.append_null();
        } else if let Some(x) = any.downcast_mut::<Float32Builder>() {
            x.append_null();
        } else if let Some(x) = any.downcast_mut::<Float64Builder>() {
            x.append_null();
        } else if let Some(x) = any.downcast_mut::<StringBuilder>() {
            x.append_null();
        } else if let Some(x) = any.downcast_mut::<arrow::array::BinaryBuilder>() {
            x.append_null();
        } else if let Some(x) = any.downcast_mut::<arrow::array::Date32Builder>() {
            x.append_null();
        } else if let Some(x) = any.downcast_mut::<arrow::array::TimestampMicrosecondBuilder>() {
            x.append_null();
        } else if let Some(x) = any.downcast_mut::<arrow::array::Decimal128Builder>() {
            x.append_null();
        } else {
            return Err(FdwError::SchemaMismatch(format!(
                "no append_null dispatcher for column {col} builder"
            )));
        }
        Ok(())
    }

    pub fn append_date(&mut self, col: usize, value: Option<i32>) -> FdwResult<()> {
        let b = self.typed_builder::<arrow::array::Date32Builder>(col)?;
        match value {
            Some(v) => b.append_value(v),
            None => b.append_null(),
        }
        Ok(())
    }

    pub fn append_ts_us(&mut self, col: usize, value: Option<i64>) -> FdwResult<()> {
        let b = self.typed_builder::<arrow::array::TimestampMicrosecondBuilder>(col)?;
        match value {
            Some(v) => b.append_value(v),
            None => b.append_null(),
        }
        Ok(())
    }

    pub fn append_tstz_us(&mut self, col: usize, value: Option<i64>) -> FdwResult<()> {
        // Builder is the same TimestampMicrosecondBuilder; timezone metadata
        // lives on the Field, not the builder per-row.
        self.append_ts_us(col, value)
    }

    pub fn append_str(&mut self, col: usize, value: Option<&str>) -> FdwResult<()> {
        let b = self.typed_builder::<StringBuilder>(col)?;
        match value {
            Some(v) => b.append_value(v),
            None => b.append_null(),
        }
        Ok(())
    }

    pub fn append_bytes(&mut self, col: usize, value: Option<&[u8]>) -> FdwResult<()> {
        let b = self.typed_builder::<arrow::array::BinaryBuilder>(col)?;
        match value {
            Some(v) => b.append_value(v),
            None => b.append_null(),
        }
        Ok(())
    }

    pub fn append_decimal128(&mut self, col: usize, value: Option<i128>) -> FdwResult<()> {
        let b = self.typed_builder::<arrow::array::Decimal128Builder>(col)?;
        match value {
            Some(v) => b.append_value(v),
            None => b.append_null(),
        }
        Ok(())
    }

    pub fn append_jsonb_text(&mut self, col: usize, value: Option<&str>) -> FdwResult<()> {
        // JSONB is stored as canonical text on disk; reuse the Utf8 builder.
        self.append_str(col, value)
    }

    /// Finalize all builders into a `RecordBatch`.
    ///
    /// Errors if columns have diverged in length — callers must append every
    /// column for every row (NULL or otherwise) before constructing the batch.
    pub fn finish(mut self) -> FdwResult<RecordBatch> {
        let arrays: Vec<ArrayRef> = self.builders.iter_mut().map(|b| b.finish()).collect();
        if let Some(first) = arrays.first() {
            let expected = first.len();
            for (i, a) in arrays.iter().enumerate().skip(1) {
                if a.len() != expected {
                    return Err(FdwError::SchemaMismatch(format!(
                        "column {i} has {} rows, column 0 has {expected}; \
                         every column must be appended for every row",
                        a.len()
                    )));
                }
            }
        }
        RecordBatch::try_new(self.schema, arrays).map_err(FdwError::from)
    }

    fn typed_builder<T: ArrayBuilder>(&mut self, col: usize) -> FdwResult<&mut T> {
        let b = self.builders.get_mut(col).ok_or_else(|| {
            FdwError::SchemaMismatch(format!(
                "column index {col} out of range (have {})",
                self.schema.fields().len()
            ))
        })?;
        b.as_any_mut().downcast_mut::<T>().ok_or_else(|| {
            FdwError::SchemaMismatch(format!(
                "builder for column {col} is not a {}",
                std::any::type_name::<T>()
            ))
        })
    }
}

fn make_builder(dt: &DataType, capacity: usize) -> FdwResult<Box<dyn ArrayBuilder>> {
    Ok(match dt {
        DataType::Boolean => Box::new(BooleanBuilder::with_capacity(capacity)),
        DataType::Int16 => Box::new(Int16Builder::with_capacity(capacity)),
        DataType::Int32 => Box::new(Int32Builder::with_capacity(capacity)),
        DataType::Int64 => Box::new(Int64Builder::with_capacity(capacity)),
        DataType::Float32 => Box::new(Float32Builder::with_capacity(capacity)),
        DataType::Float64 => Box::new(Float64Builder::with_capacity(capacity)),
        DataType::Utf8 => Box::new(StringBuilder::with_capacity(capacity, capacity * 16)),
        DataType::Binary => Box::new(arrow::array::BinaryBuilder::with_capacity(
            capacity,
            capacity * 16,
        )),
        DataType::Date32 => Box::new(arrow::array::Date32Builder::with_capacity(capacity)),
        DataType::Timestamp(TimeUnit::Microsecond, None) => Box::new(
            arrow::array::TimestampMicrosecondBuilder::with_capacity(capacity),
        ),
        DataType::Timestamp(TimeUnit::Microsecond, Some(tz)) => Box::new(
            arrow::array::TimestampMicrosecondBuilder::with_capacity(capacity)
                .with_timezone(tz.clone()),
        ),
        DataType::Decimal128(p, s) => Box::new(
            arrow::array::Decimal128Builder::with_capacity(capacity)
                .with_precision_and_scale(*p, *s)
                .map_err(FdwError::from)?,
        ),
        // Binary / Date32 / Timestamp builders are added when their PG→Arrow
        // append helpers are wired in Task 13.
        other => {
            return Err(FdwError::UnsupportedType {
                pg_type: "<n/a>".to_string(),
                arrow_type: format!("{other:?}"),
            });
        }
    })
}
