#![deny(unsafe_op_in_unsafe_fn)]
//! Walk PG executor-residual quals into [`PushedExpr`] trees.
//!
//! Invoked at `BeginForeignScan` time against `(*foreign_scan).scan.plan.qual`
//! — by then PG has stripped `RestrictInfo` wrappers and the list is a plain
//! `List<Expr*>`. Walks each top-level expression independently; an entire
//! top-level expression that doesn't fully translate is **dropped**. Soundness
//! depends on every emitted `PushedExpr` being logically implied by its
//! source.
//!
//! Pushdown is advisory: PG's executor still evaluates every original qual.

use crate::convert::arrow_to_pg::UNIX_TO_PG_EPOCH_DAYS;
use crate::fdw::pg_op_oids::lookup_op;
use crate::fdw::pushdown::{PushedExpr, PushedOp, PushedQual, ScalarValueRepr};
use pgrx::pg_sys;

/// Shift a PG date (days-since-2000-01-01) to an Arrow Date32 value
/// (days-since-UNIX-epoch). Returns `None` on overflow — callers must drop
/// the qual in that case (sound default: PG still evaluates the original).
fn pg_to_unix_days(pg_days: i32) -> Option<i32> {
    pg_days.checked_add(UNIX_TO_PG_EPOCH_DAYS)
}

/// Walk a PG `List<Expr*>` of executor-residual quals.
///
/// # Safety
/// `qual_list` may be null; if non-null it must point to a live `List` of
/// `Expr*` cells (PG's executor guarantee for `Plan.qual`).
pub unsafe fn walk_quals(qual_list: *mut pg_sys::List) -> Vec<PushedExpr> {
    if qual_list.is_null() {
        return Vec::new();
    }
    // SAFETY: qual_list non-null and is a valid `List*` per caller contract.
    let n = unsafe { pg_sys::list_length(qual_list) };
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        // SAFETY: i in [0, list_length); list_nth bounds-checks internally.
        let node = unsafe { pg_sys::list_nth(qual_list, i) as *mut pg_sys::Node };
        if let Some(e) = unsafe { walk_node(node) } {
            out.push(e);
        }
    }
    out
}

/// Peel through `RelabelType` nodes (binary-compatible casts, e.g.
/// `varchar→text`). Loops to handle the (rare) nested-relabel case. Other
/// coercion nodes (`CoerceViaIO`, `FuncExpr`) are intentionally NOT peeled —
/// those carry semantic conversion (timezone, precision) we don't model.
///
/// # Safety
/// `node` may be null; if non-null it must be a live `Node*` whose `nodeTag`
/// accurately identifies its concrete type (PG invariant). The returned
/// pointer is null iff the input was null or a `RelabelType` chain bottomed
/// out at a null `arg`.
unsafe fn peel_relabel(mut node: *mut pg_sys::Node) -> *mut pg_sys::Node {
    // SAFETY: each loop iteration re-reads `type_` only after a non-null check;
    // `arg` is documented as an `Expr*` (Node-prefixed) by PG.
    while !node.is_null() && unsafe { (*node).type_ } == pg_sys::NodeTag::T_RelabelType {
        let r = node as *mut pg_sys::RelabelType;
        // SAFETY: confirmed T_RelabelType above; `arg` field is in-layout.
        node = unsafe { (*r).arg } as *mut pg_sys::Node;
    }
    node
}

/// Walk one PG expression node. Returns `None` for anything we don't
/// translate; the caller drops it (PG still evaluates the original).
///
/// # Safety
/// `node` may be null; if non-null it must be a live PG `Node*` whose
/// `nodeTag` accurately identifies its concrete type (PG invariant).
unsafe fn walk_node(node: *mut pg_sys::Node) -> Option<PushedExpr> {
    if node.is_null() {
        return None;
    }
    // Peel binary-compatible RelabelType wrappers (e.g. varchar→text). Other
    // coercion nodes (CoerceViaIO, FuncExpr) are intentionally not peeled —
    // they carry semantic conversion (tz, precision) we don't model here.
    // SAFETY: node non-null per caller contract; peel_relabel keeps that.
    let node = unsafe { peel_relabel(node) };
    if node.is_null() {
        return None;
    }
    // SAFETY: PG sets `type_` on every Node; reading it is a single u32 load.
    let tag = unsafe { (*node).type_ };
    match tag {
        pg_sys::NodeTag::T_OpExpr => unsafe { walk_op_expr(node as *mut pg_sys::OpExpr) },
        pg_sys::NodeTag::T_BoolExpr => unsafe { walk_bool_expr(node as *mut pg_sys::BoolExpr) },
        pg_sys::NodeTag::T_NullTest => unsafe { walk_null_test(node as *mut pg_sys::NullTest) },
        pg_sys::NodeTag::T_BooleanTest => unsafe {
            walk_boolean_test(node as *mut pg_sys::BooleanTest)
        },
        pg_sys::NodeTag::T_ScalarArrayOpExpr => unsafe {
            walk_scalar_array_op(node as *mut pg_sys::ScalarArrayOpExpr)
        },
        _ => None,
    }
}

/// Apply De Morgan / `op.inverse()` to flip a `PushedExpr` without losing
/// soundness. Returns `None` if any leaf op lacks an inverse (or any nested
/// sub-tree can't be inverted).
fn invert_pushed(e: PushedExpr) -> Option<PushedExpr> {
    Some(match e {
        PushedExpr::Leaf(q) => PushedExpr::Leaf(PushedQual {
            col: q.col,
            op: q.op.inverse()?,
            value: q.value,
        }),
        PushedExpr::And(xs) => PushedExpr::Or(
            xs.into_iter()
                .map(invert_pushed)
                .collect::<Option<Vec<_>>>()?,
        ),
        PushedExpr::Or(xs) => PushedExpr::And(
            xs.into_iter()
                .map(invert_pushed)
                .collect::<Option<Vec<_>>>()?,
        ),
        PushedExpr::Not(inner) => *inner,
    })
}

/// Walk a `BoolExpr` — AND (lossy: drops unpushable conjuncts), OR
/// (all-or-nothing), or NOT (pushed through via De Morgan).
///
/// # Safety
/// `expr` is a live `BoolExpr*` (caller verified via nodeTag).
unsafe fn walk_bool_expr(expr: *mut pg_sys::BoolExpr) -> Option<PushedExpr> {
    // SAFETY: caller verified BoolExpr layout via nodeTag.
    let (bool_op, args) = unsafe { ((*expr).boolop, (*expr).args) };
    if args.is_null() {
        return None;
    }
    let mut children = Vec::new();
    // SAFETY: args is a live List* per BoolExpr layout.
    let n = unsafe { pg_sys::list_length(args) };
    for i in 0..n {
        // SAFETY: i in [0, list_length); list_nth bounds-checks internally.
        let child_node = unsafe { pg_sys::list_nth(args, i) as *mut pg_sys::Node };
        if let Some(c) = unsafe { walk_node(child_node) } {
            children.push(c);
        }
        // For AND: dropping an unpushable conjunct is sound (PG re-evaluates).
    }
    match bool_op {
        pg_sys::BoolExprType::AND_EXPR if !children.is_empty() => {
            if children.len() == 1 {
                Some(children.pop().unwrap())
            } else {
                Some(PushedExpr::And(children))
            }
        }
        // OR: all-or-nothing. Dropping any child changes semantics (the rows it
        // would have admitted would be lost).
        pg_sys::BoolExprType::OR_EXPR if children.len() as i32 == n => {
            if children.len() == 1 {
                Some(children.pop().unwrap())
            } else {
                Some(PushedExpr::Or(children))
            }
        }
        // NOT: push through known shapes via De Morgan + op.inverse(). If the
        // single child failed to walk (children.len() == 0) we drop the NOT.
        pg_sys::BoolExprType::NOT_EXPR if children.len() == 1 => {
            let inner = children.pop().unwrap();
            invert_pushed(inner)
        }
        _ => None,
    }
}

/// Walk a `NullTest` (IS NULL / IS NOT NULL on a Var).
///
/// # Safety
/// `t` is a live `NullTest*` (caller verified via nodeTag).
unsafe fn walk_null_test(t: *mut pg_sys::NullTest) -> Option<PushedExpr> {
    // SAFETY: caller verified NullTest layout via nodeTag.
    let (arg, kind) = unsafe { ((*t).arg, (*t).nulltesttype) };
    // SAFETY: arg is an Expr* (Node-prefixed) per NullTest layout.
    let col = unsafe { try_var(arg as *mut pg_sys::Node) }?;
    let op = match kind {
        pg_sys::NullTestType::IS_NULL => PushedOp::IsNull,
        pg_sys::NullTestType::IS_NOT_NULL => PushedOp::IsNotNull,
        _ => return None,
    };
    Some(PushedExpr::Leaf(PushedQual {
        col,
        op,
        value: ScalarValueRepr::Null,
    }))
}

/// Walk a `BooleanTest`. `x IS TRUE` → `x = true`, `x IS FALSE` → `x = false`.
/// Other variants (IS NOT TRUE / NOT FALSE / [NOT] UNKNOWN) are dropped — they
/// have NULL-handling semantics we don't model with a simple Eq.
///
/// # Safety
/// `t` is a live `BooleanTest*` (caller verified via nodeTag).
unsafe fn walk_boolean_test(t: *mut pg_sys::BooleanTest) -> Option<PushedExpr> {
    // SAFETY: caller verified BooleanTest layout via nodeTag.
    let (arg, kind) = unsafe { ((*t).arg, (*t).booltesttype) };
    // SAFETY: arg is an Expr* (Node-prefixed) per BooleanTest layout.
    let col = unsafe { try_var(arg as *mut pg_sys::Node) }?;
    let val = match kind {
        pg_sys::BoolTestType::IS_TRUE => ScalarValueRepr::Bool(true),
        pg_sys::BoolTestType::IS_FALSE => ScalarValueRepr::Bool(false),
        _ => return None,
    };
    Some(PushedExpr::Leaf(PushedQual {
        col,
        op: PushedOp::Eq,
        value: val,
    }))
}

/// Translate `Var op Const` (or `Const op Var` — flipped).
///
/// # Safety
/// `expr` is a live `OpExpr*` with `type_ == T_OpExpr`.
unsafe fn walk_op_expr(expr: *mut pg_sys::OpExpr) -> Option<PushedExpr> {
    // SAFETY: expr is a live OpExpr (caller contract via nodeTag).
    let (opno, args) = unsafe { ((*expr).opno, (*expr).args) };
    let (pop, _lhs_ty, _rhs_ty) = lookup_op(opno)?;
    if args.is_null() {
        return None;
    }
    // SAFETY: args is a live List* of Expr* per OpExpr layout.
    if unsafe { pg_sys::list_length(args) } != 2 {
        return None;
    }
    // SAFETY: list has length 2; indices 0 and 1 are in bounds.
    let lhs = unsafe { pg_sys::list_nth(args, 0) as *mut pg_sys::Node };
    // SAFETY: same as above.
    let rhs = unsafe { pg_sys::list_nth(args, 1) as *mut pg_sys::Node };
    // Try Var op Const, then Const op Var (flip op).
    if let (Some(col), Some(val)) = (unsafe { try_var(lhs) }, unsafe { try_const(rhs) }) {
        return Some(PushedExpr::Leaf(PushedQual {
            col,
            op: pop,
            value: val,
        }));
    }
    if let (Some(val), Some(col)) = (unsafe { try_const(lhs) }, unsafe { try_var(rhs) }) {
        let flipped = match pop {
            PushedOp::Lt => PushedOp::Gt,
            PushedOp::Gt => PushedOp::Lt,
            PushedOp::Le => PushedOp::Ge,
            PushedOp::Ge => PushedOp::Le,
            other => other,
        };
        return Some(PushedExpr::Leaf(PushedQual {
            col,
            op: flipped,
            value: val,
        }));
    }
    None
}

/// If `node` is a Var (and not a system column), return its 0-based attno.
///
/// # Safety
/// `node` non-null + has a valid `nodeTag`.
unsafe fn try_var(node: *mut pg_sys::Node) -> Option<usize> {
    if node.is_null() {
        return None;
    }
    // SAFETY: caller contract; peel_relabel preserves null-or-live-Node.
    let node = unsafe { peel_relabel(node) };
    if node.is_null() {
        return None;
    }
    // SAFETY: caller contract; nodeTag read.
    if unsafe { (*node).type_ } != pg_sys::NodeTag::T_Var {
        return None;
    }
    let v = node as *mut pg_sys::Var;
    // SAFETY: confirmed T_Var.
    let attno = unsafe { (*v).varattno };
    if attno < 1 {
        return None;
    }
    Some((attno - 1) as usize)
}

/// If `node` is a non-null Const, lower it to `ScalarValueRepr`.
///
/// # Safety
/// `node` non-null + valid `nodeTag`.
unsafe fn try_const(node: *mut pg_sys::Node) -> Option<ScalarValueRepr> {
    if node.is_null() {
        return None;
    }
    // SAFETY: caller contract; peel_relabel preserves null-or-live-Node.
    let node = unsafe { peel_relabel(node) };
    if node.is_null() {
        return None;
    }
    // SAFETY: caller contract.
    if unsafe { (*node).type_ } != pg_sys::NodeTag::T_Const {
        return None;
    }
    let c = node as *mut pg_sys::Const;
    // SAFETY: confirmed T_Const.
    let (isnull, ty, datum) = unsafe { ((*c).constisnull, (*c).consttype, (*c).constvalue) };
    // NULL in a comparison op is unsound to push; NullTest builds IS NULL leaves directly.
    if isnull {
        return None;
    }
    // Lower by PG type OID (only a handful of types — keep tight).
    //
    // PG14/15 do not FFI-export the `DatumGet{Int16,Int32,Int64,Bool}`
    // inlines; on every PG version those are pure bit-casts of the underlying
    // `uintptr_t` Datum word, so we reproduce them via `Datum::value()` to
    // stay version-portable (matches the pattern in `fdw/modify/insert.rs`).
    // `DatumGetFloat4` / `DatumGetFloat8` ARE exported and need the float
    // bit-reinterpret, so we keep calling them.
    Some(match ty.to_u32() {
        21 => ScalarValueRepr::I16(datum.value() as i16),
        23 => ScalarValueRepr::I32(datum.value() as i32),
        20 => ScalarValueRepr::I64(datum.value() as i64),
        // SAFETY: confirmed T_Const non-null with FLOAT4OID.
        700 => ScalarValueRepr::F32(unsafe { pg_sys::DatumGetFloat4(datum) }),
        // SAFETY: confirmed T_Const non-null with FLOAT8OID.
        701 => ScalarValueRepr::F64(unsafe { pg_sys::DatumGetFloat8(datum) }),
        25 | 1043 => {
            // text / varchar. Detoast then convert to a Rust String.
            // SAFETY: a non-null TEXTOID/VARCHAROID Const carries a live
            // `varlena*` payload; helper handles detoast + pfree of intermediates.
            ScalarValueRepr::Utf8(unsafe {
                text_datum_to_string(datum.cast_mut_ptr::<pg_sys::varlena>())
            }?)
        }
        // PG date is days-since-2000-01-01; Arrow Date32 is days-since-UNIX-epoch.
        1082 => {
            let pg_days = datum.value() as i32;
            ScalarValueRepr::Date32(pg_to_unix_days(pg_days)?)
        }
        _ => return None,
    })
}

/// Detoast a `varlena*` text/varchar payload and copy it into an owned String,
/// freeing the palloc'd intermediates so we don't leak in long-lived memory
/// contexts. Returns `None` on detoast failure or if the bytes aren't UTF-8.
///
/// # Safety
/// `vl` must point to a live varlena (text/varchar) or be null.
unsafe fn text_datum_to_string(vl: *mut pg_sys::varlena) -> Option<String> {
    if vl.is_null() {
        return None;
    }
    // SAFETY: caller contract: vl is a live varlena.
    let detoasted = unsafe { pg_sys::pg_detoast_datum(vl) };
    if detoasted.is_null() {
        return None;
    }
    // SAFETY: `text_to_cstring` returns a palloc'd NUL-terminated copy of the
    // text payload.
    let cstr = unsafe { pg_sys::text_to_cstring(detoasted as *const pg_sys::text) };
    if cstr.is_null() {
        if !core::ptr::eq(detoasted, vl) {
            // SAFETY: pfree the detoasted copy when it was a separately
            // allocated palloc'd buffer.
            unsafe { pg_sys::pfree(detoasted.cast()) };
        }
        return None;
    }
    // SAFETY: NUL-terminated palloc'd C string just returned above.
    let s_opt = unsafe { core::ffi::CStr::from_ptr(cstr) }
        .to_str()
        .ok()
        .map(|s| s.to_owned());
    // SAFETY: pfree the cstring (always palloc'd) and the detoast copy when
    // distinct from the input pointer.
    unsafe {
        pg_sys::pfree(cstr.cast());
        if !core::ptr::eq(detoasted, vl) {
            pg_sys::pfree(detoasted.cast());
        }
    }
    s_opt
}

/// Walk `x = ANY(ARRAY[...])` → `Or([x=a, x=b, ...])`.
/// Walk `x <> ALL(ARRAY[...])` → `And([x<>a, x<>b, ...])`.
///
/// Other `ScalarArrayOpExpr` shapes (`<>` with ANY, `<` with ALL, ...) are
/// dropped — they require row-level semantics we don't model in the Or/And
/// expansion above.
///
/// # Safety
/// `e` is a live `ScalarArrayOpExpr*` (caller verified via nodeTag).
unsafe fn walk_scalar_array_op(e: *mut pg_sys::ScalarArrayOpExpr) -> Option<PushedExpr> {
    // SAFETY: caller verified ScalarArrayOpExpr layout via nodeTag.
    let (opno, use_or, args) = unsafe { ((*e).opno, (*e).useOr, (*e).args) };
    let (op, _, _) = lookup_op(opno)?;
    // Only push `= ANY` (IN) and `<> ALL` (NOT IN); other ANY/ALL combos
    // need richer logic than a simple Or/And of leaves.
    let is_in = use_or && op == PushedOp::Eq;
    let is_not_in = !use_or && op == PushedOp::Ne;
    if !(is_in || is_not_in) {
        return None;
    }
    if args.is_null() {
        return None;
    }
    // SAFETY: args is a live List* per ScalarArrayOpExpr layout.
    if unsafe { pg_sys::list_length(args) } != 2 {
        return None;
    }
    // SAFETY: list has length 2; indices 0/1 are in bounds.
    let var_node = unsafe { pg_sys::list_nth(args, 0) as *mut pg_sys::Node };
    // SAFETY: same as above.
    let arr_node = unsafe { pg_sys::list_nth(args, 1) as *mut pg_sys::Node };
    let col = unsafe { try_var(var_node) }?;
    if arr_node.is_null() {
        return None;
    }
    // SAFETY: arr_node non-null + valid nodeTag per caller contract.
    if unsafe { (*arr_node).type_ } != pg_sys::NodeTag::T_Const {
        return None;
    }
    let scalars = unsafe { try_array_const(arr_node as *mut pg_sys::Const) }?;
    if scalars.is_empty() {
        return None;
    }
    let leaves: Vec<PushedExpr> = scalars
        .into_iter()
        .map(|v| PushedExpr::Leaf(PushedQual { col, op, value: v }))
        .collect();
    Some(if leaves.len() == 1 {
        leaves.into_iter().next().unwrap()
    } else if is_in {
        PushedExpr::Or(leaves)
    } else {
        PushedExpr::And(leaves)
    })
}

/// Lower an array `Const` to `Vec<ScalarValueRepr>`. Supports the same scalar
/// element types as [`try_const`]. NULL elements abort the whole array
/// (returns `None`) — SQL `IN` / `NOT IN` NULL semantics aren't safely
/// representable as an Or/And of equality leaves.
///
/// # Safety
/// `c` is a live `Const*` of array type.
unsafe fn try_array_const(c: *mut pg_sys::Const) -> Option<Vec<ScalarValueRepr>> {
    // SAFETY: caller contract; const layout.
    let (isnull, datum) = unsafe { ((*c).constisnull, (*c).constvalue) };
    if isnull {
        return None;
    }
    // SAFETY: a non-null array Const carries a live varlena-prefixed
    // `ArrayType*`; detoast normalises it for `deconstruct_array`.
    let arr_vl = unsafe { pg_sys::pg_detoast_datum(datum.cast_mut_ptr::<pg_sys::varlena>()) };
    if arr_vl.is_null() {
        return None;
    }
    let arr = arr_vl as *mut pg_sys::ArrayType;
    // SAFETY: arr points to a live ArrayType post-detoast.
    let elem_ty = unsafe { (*arr).elemtype };
    let mut elem_len: i16 = 0;
    let mut elem_byval: bool = false;
    let mut elem_align: core::ffi::c_char = 0;
    // SAFETY: out-params are valid stack locals; FFI fills them per signature.
    unsafe {
        pg_sys::get_typlenbyvalalign(elem_ty, &mut elem_len, &mut elem_byval, &mut elem_align);
    }
    let mut values_ptr: *mut pg_sys::Datum = core::ptr::null_mut();
    let mut nulls_ptr: *mut bool = core::ptr::null_mut();
    let mut count: core::ffi::c_int = 0;
    // SAFETY: arr is a live ArrayType; out-pointers are valid locals;
    // `deconstruct_array` palloc's `*values_ptr` and `*nulls_ptr` arrays of
    // length `count` in the current memory context.
    unsafe {
        pg_sys::deconstruct_array(
            arr,
            elem_ty,
            elem_len as core::ffi::c_int,
            elem_byval,
            elem_align,
            &mut values_ptr,
            &mut nulls_ptr,
            &mut count,
        );
    }
    if count < 0 || values_ptr.is_null() || nulls_ptr.is_null() {
        return None;
    }
    let mut out = Vec::with_capacity(count as usize);
    for i in 0..count as isize {
        // SAFETY: deconstruct_array filled both arrays with `count` slots.
        let is_null = unsafe { *nulls_ptr.offset(i) };
        if is_null {
            // Conservative: NULL element changes IN/NOT-IN semantics.
            return None;
        }
        // SAFETY: same bounds reasoning as nulls_ptr.
        let d = unsafe { *values_ptr.offset(i) };
        let v = match elem_ty.to_u32() {
            21 => ScalarValueRepr::I16(d.value() as i16),
            23 => ScalarValueRepr::I32(d.value() as i32),
            20 => ScalarValueRepr::I64(d.value() as i64),
            // SAFETY: elem is FLOAT4OID — bit-reinterpret via FFI.
            700 => ScalarValueRepr::F32(unsafe { pg_sys::DatumGetFloat4(d) }),
            // SAFETY: elem is FLOAT8OID — bit-reinterpret via FFI.
            701 => ScalarValueRepr::F64(unsafe { pg_sys::DatumGetFloat8(d) }),
            25 | 1043 => {
                // SAFETY: elem is TEXT/VARCHAR; payload is a live varlena.
                let s = unsafe { text_datum_to_string(d.cast_mut_ptr::<pg_sys::varlena>()) }?;
                ScalarValueRepr::Utf8(s)
            }
            1082 => {
                // PG date is days-since-2000-01-01; Arrow Date32 is from UNIX
                // epoch. Drop the qual on overflow (sound default).
                let pg_days = d.value() as i32;
                ScalarValueRepr::Date32(pg_to_unix_days(pg_days)?)
            }
            _ => return None,
        };
        out.push(v);
    }
    Some(out)
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use super::*;
    use pgrx::prelude::*;

    /// Walk a quals list extracted via planner round-trip. We use Spi to plan
    /// a SELECT and the FDW pushdown is verified by the end-to-end test
    /// (Task 6); here we only assert the walker compiles and `walk_quals(NULL)`
    /// is empty.
    #[pg_test]
    fn walk_quals_null_is_empty() {
        // SAFETY: explicit null is the contract-allowed input.
        let v = unsafe { walk_quals(core::ptr::null_mut()) };
        assert!(v.is_empty());
    }

    #[test]
    fn pg_to_unix_days_shifts_epoch() {
        assert_eq!(pg_to_unix_days(0), Some(10957));
        assert_eq!(pg_to_unix_days(i32::MAX), None);
    }

    /// Invert `AND(Eq, Lt)` and expect `OR(Ne, Ge)` (De Morgan + op.inverse()).
    /// Pure logic — no Postgres needed.
    #[test]
    fn invert_pushed_and_to_or_swaps_ops() {
        let inner = PushedExpr::And(vec![
            PushedExpr::Leaf(PushedQual {
                col: 0,
                op: PushedOp::Eq,
                value: ScalarValueRepr::I32(1),
            }),
            PushedExpr::Leaf(PushedQual {
                col: 1,
                op: PushedOp::Lt,
                value: ScalarValueRepr::I32(2),
            }),
        ]);
        let got = invert_pushed(inner).expect("Eq and Lt both invert");
        let expected = PushedExpr::Or(vec![
            PushedExpr::Leaf(PushedQual {
                col: 0,
                op: PushedOp::Ne,
                value: ScalarValueRepr::I32(1),
            }),
            PushedExpr::Leaf(PushedQual {
                col: 1,
                op: PushedOp::Ge,
                value: ScalarValueRepr::I32(2),
            }),
        ]);
        assert_eq!(got, expected);
    }

    /// `Not(inner)` short-circuits to `inner` (double-negation elimination).
    #[test]
    fn invert_pushed_strips_not() {
        let leaf = PushedExpr::Leaf(PushedQual {
            col: 0,
            op: PushedOp::Eq,
            value: ScalarValueRepr::I32(1),
        });
        let nested = PushedExpr::Not(Box::new(leaf.clone()));
        assert_eq!(invert_pushed(nested), Some(leaf));
    }

    /// `peel_relabel(null)` is the null-input contract: must round-trip null.
    /// Building a real `RelabelType` requires a live PG memory context (we
    /// have `walk_quals_null_is_empty` for that path under `#[pg_test]`), so
    /// the pure-logic check we can do without PG is the null pass-through.
    #[test]
    fn peel_relabel_null_is_null() {
        // SAFETY: explicit null is the contract-allowed input.
        let out = unsafe { peel_relabel(core::ptr::null_mut()) };
        assert!(out.is_null());
    }
}
