//! Inference for broadcastable operations like Hadamard product, etc.
#![allow(clippy::too_many_arguments, clippy::needless_option_as_deref)]
use crate::context::ContextRef;
use crate::op_groups::{BroadcastOp, SimpleDtype};
use crate::{Shape, expr_text_range};
use lsp_types::DiagnosticSeverity;
use rustpython_parser::ast::Expr;
use rustpython_parser::text_size::TextRange;

use super::broadcast_dims;

pub enum ShapeOrExpr<'a> {
    Shape(Option<&'a Shape>),
    Expr(&'a Expr),
}

/// Inference shapes and produce element-wise broadcastable-operations.
pub fn infer_broadcastable_poswise(
    left: &ShapeOrExpr,
    right: &Expr,
    record_hovers: bool,
    whole_range: TextRange,
    broadcast_op: BroadcastOp,
    mut context: ContextRef,
) -> Option<Shape> {
    let inferred_left: Option<Shape>;
    let mut left_range = whole_range;
    let left_shape = match left {
        ShapeOrExpr::Shape(left_shape) => *left_shape,
        ShapeOrExpr::Expr(left_expr) => {
            left_range = expr_text_range(left_expr);
            inferred_left = context
                .lookup_shape(left_expr, record_hovers)
                .or_else(|| context.infer_shape(left_expr, false));
            inferred_left.as_ref()
        }
    };

    let right_shape = context
        .lookup_shape(right, record_hovers)
        .or_else(|| context.infer_shape(right, false));

    // check bitwise operator is applied int/bool, emit diagnostic otherwise
    if let BroadcastOp::Bitwise { only_right } = broadcast_op {
        for (idx, text_range, shape) in [
            (0, left_range, &left_shape),
            (1, expr_text_range(right), &right_shape.as_ref()),
        ] {
            if idx == 0 && only_right {
                // masked_fill only checks the right tensor for boolean (the mask)
                continue;
            }
            if let Some(Shape {
                dtype: Some(dtype), ..
            }) = shape
            {
                match (SimpleDtype::from(dtype.as_str()), only_right) {
                    (SimpleDtype::Float, false)
                    | (SimpleDtype::Int { .. } | SimpleDtype::Float, true) => context
                        .push_diagnostic_text(
                            text_range,
                            DiagnosticSeverity::ERROR,
                            format!(
                                "Bitwise operations only support integer or bool dtypes, found {dtype}"
                            ),
                        ),
                    _ =>(),

                }
            }
        }
    }

    match broadcastable_poswise(left_shape, right_shape) {
        Ok(mut shape_opt) => {
            if let (Some(Shape { dtype, .. }), BroadcastOp::Eq) = (&mut shape_opt, broadcast_op) {
                // ==, !=, torch.ge always return bool dtypes
                *dtype = Some("bool".to_string());
            }
            shape_opt
        }
        Err(msg) => {
            context.push_diagnostic_text(whole_range, DiagnosticSeverity::ERROR, msg);
            None
        }
    }
}

/// Element-wise (Hadamard) multiplication with torch-style broadcasting.
/// If only one of the shapes is known, returns that shape (scalar or unknown rhs/lhs).
fn broadcastable_poswise(
    left: Option<&Shape>,
    right: Option<Shape>,
) -> Result<Option<Shape>, String> {
    match (left, right) {
        (None, None) => Ok(None),
        (Some(s), None) => Ok(Some((*s).clone())),
        (None, Some(s)) => Ok(Some(s)),
        (Some(l), Some(r)) => {
            let dims = broadcast_dims(&l.dims, &r.dims)
                .map_err(|msg| format!("Broadcasting incompatible shapes: {msg}"))?;
            Ok(Some(Shape {
                dtype: l.dtype.clone().or(r.dtype.clone()),
                dims,
            }))
        }
    }
}
