//! Inference for broadcastable operations like Hadamard product, etc.
#![allow(clippy::too_many_arguments, clippy::needless_option_as_deref)]
use crate::op_groups::{BroadcastOp, SimpleDtype};
use crate::{ClassMap, FuncMap, HoverInfo, Imports, ModuleCache, Shape, VarState, expr_text_range};
use lsp_types::{Diagnostic, DiagnosticSeverity, Range};
use rustpython_parser::ast::{Expr, Identifier};
use rustpython_parser::text_size::TextRange;
use std::collections::HashMap;
use std::path::Path;

use super::broadcast_dims;
use crate::{infer_expr_shape, lookup_shape, text_range_to_lsp};

pub enum ShapeOrExpr<'a> {
    Shape(Option<&'a Shape>),
    Expr(&'a Expr),
}

/// Inference shapes and produce element-wise broadcastable-operations.
pub fn infer_broadcastable_poswise(
    left: &ShapeOrExpr,
    right: &Expr,
    vars: &HashMap<Identifier, VarState>,
    func_map: &FuncMap,
    imports: &Imports,
    class_map: &ClassMap,
    call_stack: &mut Vec<Identifier>,
    diagnostics: &mut Vec<Diagnostic>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    record_hovers: bool,
    source: &str,
    whole_range: TextRange,
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
    broadcast_op: BroadcastOp,
) -> Option<Shape> {
    let inferred_left: Option<Shape>;
    let mut left_range = whole_range;
    let left_shape = match left {
        ShapeOrExpr::Shape(left_shape) => *left_shape,
        ShapeOrExpr::Expr(left_expr) => {
            left_range = expr_text_range(left_expr);
            inferred_left = lookup_shape(left_expr, vars, hover_entries, record_hovers, source)
                .or_else(|| {
                    infer_expr_shape(
                        left_expr,
                        vars,
                        func_map,
                        imports,
                        class_map,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        false,
                        source,
                        module_cache.as_deref_mut(),
                        module_path,
                    )
                });
            inferred_left.as_ref()
        }
    };

    let right_shape =
        lookup_shape(right, vars, hover_entries, record_hovers, source).or_else(|| {
            infer_expr_shape(
                right,
                vars,
                func_map,
                imports,
                class_map,
                call_stack,
                diagnostics,
                hover_entries,
                false,
                source,
                module_cache.as_deref_mut(),
                module_path,
            )
        });

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
                    (SimpleDtype::Float, false) | (SimpleDtype::Int{..} | SimpleDtype::Float, true) => diagnostics.push(Diagnostic {
                        range: text_range_to_lsp(text_range, source),
                        severity: Some(DiagnosticSeverity::ERROR),
                        code: None,
                        code_description: None,
                        source: Some("shapels".into()),
                        message: format!(
                            "Bitwise operations only support integer or bool dtypes, found {dtype}"
                        ),
                        related_information: None,
                        tags: None,
                        data: None,
                    }),
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
            diagnostics.push(Diagnostic {
                range: text_range_to_lsp(whole_range, source),
                severity: Some(DiagnosticSeverity::ERROR),
                code: None,
                code_description: None,
                source: Some("shapels".into()),
                message: msg,
                related_information: None,
                tags: None,
                data: None,
            });
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
