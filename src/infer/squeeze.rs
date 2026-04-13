//! Inference for squeeze, reduce/aggregation functions (like sum, mean) and unsqueeze.
use crate::{Shape, VarState, expr_text_range, text_range_to_lsp};
use lsp_types::{Diagnostic, DiagnosticSeverity};
use rustpython_parser::ast::{Constant, Expr, ExprBinOp, ExprConstant, Identifier, Operator};
use rustpython_parser::text_size::TextRange;
use std::collections::HashMap;

use super::{base_shape_or_diag, expr_to_dim_token, infer_matmul_shapes};
use crate::context::ContextRef;

/// Shared logic for squeeze (enforce_one = true) and aggregation (enforce_one = false).
///
/// An aggregation (or reduce) operation such as sum or amin is shape-wise the same
/// as an squeeze only that squeezes only applies to dim==1 and should diagnose otherwise.
pub fn infer_squeeze(
    base_expr: &Expr,
    dim_arg: Option<&Expr>,
    q_arg: Option<&Expr>,
    record_hovers: bool,
    whole_range: TextRange,
    enforce_one: bool,
    keepdim: Option<&Expr>,
    mut context: ContextRef,
) -> Option<Shape> {
    let diag_before = context.diagnostics.len();
    let base_shape = base_shape_or_diag(base_expr, record_hovers, context.reborrow())
        .or_else(|| infer_shallow_shape(base_expr, record_hovers, context.reborrow()))?;

    let q_arg_dims = q_arg.map(|q_expr| {
        // Scalar q keeps the same reduction behavior as regular aggregations.
        let (vars, diagnostics, source) = context.vars_diagnostics_source();
        if expr_to_dim_token(q_expr, vars, diagnostics, source, &mut false)
            .is_some_and(|token| token.parse::<f64>().is_ok())
        {
            Vec::new()
        } else if let Some(shape) = context.lookup_shape(q_expr, record_hovers) {
            shape.dims
        } else {
            vec!["q".to_string()]
        }
    });
    let dims_to_remove = if let Some(dim) = dim_arg {
        let (vars, diagnostics, source) = context.vars_diagnostics_source();
        match parse_dims(dim, base_shape.dims.len(), vars, diagnostics, source) {
            Ok(v) => v,
            Err(_) => return None,
        }
    } else {
        // No dim specified: squeeze removes ones, sum collapses all dims.
        let mut dims = base_shape.dims.clone();
        if enforce_one {
            dims.retain(|d| d != "1");
        } else {
            dims.clear();
        }
        if let Some(mut q_dims) = q_arg_dims {
            q_dims.extend(dims);
            dims = q_dims;
        }
        return Some(Shape {
            dtype: base_shape.dtype.clone(),
            dims,
        });
    };

    let keepdim = keepdim
        .map(|expr| {
            if let Expr::Constant(ExprConstant { value, .. }) = expr {
                match value {
                    Constant::Bool(b) => *b,
                    Constant::Int(i) => i.to_string() != "0",
                    _ => {
                        context.push_diagnostic_text(
                            whole_range,
                            DiagnosticSeverity::INFORMATION,
                            "keepdim argument was not understood; only constant False/True or 0/1 are supported.".into(),
                        );
                        false
                    }
                }
            } else {
                context.push_diagnostic_text(
                    whole_range,
                    DiagnosticSeverity::INFORMATION,
                    "keepdim argument was not understood; only constant False/True or 0/1 are supported.".into(),
                );
                false
            }
        })
        .unwrap_or(false);
    let mut dims = base_shape.dims.clone();
    for idx in dims_to_remove.into_iter().rev() {
        if idx >= dims.len() {
            context.push_diagnostic_text(
                whole_range,
                DiagnosticSeverity::ERROR,
                "Invalid dim".into(),
            );
            continue;
        }
        if let Some(d) = dims.get(idx) {
            if enforce_one && d != "1" {
                context.push_diagnostic_text(
                    whole_range,
                    d.parse::<i32>()
                        .map_or(DiagnosticSeverity::WARNING, |_| DiagnosticSeverity::ERROR),
                    "Cannot squeeze dimension not equal to 1".into(),
                );
                continue;
            } else if keepdim {
                dims[idx] = String::from("1");
                continue;
            }
        }
        dims.remove(idx);
    }
    if let Some(mut q_dims) = q_arg_dims {
        q_dims.extend(dims);
        dims = q_dims;
    }
    Some(Shape {
        dtype: base_shape.dtype.clone(),
        dims,
    })
    .or_else(|| {
        if context.diagnostics.len() == diag_before {
            context.push_diagnostic_text(
                whole_range,
                DiagnosticSeverity::ERROR,
                "Invalid dim".into(),
            );
        }
        None
    })
}

pub fn infer_unsqueeze(
    base_expr: &Expr,
    dim_arg: Option<&Expr>,
    record_hovers: bool,
    mut context: ContextRef,
) -> Option<Shape> {
    let base_shape = base_shape_or_diag(base_expr, record_hovers, context.reborrow())?;
    let dim = dim_arg.and_then(|expr| {
        let (vars, diagnostics, source) = context.vars_diagnostics_source();
        expr_to_dim_token(expr, vars, diagnostics, source, &mut false)
    })?;
    let dim_i: i16 = dim.parse().ok()?;
    let idx = normalize_dim_index_unsqueeze(dim_i, base_shape.dims.len())?;
    let mut dims = base_shape.dims.clone();
    dims.insert(idx, "1".to_string());
    Some(Shape {
        dtype: base_shape.dtype.clone(),
        dims,
    })
}

fn infer_shallow_shape(expr: &Expr, record_hovers: bool, mut context: ContextRef) -> Option<Shape> {
    match expr {
        Expr::BinOp(ExprBinOp {
            left,
            op,
            right,
            range,
        }) => {
            if matches!(op, Operator::MatMult) {
                return infer_matmul_shapes(left, right, record_hovers, *range, context);
            }
            None
        }
        _ => context.lookup_shape(expr, record_hovers),
    }
}

fn normalize_dim_index_squeeze(idx: i16, len: usize) -> Option<usize> {
    let size = len;
    let adj = if idx >= 0 { idx } else { size as i16 + idx };
    if adj >= 0 && (adj as usize) < size {
        Some(adj as usize)
    } else {
        None
    }
}

fn parse_dims(
    dim_expr: &Expr,
    len: usize,
    vars: &HashMap<Identifier, VarState>,
    diagnostics: &mut Vec<Diagnostic>,
    source: &str,
) -> Result<Vec<usize>, ()> {
    let mut has_err = false;
    let mut to_i16 = |e: &Expr| {
        expr_to_dim_token(e, vars, diagnostics, source, &mut has_err)
            .and_then(|s| s.parse::<i16>().ok())
    };
    let dims_i: Vec<i16> = match dim_expr {
        Expr::Tuple(t) => t.elts.iter().filter_map(to_i16).collect(),
        other => to_i16(other).into_iter().collect(),
    };

    if dims_i.is_empty() {
        diagnostics.push(Diagnostic {
            range: text_range_to_lsp(expr_text_range(dim_expr), source),
            severity: Some(DiagnosticSeverity::ERROR),
            code: None,
            code_description: None,
            source: Some("shapels".into()),
            message: "Invalid dim".into(),
            related_information: None,
            tags: None,
            data: None,
        });
        return Err(());
    }
    let mut out = Vec::new();
    for d in dims_i {
        if let Some(idx) = normalize_dim_index_squeeze(d, len) {
            out.push(idx);
        } else {
            has_err = true;
            // defer diagnostic emission until after loop to ensure only one per call
        }
    }
    if has_err {
        diagnostics.push(Diagnostic {
            range: text_range_to_lsp(expr_text_range(dim_expr), source),
            severity: Some(DiagnosticSeverity::ERROR),
            code: None,
            code_description: None,
            source: Some("shapels".into()),
            message: "Invalid dim".into(),
            related_information: None,
            tags: None,
            data: None,
        });
        return Err(());
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

fn normalize_dim_index_unsqueeze(idx: i16, len: usize) -> Option<usize> {
    let size = len + 1;
    let adj = if idx >= 0 { idx } else { size as i16 + idx };
    if adj >= 0 && (adj as usize) < size {
        Some(adj as usize)
    } else {
        None
    }
}
