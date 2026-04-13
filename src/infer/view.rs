//! Inference for view/reshape and permute/transpose.
use crate::context::ContextRef;
use crate::{Shape, expr_text_range};
use lsp_types::{Diagnostic, DiagnosticSeverity};
use rustpython_parser::ast::Expr;
use rustpython_parser::text_size::TextRange;
use std::borrow::Cow;

use super::expr_to_int;
use super::{expr_to_dim_token, text_range_to_lsp};
use crate::op_groups::TorchOp;

/// Inference for `torch.view`, `torch.reshape` and `torch.expand`.
pub fn infer_view_like(
    base_expr: &Expr,
    args: &[&Expr],
    torch_op: &TorchOp,
    base_hint: Option<Shape>,
    record_hovers: bool,
    whole_range: TextRange,
    mut context: ContextRef,
) -> Option<Shape> {
    let base_shape = base_hint.or_else(|| context.lookup_shape(base_expr, record_hovers));
    let target_tokens = args
        .iter()
        .map(|e| {
            expr_to_dim_token(
                e,
                context.vars,
                context.diagnostics,
                context.source,
                &mut false,
            )
        })
        .collect::<Option<Vec<_>>>()?;

    if target_tokens.is_empty() {
        return None;
    }
    let res = match torch_op {
        TorchOp::View => reshape_dims(base_shape, target_tokens.as_slice()),
        TorchOp::Expand => expand_dims(base_shape, target_tokens.as_slice()),
        _ => unreachable!(),
    };
    match res {
        Ok(shape) => Some(shape),
        Err(msg) => {
            context.diagnostics.push(Diagnostic {
                range: text_range_to_lsp(whole_range, context.source),
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

/// Variants for an operation that transposes dimensions.
pub enum Transpose {
    Explicit,
    T,
    Permute,
}

/// Permute dimensions of a tensor based on provided order.
pub fn infer_permute(
    base_expr: &Expr,
    order_args: &[&Expr],
    transpose: Transpose,
    base_hint: Option<Shape>,
    record_hovers: bool,
    whole_range: TextRange,
    mut context: ContextRef,
) -> Option<Shape> {
    let base_shape = base_hint.or_else(|| context.lookup_shape(base_expr, record_hovers))?;
    let dims_len = base_shape.dims.len();
    let mut order = Vec::new();
    match transpose {
        Transpose::Explicit => {
            if order_args.len() != 2 {
                context.push_diagnostic_text(
                    whole_range,
                    DiagnosticSeverity::ERROR,
                    "transpose expects exactly two dimensions".into(),
                );
                return None;
            }
            let mut dims = Vec::with_capacity(2);
            for expr in order_args {
                if let Some(val) = expr_to_int(expr, Some(dims_len)) {
                    dims.push(val as usize);
                    continue;
                }
                context.push_diagnostic_text(
                    expr_text_range(expr),
                    DiagnosticSeverity::ERROR,
                    "Invalid transpose index".into(),
                );
                return None;
            }
            if dims.iter().any(|&d| d >= base_shape.dims.len()) || dims[0] == dims[1] {
                context.push_diagnostic_text(
                    whole_range,
                    DiagnosticSeverity::ERROR,
                    "Invalid transpose dimensions".into(),
                );
                return None;
            }
            order = (0..base_shape.dims.len()).collect();
            order.swap(dims[0], dims[1]);
        }
        Transpose::T => {
            order = (0..base_shape.dims.len()).collect();
            let order_len = order.len();
            if order_len > 1 {
                order.swap(order_len - 2, order_len - 1);
            }
        }
        Transpose::Permute => {
            order.reserve(order_args.len());
            for expr in order_args {
                if let Some(val) = expr_to_int(expr, Some(dims_len)) {
                    order.push(val as usize);
                    continue;
                }
                context.push_diagnostic_text(
                    expr_text_range(expr),
                    DiagnosticSeverity::ERROR,
                    "Invalid permute index".into(),
                );
                return None;
            }
        }
    }

    if order.len() != base_shape.dims.len()
        || order.iter().any(|&i| i >= base_shape.dims.len())
        || {
            let mut uniq = order.clone();
            uniq.sort_unstable();
            uniq.dedup();
            uniq.len() != order.len()
        }
    {
        context.push_diagnostic_text(
            whole_range,
            DiagnosticSeverity::ERROR,
            if !matches!(transpose, Transpose::Permute) {
                "Invalid transpose dimensions".into()
            } else {
                "Invalid permute dimensions".into()
            },
        );
        return None;
    }

    let dims = order
        .iter()
        .map(|&idx| base_shape.dims[idx].clone())
        .collect::<Vec<_>>();

    Some(Shape {
        dtype: base_shape.dtype.clone(),
        dims,
    })
}

fn reshape_dims(base: Option<Shape>, target: &[Cow<str>]) -> Result<Shape, String> {
    let mut tokens = target.to_vec();
    let mut minus_one_idx = None;
    for (i, t) in tokens.iter().enumerate() {
        if t == "-1" {
            if minus_one_idx.is_some() {
                return Err("Only one -1 is allowed in view/reshape".into());
            }
            minus_one_idx = Some(i);
        }
    }

    if let Some(base_shape) = base {
        let mut remaining = flatten_dims(&base_shape.dims);
        if let Some(idx) = minus_one_idx {
            let mut target_factors = Vec::new();
            for t in &tokens {
                if t == "-1" {
                    continue;
                }
                let facs = split_dim(t);
                target_factors.extend(facs);
            }
            for f in &target_factors {
                if let Some(pos) = remaining.iter().position(|x| x == f) {
                    remaining.remove(pos);
                }
            }
            let inferred = if remaining.is_empty() {
                "1".to_string()
            } else {
                remaining.join("*")
            };
            tokens[idx] = Cow::Owned(inferred);
        }
        return Ok(Shape {
            dtype: base_shape.dtype.clone(),
            dims: tokens.into_iter().map(|c| c.into_owned()).collect(),
        });
    }

    // No base shape: still return with -1 replaced by "Infer"
    if let Some(idx) = minus_one_idx {
        tokens[idx] = Cow::Borrowed("Infer");
    }
    Ok(Shape {
        dtype: None,
        dims: tokens.into_iter().map(|c| c.into_owned()).collect(),
    })
}

fn expand_dims(base: Option<Shape>, target: &[Cow<str>]) -> Result<Shape, String> {
    if let Some(Shape { dtype, dims }) = base {
        let d = dims.len();
        let t = target.len();
        if dims.len() != target.len() {
            return Err(format!("Incorrect number of dimensions: {d} vs. {t}"));
        }
        let new_dims = dims
            .iter()
            .zip(target.iter())
            .map(
                |(left, right)| match (left.parse::<i32>(), right.parse::<i32>()) {
                    (Ok(_), Ok(_)) | (Err(_), Err(_)) if left.as_str() == right => {
                        Some(Cow::Borrowed(left.as_str()))
                    }
                    (_, Ok(-1)) => Some(Cow::Borrowed(left.as_str())),
                    (Ok(1), _) => Some(right.clone()),
                    _ => None,
                },
            )
            .collect::<Vec<_>>();
        if let Some(invalid_idx) = new_dims.iter().position(|x| x.is_none()) {
            return Err(format!(
                "Invalid dimension at non-singleton position {invalid_idx}"
            ));
        }
        Ok(Shape {
            dims: new_dims
                .into_iter()
                // we can unwrap here since we have checked for any None just before
                .map(|c| c.unwrap().into_owned())
                .collect(),
            dtype,
        })
    } else {
        Err("Could not infer shape of base tensor".to_string())
    }
}

fn flatten_dims(dims: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for d in dims {
        out.extend(split_dim(d));
    }
    out
}

fn split_dim(dim: &str) -> Vec<String> {
    dim.split('*').map(|s| s.to_string()).collect()
}
