//! Inference for repeat, etc.
use crate::{Shape, VarState, expr_text_range, get_arg};
use lsp_types::{Diagnostic, DiagnosticSeverity};
use rustpython_parser::ast::{self, Expr, ExprCall, Identifier};
use std::borrow::Cow;
use std::collections::HashMap;

use super::{expr_to_dim_token, product_token, text_range_to_lsp};

pub fn infer_repeat(
    base: Shape,
    call: &ExprCall,
    vars: &HashMap<Identifier, VarState>,
    diagnostics: &mut Vec<Diagnostic>,
    source: &str,
) -> Option<Shape> {
    let list = match call.args.as_slice() {
        [Expr::List(list), ..] => list.elts.as_slice(),
        [Expr::Tuple(seq), ..] => seq.elts.as_slice(),
        rest => rest,
    };
    let mut diag_already = false;

    let factors: Vec<Option<Cow<_>>> = list
        .iter()
        .map(|expr| expr_to_dim_token(expr, vars, diagnostics, source, &mut diag_already))
        .collect();
    if factors.is_empty() || factors.iter().any(|x| x.is_none()) {
        // this diagnostic is left for a generalist language server
        return None;
    }
    // we can unwrap since we have checked for any None just before
    let factors: Vec<_> = factors.into_iter().map(|dim| dim.unwrap()).collect();

    // check for negative dimensions
    if factors
        .iter()
        .any(|f| f.parse::<i64>().map(|x| x < 0).unwrap_or(false))
    {
        diagnostics.push(Diagnostic {
            range: text_range_to_lsp(call.range, source),
            severity: Some(DiagnosticSeverity::ERROR),
            code: None,
            code_description: None,
            source: Some("shapels".into()),
            message: "Trying to create tensor with negative dimensions".into(),
            related_information: None,
            tags: None,
            data: None,
        });
        return None;
    }

    let base_len = base.dims.len();
    let factors_len = factors.len();
    if factors_len < base_len {
        diagnostics.push(Diagnostic {
            range: text_range_to_lsp(call.range, source),
            severity: Some(DiagnosticSeverity::ERROR),
            code: None,
            code_description: None,
            source: Some("shapels".into()),
            message: format!("Number of dimensions of repeat dims ({factors_len}) can not be smaller than number of dimensions of tensor ({base_len})"),
            related_information: None,
            tags: None,
            data: None,
        });
        return None;
    }
    let mut out_dims = Vec::with_capacity(factors_len);
    let leading = factors_len.saturating_sub(base_len);
    for (idx, f) in factors.into_iter().enumerate() {
        if idx < leading {
            out_dims.push(f.into_owned());
        } else {
            let base_dim = &base.dims[idx - leading];
            let prod = match (f.parse::<i64>(), base_dim.parse::<i64>()) {
                (Ok(0), _) => "0".to_string(),
                (Ok(a), Ok(b)) => (a * b).to_string(),
                _ if f == "1" => base_dim.clone(),
                _ if base_dim == "1" => f.into_owned(),
                _ => format!("{f}*{base_dim}"),
            };
            out_dims.push(prod);
        }
    }
    Some(Shape {
        dtype: base.dtype,
        dims: out_dims,
    })
}

fn multiply_tokens(a: &str, b: &str) -> String {
    if let (Ok(ia), Ok(ib)) = (a.parse::<i64>(), b.parse::<i64>()) {
        (ia * ib).to_string()
    } else if a == "1" {
        b.to_string()
    } else if b == "1" {
        a.to_string()
    } else {
        format!("{a}*{b}")
    }
}

// TODO(carrascomj): piggyback on expr_to_dim_token for the non-tensor cases
fn repeats_count(expr: &Expr) -> Option<(String, bool, bool)> {
    match expr {
        Expr::Constant(c) => match &c.value {
            ast::Constant::Int(i) => {
                let val = i.to_string().parse::<i64>().unwrap_or(0);
                let s = val.to_string();
                Some((s.clone(), val < 0, true))
            }
            _ => None,
        },
        Expr::Name(n) => Some((n.id.to_string(), false, true)),
        Expr::UnaryOp(u) if matches!(u.op, ast::UnaryOp::USub) => {
            if let Expr::Constant(c) = u.operand.as_ref()
                && let ast::Constant::Int(i) = &c.value
            {
                let val = i.to_string().parse::<i64>().unwrap_or(0);
                let s = format!("-{}", val.abs());
                return Some((s, val < 0, true));
            }
            None
        }
        // torch.tensor call, only implemented if init for 1-d tensor
        Expr::Call(call) => {
            if let Expr::Attribute(attr) = call.func.as_ref()
                && attr.attr.as_str() == "tensor"
                && let Some(arg0) = call.args.first()
            {
                if let Expr::List(list) = arg0 {
                    let mut sum: i64 = 0;
                    for elt in &list.elts {
                        if let Expr::Constant(c) = elt
                            && let ast::Constant::Int(i) = &c.value
                        {
                            sum += i.to_string().parse::<i64>().unwrap_or(0);
                        } else {
                            return None;
                        }
                    }
                    return Some((sum.to_string(), sum < 0, false));
                }
                if let Expr::Tuple(tup) = arg0 {
                    let mut sum: i64 = 0;
                    for elt in &tup.elts {
                        if let Expr::Constant(c) = elt
                            && let ast::Constant::Int(i) = &c.value
                        {
                            sum += i.to_string().parse::<i64>().unwrap_or(0);
                        } else {
                            return None;
                        }
                    }
                    return Some((sum.to_string(), sum < 0, false));
                }
            }
            None
        }
        _ => None,
    }
}

fn dim_from_expr(expr: &Expr) -> Option<i16> {
    match expr {
        Expr::Constant(c) => match &c.value {
            ast::Constant::Int(i) => i.to_string().parse::<i16>().ok(),
            _ => None,
        },
        Expr::UnaryOp(u) if matches!(u.op, ast::UnaryOp::USub) => {
            if let Expr::Constant(c) = u.operand.as_ref()
                && let ast::Constant::Int(i) = &c.value
            {
                i.to_string().parse::<i16>().ok().map(|v| -v)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Infer for `torch.repeat_interleave` given a base [`Shape`].
pub fn infer_repeat_interleave(
    base: Shape,
    call: &ExprCall,
    offset: usize,
    diagnostics: &mut Vec<Diagnostic>,
    source: &str,
) -> Option<Shape> {
    let repeats_expr = get_arg(call, "repeats", offset);
    let (repeats, is_neg, is_scalar) =
        repeats_expr
            .and_then(repeats_count)
            .unwrap_or((String::new(), false, true));
    if repeats.is_empty() {
        diagnostics.push(Diagnostic {
            range: text_range_to_lsp(
                repeats_expr.map(expr_text_range).unwrap_or(call.range),
                source,
            ),
            severity: Some(DiagnosticSeverity::INFORMATION),
            code: None,
            code_description: None,
            source: Some("shapels".into()),
            message: "Dims could not be inferred statically from `repeats`.".into(),
            related_information: None,
            tags: None,
            data: None,
        });
        return None;
    }
    if is_neg {
        diagnostics.push(Diagnostic {
            range: text_range_to_lsp(call.range, source),
            severity: Some(DiagnosticSeverity::ERROR),
            code: None,
            code_description: None,
            source: Some("shapels".into()),
            message: "repeat_interleave repeats must be non-negative".into(),
            related_information: None,
            tags: None,
            data: None,
        });
        return None;
    }
    let dim_expr = get_arg(call, "dim", offset + 1).or_else(|| call.args.get(offset + 1));
    if dim_expr.is_none() {
        let flat = product_token(&base.dims);
        let new_dim = if is_scalar {
            multiply_tokens(&flat, &repeats)
        } else {
            repeats.clone()
        };
        return Some(Shape {
            dtype: base.dtype.clone(),
            dims: vec![new_dim],
        });
    }
    let len = base.dims.len();
    let dim_idx = dim_expr.and_then(dim_from_expr).and_then(|d| {
        let size = len as i16;
        let adj = if d >= 0 { d } else { size + d };
        if adj >= 0 && (adj as usize) < len {
            Some(adj as usize)
        } else {
            None
        }
    });
    let idx = match dim_idx {
        Some(i) => i,
        None => {
            diagnostics.push(Diagnostic {
                range: text_range_to_lsp(call.range, source),
                severity: Some(DiagnosticSeverity::ERROR),
                code: None,
                code_description: None,
                source: Some("shapels".into()),
                message: "Invalid dim for repeat_interleave".into(),
                related_information: None,
                tags: None,
                data: None,
            });
            return None;
        }
    };
    let mut dims = base.dims.clone();
    dims[idx] = if is_scalar {
        multiply_tokens(&dims[idx], &repeats)
    } else {
        repeats
    };
    Some(Shape {
        dtype: base.dtype,
        dims,
    })
}
