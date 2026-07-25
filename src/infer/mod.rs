//! Specialized inference of `Shape`s for the various implemented operations.
//!
//! Word of caution: the `dim` argument in torch methods and functions expects
//! a int64_t in the ATen implementation
//! (e.g., [here](https://github.com/pytorch/pytorch/blob/9f7fceb887d0cfa0326a59b887821c63ff11340a/torch/csrc/lazy/core/ops/utils.cpp#L92)).
//! However, in functions like (un)squeeze or reduce ops, shapels parses dim as i16 because
//! i64 is excessive for operations that relate to the number of dimensions and
//! not the dimenions themselves. This should be revisited if bugs come.
#![allow(clippy::too_many_arguments)]
use crate::context::ContextRef;
use crate::expr_tokens::{DIM_TOKEN_OPTIONS, expr_to_symbolic_token};
use crate::op_groups::{RangeOps, SimpleDtype};
use crate::{Imports, Shape, VarState, expr_text_range, get_arg, get_dtype};
use lsp_types::{Diagnostic, DiagnosticSeverity};
use rustpython_parser::ast::{
    self, Constant, Expr, ExprAttribute, ExprCall, ExprConstant, ExprName, ExprSubscript,
    ExprUnaryOp, Identifier,
};
use rustpython_parser::text_size::TextRange;
use std::borrow::Cow;
use std::collections::HashMap;

mod broadcastable;
pub use broadcastable::{ShapeOrExpr, infer_broadcastable_poswise};
mod conv;
pub use conv::{infer_conv, infer_conv_module, infer_conv_transpose};
mod loss;
pub use loss::{LossExpectedInputs, LossParams, Reduction, infer_loss};
mod index;
pub use index::infer_index;
mod pool;
pub use pool::{PoolKind, infer_pool_module};
mod repeat;
pub use repeat::{infer_repeat, infer_repeat_interleave};
mod squeeze;
pub use squeeze::{infer_squeeze, infer_unsqueeze};
mod view;
pub use view::{Transpose, infer_permute, infer_view_like};

use crate::text_range_to_lsp;

/// A destructuring assignment whose values are dimensions read from the same
/// tensor's `.shape`, such as `B, C = x.shape[0], x.shape[1]`.
///
/// Keeping this small bit of syntax recognition with inference utilities lets
/// assignment handling reuse the normal shape-state update path.
pub(crate) fn indexed_shape_assignment(
    target: &Expr,
    value: &Expr,
) -> Option<(Identifier, Vec<(usize, Identifier)>)> {
    let Expr::Tuple(targets) = target else {
        return None;
    };
    let Expr::Tuple(values) = value else {
        return None;
    };
    if targets.elts.is_empty() || targets.elts.len() != values.elts.len() {
        return None;
    }

    let mut base_id = None;
    let mut assignments = Vec::with_capacity(targets.elts.len());
    for (target, value) in targets.elts.iter().zip(&values.elts) {
        let Expr::Name(target_name) = target else {
            return None;
        };
        let (base, index) = shape_index(value)?;
        if base_id.as_ref().is_some_and(|known| known != &base) {
            return None;
        }
        base_id = Some(base);
        assignments.push((index, target_name.id.clone()));
    }
    Some((base_id?, assignments))
}

fn shape_index(expr: &Expr) -> Option<(Identifier, usize)> {
    let Expr::Subscript(ExprSubscript { value, slice, .. }) = expr else {
        return None;
    };
    let Expr::Attribute(ExprAttribute { value, attr, .. }) = value.as_ref() else {
        return None;
    };
    let Expr::Name(base) = value.as_ref() else {
        return None;
    };
    let Expr::Constant(ExprConstant {
        value: Constant::Int(index),
        ..
    }) = slice.as_ref()
    else {
        return None;
    };
    (attr.as_str() == "shape")
        .then(|| usize::try_from(index).ok())
        .flatten()
        .map(|index| (base.id.clone(), index))
}

/// Specific to index/permute/transpose/etc. that need negative indexing normalization.
fn expr_to_int(expr: &Expr, dims_len: Option<usize>) -> Option<i64> {
    match expr {
        Expr::Constant(ExprConstant {
            value: ast::Constant::Int(i),
            ..
        }) => i.to_string().parse::<i64>().ok(),
        Expr::UnaryOp(ExprUnaryOp { op, operand, .. }) => {
            expr_to_int(operand, dims_len).and_then(|val| match (op, dims_len) {
                (ast::UnaryOp::UAdd, _) => Some(val),
                (ast::UnaryOp::USub, Some(n)) if val <= n as i64 => Some(n as i64 - val),
                (ast::UnaryOp::USub, None) => Some(-val),
                _ => None,
            })
        }
        _ => None,
    }
}

fn concrete_dim_mismatch(actual: &str, expected: &str) -> bool {
    match (actual.parse::<i64>(), expected.parse::<i64>()) {
        (Ok(current), Ok(expected)) => current != expected,
        _ => false,
    }
}

fn expand_conv_params(values: &[String], conv_dim: usize, default: &str) -> Vec<String> {
    match values.len() {
        0 => vec![default.to_string(); conv_dim],
        1 => vec![values[0].clone(); conv_dim],
        len if len >= conv_dim => values[..conv_dim].to_vec(),
        _ => {
            let mut expanded = values.to_vec();
            while expanded.len() < conv_dim {
                expanded.push(values[0].clone());
            }
            expanded
        }
    }
}

fn broadcast_dims(a: &[String], b: &[String]) -> Result<Vec<String>, String> {
    if a.is_empty() && b.is_empty() {
        return Ok(Vec::new());
    } else if a.is_empty() {
        return Ok(b.into());
    } else if b.is_empty() {
        return Ok(a.into());
    }
    let mut out = Vec::new();
    let mut idx = 0usize;
    let max_len = a.len().max(b.len());
    while idx < max_len {
        let a_dim = a.get(a.len().wrapping_sub(1 + idx)).map(String::as_str);
        let b_dim = b.get(b.len().wrapping_sub(1 + idx)).map(String::as_str);
        let res = match (a_dim, b_dim) {
            (Some(ad), Some(bd)) if ad == bd => ad.to_string(),
            (Some("1"), Some(bd)) => bd.to_string(),
            (Some(ad), Some("1")) => ad.to_string(),
            (Some(ad), None) => ad.to_string(),
            (None, Some(bd)) => bd.to_string(),
            (Some(_), Some(_)) => {
                return Err("dimension mismatch".into());
            }
            (None, None) => unreachable!(),
        };
        out.push(res);
        idx += 1;
    }
    out.reverse();
    Ok(out)
}

fn product_token(tokens: &[String]) -> String {
    let (sym_dim, conc_dim) =
        tokens.iter().fold((String::new(), 1), |(sym, conc), dim| {
            match dim.parse::<usize>() {
                Ok(d) => (sym, conc * d),
                Err(_) if sym.is_empty() => (dim.to_string(), conc),
                _ => (sym + "*" + dim, conc),
            }
        });
    if sym_dim.is_empty() {
        conc_dim.to_string()
    } else if conc_dim > 1 {
        sym_dim + &format!("*{conc_dim}")
    } else {
        sym_dim
    }
}

/// Inference and matrix multiplication shape inference shared by `@` and `torch.mm`.
/// Keeps all leading dims of left except the last, then appends all trailing dims of right except the first.
pub fn infer_matmul_shapes(
    left: &Expr,
    right: &Expr,
    record_hovers: bool,
    whole_range: TextRange,
    mut context: ContextRef,
) -> Option<Shape> {
    let left_shape = context.lookup_or_infer(left, record_hovers);
    let right_shape = context.lookup_or_infer(right, record_hovers);
    match (left_shape, right_shape) {
        (Some(l), Some(r)) => match matmul(&l, &r) {
            Ok(shape) => Some(shape),
            Err(msg) => {
                context.push_diagnostic_text(whole_range, DiagnosticSeverity::ERROR, msg);
                None
            }
        },
        _ => None,
    }
}

/// Matrix multiplication shape inference shared by `@` and `torch.mm`.
/// Mirrors `torch.matmul` semantics, including batch-dimension broadcasting.
fn matmul(left: &Shape, right: &Shape) -> Result<Shape, String> {
    if left.dims.is_empty() || right.dims.is_empty() {
        return Err("Matmul requires both operands to have shapes".into());
    }

    let left_promoted = left.dims.len() == 1;
    let right_promoted = right.dims.len() == 1;

    let left_dims = if left_promoted {
        vec!["1".to_string(), left.dims[0].clone()]
    } else {
        left.dims.clone()
    };
    let right_dims = if right_promoted {
        vec![right.dims[0].clone(), "1".to_string()]
    } else {
        right.dims.clone()
    };

    let left_inner = left_dims.last().unwrap();
    let right_inner = &right_dims[right_dims.len() - 2];
    if left_inner != right_inner {
        return Err(format!(
            "Matmul inner dimensions mismatch: {} vs {}",
            left_inner, right_inner
        ));
    }

    let batch_dims = broadcast_dims(
        &left_dims[..left_dims.len() - 2],
        &right_dims[..right_dims.len() - 2],
    )
    .map_err(|msg| format!("Matmul batch dimensions mismatch: {msg}"))?;

    let mut dims = batch_dims;
    dims.push(left_dims[left_dims.len() - 2].clone());
    dims.push(right_dims.last().unwrap().clone());

    if left_promoted {
        let matrix_start = dims.len().saturating_sub(2);
        dims.remove(matrix_start);
    }
    if right_promoted {
        dims.pop();
    }

    Ok(Shape {
        dtype: left.dtype.clone().or(right.dtype.clone()),
        dims,
    })
}

/// Infer softmax-like: no-op shapewise, but need to report diagnositcs
pub fn infer_noop(
    base_expr: Option<Shape>,
    dim_arg: Option<&Expr>,
    vars: &HashMap<Identifier, VarState>,
    diagnostics: &mut Vec<Diagnostic>,
    source: &str,
    whole_range: TextRange,
    enforce_dim: bool,
) -> Option<Shape> {
    let mut diag_already = false;
    base_expr.filter(|shape| {
        let Some(Ok(dim_i)) = dim_arg
            .and_then(|expr| expr_to_dim_token(expr, vars, diagnostics, source, &mut diag_already))
            .map(|x| x.parse::<i16>())
        else {
            // for softmax at least, dim is always necessary since
            // some torch version, but it's not explicitly indicated
            // in the python API so a general LSP won't catch this
            if enforce_dim {
                diagnostics.push(Diagnostic {
                    range: text_range_to_lsp(whole_range, source),
                    severity: Some(DiagnosticSeverity::ERROR),
                    code: None,
                    code_description: None,
                    source: Some("shapels".into()),
                    message: "dim argument is required!".into(),
                    related_information: None,
                    tags: None,
                    data: None,
                });
            }
            return false;
        };
        let ndim = shape.dims.len() as i16;
        if (dim_i > 0) && (dim_i > (ndim - 1))  // positive dims must be a valid index
            || (dim_i < 0 && dim_i.abs() > ndim && ndim > 0)  // negative dims can be at most -ndim
            // dim 0 or -1 is always valid even for empty tensors
            || (ndim == 0 && !(-1..=0).contains(&dim_i))
        {
            diagnostics.push(Diagnostic {
                range: text_range_to_lsp(whole_range, source),
                severity: Some(DiagnosticSeverity::ERROR),
                code: None,
                code_description: None,
                source: Some("shapels".into()),
                message: format!(
                    "dim={dim_i} is out of range for tensor dims {}",
                    shape.render()
                ),
                related_information: None,
                tags: None,
                data: None,
            });
            false
        } else {
            true
        }
    })
}

/// Arguments by position establish a shape by a multiple arguments or by a sequence on the first argument.
///
/// An optional dtype might be used for the dtype. Otherwise, use Float.
pub fn infer_creation_size(
    call: &ExprCall<TextRange>,
    vars: &HashMap<Identifier, VarState>,
    diagnostics: &mut Vec<Diagnostic>,
    source: &str,
    shape_hint: Option<Shape>,
    dtype: Option<String>,
    not_tensor: bool,
) -> Option<Shape> {
    if let Some(shape) = shape_hint {
        return Some(Shape {
            dims: shape.dims,
            dtype,
        });
    }
    // lists and tuples are parsed their lengths (assumend non-nested)
    if !not_tensor {
        if let [Expr::List(list), ..] = call.args.as_slice() {
            return Some(Shape {
                dtype,
                dims: vec![list.elts.len().to_string()],
            });
        }
        if let [Expr::Tuple(seq), ..] = call.args.as_slice() {
            return Some(Shape {
                dtype,
                dims: vec![seq.elts.len().to_string()],
            });
        }
    }
    let list = match call.args.as_slice() {
        [Expr::List(list), ..] if not_tensor => list.elts.as_slice(),
        [Expr::Tuple(seq), ..] if not_tensor => seq.elts.as_slice(),
        rest => rest,
    };
    let mut diag_already = false;

    let vec_dims: Vec<Option<Cow<_>>> = list
        .iter()
        .map(|expr| expr_to_dim_token(expr, vars, diagnostics, source, &mut diag_already))
        .collect();

    let maybe_dims: Option<Vec<String>> = vec_dims
        .into_iter()
        .map(|o| o.map(|c| c.into_owned()))
        .collect();
    maybe_dims.map_or_else(
        || {
            if !diag_already {
                diagnostics.push(Diagnostic {
                    range: text_range_to_lsp(call.range, source),
                    severity: Some(DiagnosticSeverity::INFORMATION),
                    code: None,
                    code_description: None,
                    source: Some("shapels".into()),
                    message: "Shape could not be initialized from expression".to_string(),
                    related_information: None,
                    tags: None,
                    data: None,
                });
            }
            None
        },
        |dims| {
            Some(Shape {
                dtype,
                dims: dims.into_iter().map(|x| x.to_string()).collect(),
            })
        },
    )
}

/// An expr that is to be resolved as a dimension like variadic args of
/// `torch.zeros(8, x.shape[1], Batch, 10*Feat)`.
fn expr_to_dim_token<'a>(
    x: &'a Expr,
    vars: &'a HashMap<Identifier, VarState>,
    diagnostics: &mut Vec<Diagnostic>,
    source: &'a str,
    diag_already: &mut bool,
) -> Option<Cow<'a, str>> {
    if let Some(token) = expr_to_symbolic_token(x, DIM_TOKEN_OPTIONS) {
        return Some(token);
    }

    match x {
        Expr::Constant(constant) => {
            if !*diag_already {
                diagnostics.push(Diagnostic {
                    range: text_range_to_lsp(constant.range, source),
                    severity: Some(DiagnosticSeverity::INFORMATION),
                    code: None,
                    code_description: None,
                    source: Some("shapels".into()),
                    message: "Dim was not understood from this argument".into(),
                    related_information: None,
                    tags: None,
                    data: None,
                });
                *diag_already = true;
            }
            None
        }
        Expr::BinOp(_) | Expr::UnaryOp(_) => None,
        // e.g., torch.zeros(x.shape[int])
        Expr::Subscript(ExprSubscript { value, slice, .. }) => {
            if let (Expr::Attribute(attr), Expr::Constant(c)) = (value.as_ref(), slice.as_ref())
                && let Expr::Name(name) = attr.value.as_ref()
                && let Constant::Int(i) = &c.value
                && let Ok(idx) = usize::try_from(i)
                && attr.attr.as_str() == "shape"
            {
                size_to_dim(vars, name, idx, x, source, diagnostics, true)
            } else {
                diagnostics.push(Diagnostic {
                    range: text_range_to_lsp(expr_text_range(x), source),
                    severity: Some(DiagnosticSeverity::INFORMATION),
                    code: None,
                    code_description: None,
                    source: Some("shapels".into()),
                    message: "Dim was not understood from this argument".into(),
                    related_information: None,
                    tags: None,
                    data: None,
                });
                *diag_already = true;
                None
            }
        }
        // e.g., torch.zeros(x.size(int))
        Expr::Call(ExprCall {
            func,
            args,
            keywords,
            ..
        }) => {
            if let (true, [arg0], Expr::Attribute(attr)) =
                (keywords.is_empty(), args.as_slice(), func.as_ref())
                && attr.attr.as_str() == "size"
                && let Expr::Name(name) = attr.value.as_ref()
                && let Expr::Constant(c) = arg0
                && let Constant::Int(i) = &c.value
                && let Ok(idx) = usize::try_from(i)
            {
                size_to_dim(vars, name, idx, x, source, diagnostics, true)
            } else {
                diagnostics.push(Diagnostic {
                    range: text_range_to_lsp(expr_text_range(x), source),
                    severity: Some(DiagnosticSeverity::INFORMATION),
                    code: None,
                    code_description: None,
                    source: Some("shapels".into()),
                    message: "Dim was not understood from this argument".into(),
                    related_information: None,
                    tags: None,
                    data: None,
                });
                *diag_already = true;
                None
            }
        }
        _ => {
            if !*diag_already {
                diagnostics.push(Diagnostic {
                    range: text_range_to_lsp(expr_text_range(x), source),
                    severity: Some(DiagnosticSeverity::INFORMATION),
                    code: None,
                    code_description: None,
                    source: Some("shapels".into()),
                    message: "Dim was not understood from this expression".into(),
                    related_information: None,
                    tags: None,
                    data: None,
                });
                *diag_already = true;
            }
            None
        }
    }
}

fn size_to_dim<'a>(
    vars: &'a HashMap<Identifier, VarState>,
    name: &ExprName,
    idx: usize,
    x: &Expr,
    source: &str,
    diagnostics: &mut Vec<Diagnostic>,
    // could be a .shape[] attr index or a .size() method call
    is_size: bool,
) -> Option<Cow<'a, str>> {
    let shape = vars
        .get(&name.id)
        .and_then(|v| v.annotated.as_ref().or(v.inferred.as_ref()));

    let out = shape.and_then(|sh| sh.dims.get(idx).map(|dim| Cow::Borrowed(dim.as_str())));

    if out.is_none() {
        let (severity, message) = if shape.is_none() {
            (
                DiagnosticSeverity::WARNING,
                "This tensor shape is unknown at this point".to_string(),
            )
        } else {
            (
                DiagnosticSeverity::ERROR,
                format!(
                    "No dim found at .{} `{idx}`",
                    if is_size { "size" } else { "shape" }
                ),
            )
        };

        diagnostics.push(Diagnostic {
            range: text_range_to_lsp(expr_text_range(x), source),
            severity: Some(severity),
            code: None,
            code_description: None,
            source: Some("shapels".into()),
            message,
            related_information: None,
            tags: None,
            data: None,
        });
    }

    out
}

/// `torch.Tensor.to` changes the dtype.
///
/// It returns `Some` if `base_expr` is Some(Shape), since the argument
/// is not required.
pub fn infer_to(
    base_expr: Option<Shape>,
    dtype_arg: Option<&Expr>,
    diagnostics: &mut Vec<Diagnostic>,
    source: &str,
    imports: &Imports,
) -> Option<Shape> {
    if let Some(dtype_expr) = dtype_arg {
        match get_dtype(dtype_expr, imports) {
            Some(dtype) => base_expr.map(|x| Shape {
                dtype: Some(dtype.to_string()),
                dims: x.dims,
            }),
            None => {
                // dtype provided but not understood
                diagnostics.push(Diagnostic {
                    range: text_range_to_lsp(expr_text_range(dtype_expr), source),
                    // NOTE: this might be turned into a WARNING or ERROR if deemed
                    // reliable enough
                    severity: Some(DiagnosticSeverity::INFORMATION),
                    code: None,
                    code_description: None,
                    source: Some("shapels".into()),
                    message: "dtype not understood".to_string(),
                    related_information: None,
                    tags: None,
                    data: None,
                });
                base_expr
            }
        }
    } else {
        // no dtype arg provided, return base_expr as is
        base_expr
    }
}

/// Infer shape from creation operations [`RangeOps`].
pub fn infer_range_size(
    call: &ExprCall<TextRange>,
    range_op: RangeOps,
    record_hovers: bool,
    mut context: ContextRef,
) -> Option<Shape> {
    // helper function
    let push_diag = |context: &mut ContextRef<'_>, range, msg: &str| {
        context.push_diagnostic_text(range, DiagnosticSeverity::INFORMATION, msg.into())
    };
    let parse_dim = |expr: &Expr, context: &mut ContextRef<'_>| {
        let (vars, diagnostics, source) = context.vars_diagnostics_source();
        expr_to_dim_token(expr, vars, diagnostics, source, &mut false).map(Cow::into_owned)
    };

    let dim_expr = match range_op {
        RangeOps::Randperm => get_arg(call, "n", 0).map(Cow::Borrowed).or_else(|| {
            push_diag(&mut context, call.range, "Failed shape init: `n` arg");
            None
        }),
        RangeOps::Linspace | RangeOps::Logspace => {
            get_arg(call, "steps", 2).map(Cow::Borrowed).or_else(|| {
                push_diag(&mut context, call.range, "Failed shape init: `steps`");
                None
            })
        }
        RangeOps::Range | RangeOps::Arange => {
            let mut start_expr = get_arg(call, "start", 0);
            let mut end_expr = get_arg(call, "end", 1);
            let step_expr = get_arg(call, "step", 2);
            let has_kw_start = call
                .keywords
                .iter()
                .any(|kw| kw.arg.as_deref() == Some("start"));
            let has_kw_end = call
                .keywords
                .iter()
                .any(|kw| kw.arg.as_deref() == Some("end"));
            let implicit_end = range_op == RangeOps::Arange
                && call.args.len() == 1
                && !has_kw_start
                && !has_kw_end;
            if implicit_end {
                end_expr = start_expr;
                start_expr = None;
            }
            let start = start_expr
                .and_then(|expr| parse_dim(expr, &mut context))
                .unwrap_or_else(|| "0".to_string());
            let end = end_expr.and_then(|expr| parse_dim(expr, &mut context));
            let step = step_expr
                .and_then(|expr| parse_dim(expr, &mut context))
                .unwrap_or_else(|| "1".to_string());
            if let Some(end) = end {
                let is_range = range_op == RangeOps::Range;
                let ident = match (
                    start.parse::<f32>(),
                    end.parse::<f32>(),
                    step.parse::<f32>(),
                ) {
                    (Ok(s), Ok(e), Ok(ste)) => {
                        let raw = (e - s) / ste;
                        let count = if is_range {
                            raw.floor() + 1.0
                        } else {
                            raw.ceil()
                        };
                        (count as i64).to_string()
                    }
                    (Ok(s), Ok(e), Err(_)) => {
                        let plus_one = if is_range { "+1" } else { "" };
                        (e - s).to_string() + format!("/{step}{plus_one}").as_str()
                    }
                    _ => {
                        let plus_one = if is_range { "+1" } else { "" };
                        format!("{end}-{start}/{step}{plus_one}")
                    }
                };
                Some(Cow::Owned(Expr::Name(ast::ExprName {
                    range: call.range,
                    id: ast::Identifier::new(ident),
                    ctx: ast::ExprContext::Store,
                })))
            } else {
                None
            }
        }
    };
    let dim = dim_expr
        .as_deref()
        .and_then(|expr| parse_dim(expr, &mut context))
        .or_else(|| {
            push_diag(
                &mut context,
                call.range,
                "Argument was not understood as shape",
            );
            None
        });
    if let Some(dim) = dim {
        let dtype = get_arg(call, "dtype", range_op.dtype_arg_pos()).and_then(|expr| {
            // dtype as torch.Tensor.dtype
            let attr_dtype = if matches!(expr, Expr::Attribute(_)) {
                context
                    .infer_shape(expr, record_hovers)
                    .and_then(|shape| shape.dtype)
            } else {
                None
            };
            attr_dtype.or_else(|| get_dtype(expr, context.imports).map(|x| x.to_string()))
        });
        let dims = vec![dim];
        Some(Shape { dtype, dims })
    } else {
        None
    }
}

pub(crate) fn infer_linear_module(
    mut base: Shape,
    in_features: Option<&str>,
    out_features: Option<&str>,
    range: TextRange,
    diagnostics: &mut Vec<Diagnostic>,
    source: &str,
) -> Option<Shape> {
    let Some(last_dim) = base.dims.last_mut() else {
        diagnostics.push(Diagnostic {
            range: text_range_to_lsp(range, source),
            severity: Some(DiagnosticSeverity::ERROR),
            code: None,
            code_description: None,
            source: Some("shapels".into()),
            message: "Input tensor has incorrect dims for Linear".into(),
            related_information: None,
            tags: None,
            data: None,
        });
        return None;
    };
    if let Some(expected) = in_features
        && concrete_dim_mismatch(last_dim, expected)
    {
        diagnostics.push(Diagnostic {
            range: text_range_to_lsp(range, source),
            severity: Some(DiagnosticSeverity::ERROR),
            code: None,
            code_description: None,
            source: Some("shapels".into()),
            message: format!("Linear input feature mismatch: expected {expected}, got {last_dim}"),
            related_information: None,
            tags: None,
            data: None,
        });
    }
    if let Some(out_features) = out_features {
        *last_dim = out_features.to_string();
    }
    Some(base)
}

pub(crate) fn infer_batchnorm_module(
    base: Shape,
    dims: usize,
    num_features: Option<&str>,
    range: TextRange,
    diagnostics: &mut Vec<Diagnostic>,
    source: &str,
) -> Option<Shape> {
    let rank = base.dims.len();
    let valid_rank = match dims {
        1 => matches!(rank, 2 | 3),
        2 => rank == 4,
        3 => rank == 5,
        _ => false,
    };
    if !valid_rank {
        diagnostics.push(Diagnostic {
            range: text_range_to_lsp(range, source),
            severity: Some(DiagnosticSeverity::ERROR),
            code: None,
            code_description: None,
            source: Some("shapels".into()),
            message: format!("Input tensor has incorrect dims for BatchNorm{dims}d"),
            related_information: None,
            tags: None,
            data: None,
        });
        return None;
    }
    if let Some(expected) = num_features
        && let Some(channels) = base.dims.get(1)
        && concrete_dim_mismatch(channels, expected)
    {
        diagnostics.push(Diagnostic {
            range: text_range_to_lsp(range, source),
            severity: Some(DiagnosticSeverity::ERROR),
            code: None,
            code_description: None,
            source: Some("shapels".into()),
            message: format!(
                "BatchNorm{dims}d channel mismatch: expected {expected}, got {channels}"
            ),
            related_information: None,
            tags: None,
            data: None,
        });
    }
    Some(base)
}

#[derive(Clone, Debug)]
pub struct FlattenDims<'a> {
    pub start_dim: Option<Cow<'a, str>>,
    pub end_dim: Option<Cow<'a, str>>,
}

impl<'a> FlattenDims<'a> {
    fn unwrap_with_len(&self, base_len: usize) -> (Cow<'_, str>, Cow<'_, str>) {
        (
            self.start_dim
                .as_deref()
                .map(Cow::Borrowed)
                .unwrap_or(Cow::Borrowed("0")),
            self.end_dim
                .as_deref()
                .map(Cow::Borrowed)
                .unwrap_or_else(|| Cow::Owned((base_len - 1).to_string())),
        )
    }
}

/// Parse arguments from a call to `.flatten`.
pub fn get_flatten_dims<'a, 'b: 'a>(
    offset: usize,
    call: &'a ExprCall,
    vars: &'b HashMap<Identifier, VarState>,
    diagnostics: &mut Vec<Diagnostic>,
    source: &'b str,
) -> FlattenDims<'a> {
    let start_dim = get_arg(call, "start_dim", offset)
        .and_then(|e| expr_to_dim_token(e, vars, diagnostics, source, &mut false));
    let end_dim = get_arg(call, "end_dim", offset + 1)
        .and_then(|e| expr_to_dim_token(e, vars, diagnostics, source, &mut false));
    FlattenDims { start_dim, end_dim }
}

/// Inference for flatten from args parsed as [`FlattenDims`].
pub fn infer_flatten(
    base: Shape,
    call_range: &TextRange,
    flatten_arg_dims: &FlattenDims,
    diagnostics: &mut Vec<Diagnostic>,
    source: &str,
) -> Option<Shape> {
    let base_len = base.dims.len();

    let (start_dim, end_dim) = flatten_arg_dims.unwrap_with_len(base_len);
    if let (Ok(start), Ok(end)) = (start_dim.parse::<i32>(), end_dim.parse::<i32>()) {
        let start = resolve_dim_in_bounds(start, base_len, diagnostics, source, call_range)?;
        let end = resolve_dim_in_bounds(end, base_len, diagnostics, source, call_range)?;
        let mut out_dims = vec![String::new(); base_len - (end - start)];
        // fill in left and right of [start, end) interval
        let mut out_oft = 0;
        for i in 0..base_len {
            if i >= start && i < end {
                out_oft += 1;
            } else {
                out_dims[i - out_oft] = base.dims[i].clone();
            }
        }
        // accumulate symbolic and concrete dims separately
        out_dims[start] = product_token(&base.dims[start..=end]);
        Some(Shape {
            dtype: base.dtype,
            dims: out_dims,
        })
    } else {
        diagnostics.push(Diagnostic {
            range: text_range_to_lsp(*call_range, source),
            severity: Some(DiagnosticSeverity::INFORMATION),
            code: None,
            code_description: None,
            source: Some("shapels".into()),
            message: "Shape of flatten cannot be computed statically from non-concrete dims."
                .to_string(),
            related_information: None,
            tags: None,
            data: None,
        });
        None
    }
}

fn resolve_dim_in_bounds(
    dim: i32,
    base_len: usize,
    diagnostics: &mut Vec<Diagnostic>,
    source: &str,
    range: &TextRange,
) -> Option<usize> {
    let dim = if dim < 0 { base_len as i32 + dim } else { dim } as usize;
    if dim >= base_len {
        diagnostics.push(Diagnostic {
            range: text_range_to_lsp(*range, source),
            severity: Some(DiagnosticSeverity::ERROR),
            code: None,
            code_description: None,
            source: Some("shapels".into()),
            message: format!("Dim {dim} is out of bounds for tensor with ndims {base_len}"),
            related_information: None,
            tags: None,
            data: None,
        });
        return None;
    }
    Some(dim)
}

fn push_error_diagnostic(
    diagnostics: &mut Vec<Diagnostic>,
    range: TextRange,
    source: &str,
    message: String,
) {
    diagnostics.push(Diagnostic {
        range: text_range_to_lsp(range, source),
        severity: Some(DiagnosticSeverity::ERROR),
        code: None,
        code_description: None,
        source: Some("shapels".into()),
        message,
        related_information: None,
        tags: None,
        data: None,
    });
}

/// Inference for [`ast::UnaryOp`] such as
///
/// ```python
/// a = ~x
/// b = not x.sum()
/// c = -b
/// ```
///
/// Check dtype and, if compatible, return the base `shape` or `None` if op is `Not`.
pub fn infer_unary_dtype(
    op: ast::UnaryOp,
    shape: Shape,
    range: TextRange,
    diagnostics: &mut Vec<Diagnostic>,
    source: &str,
) -> Option<Shape> {
    // Not is independent of the dtype
    if matches!(op, ast::UnaryOp::Not) {
        if shape.dims.iter().any(|dim| dim.trim_ascii() != "1") {
            push_error_diagnostic(
                diagnostics,
                range,
                source,
                format!(
                    "Not is ambiguous for tensor with more than one value {}",
                    shape.render(),
                ),
            );
            return None;
        } else {
            // shape was singleton or empty
            return None;
        }
    }
    let Some(dtype) = shape.dtype.as_deref() else {
        // if dtype is unknown, just return the shape
        return Some(shape);
    };
    match (op, SimpleDtype::from(dtype)) {
        (ast::UnaryOp::Invert, SimpleDtype::Float) => {
            push_error_diagnostic(
                diagnostics,
                range,
                source,
                format!("Bitwise invert only supports integer or bool dtypes, found {dtype}"),
            );
            None
        }
        (ast::UnaryOp::UAdd | ast::UnaryOp::USub, SimpleDtype::Bool) => {
            push_error_diagnostic(
                diagnostics,
                range,
                source,
                format!("Unary +/- operations do not support bool dtype, found {dtype}"),
            );
            None
        }
        _ => Some(shape),
    }
}

fn base_shape_or_diag(
    base_expr: &Expr,
    record_hovers: bool,
    mut context: ContextRef,
) -> Option<Shape> {
    let maybe_base_shape = context
        .lookup_shape(base_expr, record_hovers)
        .or_else(|| context.infer_shape(base_expr, false));
    if maybe_base_shape.is_none() {
        context.push_diagnostic_text(
            expr_text_range(base_expr),
            DiagnosticSeverity::INFORMATION,
            String::from("Tensor shape unknown at this point"),
        );
    }
    maybe_base_shape
}

/// [`torch.where`](https://docs.pytorch.org/docs/stable/generated/torch.where.html)
///
/// This is a Noop shape-wise but requires checking that the shapes of the tensors
/// in the arguments (possibly only a value for `input` and `other`) match.
///
/// Also, the base_expr `condition` must be of dtype `bool`.
///
/// The inferred shape is of any dtype found in the `input` and `other`, by default, `Float`.
/// At runtime, there is coercion to most precision and from integer to floats when
/// `input.dtype != other.dtype`, but this is not model for now.
pub fn infer_condition(
    base_expr: &Expr,
    call: &ExprCall<TextRange>,
    record_hovers: bool,
    offset: usize,
    mut context: ContextRef,
) -> Option<Shape> {
    let maybe_base_shape = base_shape_or_diag(base_expr, record_hovers, context.reborrow());
    // TODO(carrascomj): coercion of dtypes to the most precise and floaty.
    let mut dtype = Some("Float".to_string());
    if let Some(base_shape) = maybe_base_shape.as_ref() {
        if let Some(SimpleDtype::Float | SimpleDtype::Int { .. }) = base_shape
            .dtype
            .as_ref()
            .map(|x| SimpleDtype::from(x.as_str()))
        {
            context.push_diagnostic_text(
                expr_text_range(base_expr),
                DiagnosticSeverity::ERROR,
                String::from("Condition must be of boolean dtype"),
            );
        }
        for (arg_name, off) in [("input", offset), ("other", offset + 1)] {
            if let Some(arg_expr) = get_arg(call, arg_name, off)
                && let Some(arg_shape) = context
                    .lookup_shape(arg_expr, record_hovers)
                    .or_else(|| context.infer_shape(arg_expr, false))
            {
                if arg_shape.dims != base_shape.dims {
                    context.push_diagnostic_text(
                        expr_text_range(arg_expr),
                        DiagnosticSeverity::ERROR,
                        format!(
                            "Condition vs {} must have the same shape: {} vs {}",
                            arg_name,
                            arg_shape.render(),
                            base_shape.render()
                        ),
                    )
                } else {
                    dtype = arg_shape.dtype;
                }
            }
        }
    }
    Some(Shape {
        dtype,
        dims: maybe_base_shape.map(|x| x.dims).unwrap_or_default(),
    })
}

pub fn infer_take(
    call: &ExprCall<TextRange>,
    record_hovers: bool,
    offset: usize,
    mut context: ContextRef,
) -> Option<Shape> {
    let index_shape = base_shape_or_diag(
        get_arg(call, "index", offset)?,
        record_hovers,
        context.reborrow(),
    )?;
    if !index_shape
        .dtype
        .as_ref()
        .map(|x| SimpleDtype::from(x.as_str()).is_long())
        .unwrap_or(false)
    {
        context.push_diagnostic_text(
            call.range,
            DiagnosticSeverity::ERROR,
            "Index must be of dtype long".to_string(),
        );
    }
    Some(index_shape)
}
