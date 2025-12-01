//! Specialized inference of `Shape`s for the various implemented operations.
//!
//! Word of caution: the `dim` argument in torch methods and functions expects
//! a int64_t in the ATen implementation
//! (e.g., [here](https://github.com/pytorch/pytorch/blob/9f7fceb887d0cfa0326a59b887821c63ff11340a/torch/csrc/lazy/core/ops/utils.cpp#L92)).
//! However, in functions like (un)squeeze or reduce ops, shapels parses dim as i16 because
//! i64 is excessive for operations that relate to the number of dimensions and
//! not the dimenions themselves. This should be revisited if bugs come.
#![allow(clippy::too_many_arguments)]
use crate::{HoverInfo, Imports, ModuleCache, Shape, VarState, expr_text_range};
use lsp_types::{Diagnostic, DiagnosticSeverity, Range};
use rustpython_parser::ast::{self, Arguments, Expr, ExprBinOp, Identifier, Operator, Stmt};
use rustpython_parser::text_size::TextRange;
use std::collections::HashMap;
use std::path::Path;

use crate::{infer_expr_shape, lookup_shape, text_range_to_lsp};

/// Inference and matrix multiplication shape inference shared by `@` and `torch.mm`.
/// Keeps all leading dims of left except the last, then appends all trailing dims of right except the first.
pub fn infer_matmul_shapes(
    left: &Expr,
    right: &Expr,
    vars: &HashMap<Identifier, VarState>,
    func_map: &HashMap<Identifier, (Box<Arguments>, Vec<Stmt>)>,
    imports: &Imports,
    call_stack: &mut Vec<Identifier>,
    diagnostics: &mut Vec<Diagnostic>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    record_hovers: bool,
    source: &str,
    whole_range: TextRange,
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> Option<Shape> {
    let left_shape = lookup_shape(left, vars, hover_entries, record_hovers, source).or_else(|| {
        infer_expr_shape(
            left,
            vars,
            func_map,
            imports,
            call_stack,
            diagnostics,
            hover_entries,
            false,
            source,
            module_cache.as_deref_mut(),
            module_path,
        )
    });
    let right_shape =
        lookup_shape(right, vars, hover_entries, record_hovers, source).or_else(|| {
            infer_expr_shape(
                right,
                vars,
                func_map,
                imports,
                call_stack,
                diagnostics,
                hover_entries,
                false,
                source,
                module_cache.as_deref_mut(),
                module_path,
            )
        });
    match (left_shape, right_shape) {
        (Some(l), Some(r)) => match matmul(&l, &r) {
            Ok(shape) => Some(shape),
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
        },
        _ => None,
    }
}

/// Matrix multiplication shape inference shared by `@` and `torch.mm`.
/// Keeps all leading dims of left except the last, then appends all trailing dims of right except the first.
fn matmul(left: &Shape, right: &Shape) -> Result<Shape, String> {
    if left.dims.is_empty() || right.dims.is_empty() {
        return Err("Matmul requires both operands to have shapes".into());
    }
    let left_inner = left.dims.last().unwrap();
    let right_inner = right.dims.first().unwrap();
    if left_inner != right_inner {
        return Err(format!(
            "Matmul inner dimensions mismatch: {} vs {}",
            left_inner, right_inner
        ));
    }
    let mut dims: Vec<String> = left.dims[..left.dims.len() - 1].to_vec();
    dims.extend_from_slice(&right.dims[1..]);
    Ok(Shape {
        dtype: left.dtype.clone().or(right.dtype.clone()),
        dims,
    })
}

/// Shared logic for squeeze (enforce_one = true) and aggregation (enforce_one = false).
///
/// An aggregation (or reduce) operation such as sum or amin is shape-wise the same
/// as an squeeze only that squeezes only applies to dim==1 and should diagnose otherwise.
pub fn infer_squeeze(
    base_expr: &Expr,
    dim_arg: Option<&Expr>,
    vars: &HashMap<Identifier, VarState>,
    func_map: &HashMap<Identifier, (Box<Arguments>, Vec<Stmt>)>,
    imports: &Imports,
    call_stack: &mut Vec<Identifier>,
    diagnostics: &mut Vec<Diagnostic>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    record_hovers: bool,
    source: &str,
    whole_range: TextRange,
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
    enforce_one: bool,
) -> Option<Shape> {
    let diag_before = diagnostics.len();
    let base_shape = infer_expr_shape(
        base_expr,
        vars,
        func_map,
        imports,
        call_stack,
        diagnostics,
        hover_entries,
        record_hovers,
        source,
        module_cache.as_deref_mut(),
        module_path,
    )
    .or_else(|| {
        infer_shallow_shape(
            base_expr,
            vars,
            func_map,
            imports,
            call_stack,
            diagnostics,
            hover_entries,
            record_hovers,
            source,
            module_cache,
            module_path,
        )
    })?;

    // No dim specified: squeeze removes ones, sum collapses all dims.
    if dim_arg.is_none() {
        let mut dims = base_shape.dims.clone();
        if enforce_one {
            dims.retain(|d| d != "1");
        } else {
            dims.clear();
        }
        return Some(Shape {
            dtype: base_shape.dtype.clone(),
            dims,
        });
    }

    let dims_to_remove =
        match parse_dims(dim_arg.unwrap(), base_shape.dims.len(), diagnostics, source) {
            Ok(v) => v,
            Err(_) => return None,
        };

    let mut dims = base_shape.dims.clone();
    for idx in dims_to_remove.into_iter().rev() {
        if idx >= dims.len() {
            diagnostics.push(Diagnostic {
                range: text_range_to_lsp(whole_range, source),
                severity: Some(DiagnosticSeverity::ERROR),
                code: None,
                code_description: None,
                source: Some("shapels".into()),
                message: "Invalid dim".into(),
                related_information: None,
                tags: None,
                data: None,
            });
            continue;
        }
        if enforce_one && dims.get(idx).map(|d| d != "1").unwrap_or(false) {
            diagnostics.push(Diagnostic {
                range: text_range_to_lsp(whole_range, source),
                severity: Some(DiagnosticSeverity::ERROR),
                code: None,
                code_description: None,
                source: Some("shapels".into()),
                message: "Cannot squeeze dimension not equal to 1".into(),
                related_information: None,
                tags: None,
                data: None,
            });
            continue;
        }
        dims.remove(idx);
    }
    Some(Shape {
        dtype: base_shape.dtype.clone(),
        dims,
    })
    .or_else(|| {
        if diagnostics.len() == diag_before {
            diagnostics.push(Diagnostic {
                range: text_range_to_lsp(whole_range, source),
                severity: Some(DiagnosticSeverity::ERROR),
                code: None,
                code_description: None,
                source: Some("shapels".into()),
                message: "Invalid dim".into(),
                related_information: None,
                tags: None,
                data: None,
            });
        }
        None
    })
}

fn infer_shallow_shape(
    expr: &Expr,
    vars: &HashMap<Identifier, VarState>,
    func_map: &HashMap<Identifier, (Box<Arguments>, Vec<Stmt>)>,
    imports: &Imports,
    call_stack: &mut Vec<Identifier>,
    diagnostics: &mut Vec<Diagnostic>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    record_hovers: bool,
    source: &str,
    module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> Option<Shape> {
    match expr {
        Expr::BinOp(ExprBinOp {
            left,
            op,
            right,
            range,
        }) => {
            if matches!(op, Operator::MatMult) {
                return infer_matmul_shapes(
                    left,
                    right,
                    vars,
                    func_map,
                    imports,
                    call_stack,
                    diagnostics,
                    hover_entries,
                    record_hovers,
                    source,
                    *range,
                    module_cache,
                    module_path,
                );
            }
            None
        }
        _ => lookup_shape(expr, vars, hover_entries, record_hovers, source),
    }
}

fn expr_to_dim_token(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Constant(c) => match &c.value {
            ast::Constant::Int(i) => Some(i.to_string()),
            _ => None,
        },
        Expr::Name(n) => Some(n.id.to_string()),
        Expr::BinOp(bin) => {
            if matches!(bin.op, Operator::Mult) {
                let l = expr_to_dim_token(&bin.left)?;
                let r = expr_to_dim_token(&bin.right)?;
                Some(format!("{l}*{r}"))
            } else {
                None
            }
        }
        Expr::UnaryOp(u) => match (u.op, u.operand.as_ref()) {
            (ast::UnaryOp::USub, Expr::Constant(c)) => {
                if let ast::Constant::Int(i) = &c.value {
                    let s = i.to_string();
                    Some(format!("-{s}"))
                } else {
                    None
                }
            }
            _ => None,
        },
        _ => None,
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
    diagnostics: &mut Vec<Diagnostic>,
    source: &str,
) -> Result<Vec<usize>, ()> {
    let to_i16 = |e: &Expr| expr_to_dim_token(e).and_then(|s| s.parse::<i16>().ok());
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
    let mut has_err = false;
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

pub fn infer_unsqueeze(
    base_expr: &Expr,
    dim_arg: Option<&Expr>,
    vars: &HashMap<Identifier, VarState>,
    func_map: &HashMap<Identifier, (Box<Arguments>, Vec<Stmt>)>,
    imports: &Imports,
    call_stack: &mut Vec<Identifier>,
    diagnostics: &mut Vec<Diagnostic>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    record_hovers: bool,
    source: &str,
    module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> Option<Shape> {
    let base_shape = infer_expr_shape(
        base_expr,
        vars,
        func_map,
        imports,
        call_stack,
        diagnostics,
        hover_entries,
        record_hovers,
        source,
        module_cache,
        module_path,
    );
    let base_shape = base_shape?;
    let dim = dim_arg.and_then(expr_to_dim_token)?;
    let dim_i: i16 = dim.parse().ok()?;
    let idx = normalize_dim_index_unsqueeze(dim_i, base_shape.dims.len())?;
    let mut dims = base_shape.dims.clone();
    dims.insert(idx, "1".to_string());
    Some(Shape {
        dtype: base_shape.dtype.clone(),
        dims,
    })
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

pub fn infer_view_like(
    base_expr: &Expr,
    args: &[&Expr],
    vars: &HashMap<Identifier, VarState>,
    diagnostics: &mut Vec<Diagnostic>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    record_hovers: bool,
    source: &str,
    whole_range: TextRange,
) -> Option<Shape> {
    let base_shape = lookup_shape(base_expr, vars, hover_entries, record_hovers, source);
    let target_tokens = args
        .iter()
        .filter_map(|e| expr_to_dim_token(e))
        .collect::<Vec<_>>();
    if target_tokens.is_empty() {
        return None;
    }
    let res = reshape_dims(base_shape.as_ref(), &target_tokens);
    match res {
        Ok(shape) => Some(shape),
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

fn reshape_dims(base: Option<&Shape>, target: &[String]) -> Result<Shape, String> {
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
            tokens[idx] = inferred;
        }
        return Ok(Shape {
            dtype: base_shape.dtype.clone(),
            dims: tokens,
        });
    }

    // No base shape: still return with -1 replaced by "Infer"
    if let Some(idx) = minus_one_idx {
        tokens[idx] = "Infer".to_string();
    }
    Ok(Shape {
        dtype: None,
        dims: tokens,
    })
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

pub fn shape_dims_equal(a: &Shape, b: &Shape) -> bool {
    flatten_dims(&a.dims) == flatten_dims(&b.dims)
}

/// Inference shapes and produce element-wise broadcastable-operations.
pub fn infer_broadcastable_poswise(
    left: &Expr,
    right: &Expr,
    vars: &HashMap<Identifier, VarState>,
    func_map: &HashMap<Identifier, (Box<Arguments>, Vec<Stmt>)>,
    imports: &Imports,
    call_stack: &mut Vec<Identifier>,
    diagnostics: &mut Vec<Diagnostic>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    record_hovers: bool,
    source: &str,
    whole_range: TextRange,
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> Option<Shape> {
    let left_shape = lookup_shape(left, vars, hover_entries, record_hovers, source).or_else(|| {
        infer_expr_shape(
            left,
            vars,
            func_map,
            imports,
            call_stack,
            diagnostics,
            hover_entries,
            false,
            source,
            module_cache.as_deref_mut(),
            module_path,
        )
    });
    let right_shape =
        lookup_shape(right, vars, hover_entries, record_hovers, source).or_else(|| {
            infer_expr_shape(
                right,
                vars,
                func_map,
                imports,
                call_stack,
                diagnostics,
                hover_entries,
                false,
                source,
                module_cache.as_deref_mut(),
                module_path,
            )
        });

    match broadcastable_poswise(left_shape, right_shape) {
        Ok(shape_opt) => shape_opt,
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
    left: Option<Shape>,
    right: Option<Shape>,
) -> Result<Option<Shape>, String> {
    match (left, right) {
        (None, None) => Ok(None),
        (Some(s), None) | (None, Some(s)) => Ok(Some(s)),
        (Some(l), Some(r)) => {
            if l.dims.is_empty() || r.dims.is_empty() {
                return Err("Broadcasting requires operands with at least one dimension".into());
            }
            let dims = broadcast_dims(&l.dims, &r.dims)
                .map_err(|msg| format!("Broadcasting incompatible shapes: {msg}"))?;
            Ok(Some(Shape {
                dtype: l.dtype.clone().or(r.dtype.clone()),
                dims,
            }))
        }
    }
}

fn broadcast_dims(a: &[String], b: &[String]) -> Result<Vec<String>, String> {
    if a.is_empty() || b.is_empty() {
        return Err("tensor has zero dimensions".into());
    }
    let mut out = Vec::new();
    let mut idx = 0usize;
    let max_len = a.len().max(b.len());
    while idx < max_len {
        let a_dim = a.get(a.len().wrapping_sub(1 + idx)).map(String::as_str);
        let b_dim = b.get(b.len().wrapping_sub(1 + idx)).map(String::as_str);
        let res = match (a_dim, b_dim) {
            (Some(ad), Some(bd)) if ad == bd => ad.to_string(),
            (Some(ad), Some(bd)) if ad == "1" => bd.to_string(),
            (Some(ad), Some(bd)) if bd == "1" => ad.to_string(),
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

/// Variants for an operation that transposes dimensions.
pub enum Transpose {
    Transpose,
    T,
    Permute,
}

/// Permute dimensions of a tensor based on provided order.
pub fn infer_permute(
    base_expr: &Expr,
    order_args: &[&Expr],
    transpose: Transpose,
    base_hint: Option<Shape>,
    vars: &HashMap<Identifier, VarState>,
    diagnostics: &mut Vec<Diagnostic>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    record_hovers: bool,
    source: &str,
    whole_range: TextRange,
) -> Option<Shape> {
    let base_shape = base_hint
        .or_else(|| lookup_shape(base_expr, vars, hover_entries, record_hovers, source))?;
    let mut order = Vec::new();
    match transpose {
        Transpose::Transpose => {
            if order_args.len() != 2 {
                diagnostics.push(Diagnostic {
                    range: text_range_to_lsp(whole_range, source),
                    severity: Some(DiagnosticSeverity::ERROR),
                    code: None,
                    code_description: None,
                    source: Some("shapels".into()),
                    message: "transpose expects exactly two dimensions".into(),
                    related_information: None,
                    tags: None,
                    data: None,
                });
                return None;
            }
            let mut dims = Vec::with_capacity(2);
            for expr in order_args {
                if let Expr::Constant(c) = expr {
                    if let ast::Constant::Int(i) = &c.value {
                        if let Ok(val) = i.to_string().parse::<isize>() {
                            if val >= 0 {
                                dims.push(val as usize);
                                continue;
                            }
                        }
                    }
                }
                diagnostics.push(Diagnostic {
                    range: text_range_to_lsp(expr_text_range(expr), source),
                    severity: Some(DiagnosticSeverity::ERROR),
                    code: None,
                    code_description: None,
                    source: Some("shapels".into()),
                    message: "Invalid transpose index".into(),
                    related_information: None,
                    tags: None,
                    data: None,
                });
                return None;
            }
            if dims.iter().any(|&d| d >= base_shape.dims.len()) || dims[0] == dims[1] {
                diagnostics.push(Diagnostic {
                    range: text_range_to_lsp(whole_range, source),
                    severity: Some(DiagnosticSeverity::ERROR),
                    code: None,
                    code_description: None,
                    source: Some("shapels".into()),
                    message: "Invalid transpose dimensions".into(),
                    related_information: None,
                    tags: None,
                    data: None,
                });
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
                if let Expr::Constant(c) = expr {
                    if let ast::Constant::Int(i) = &c.value {
                        if let Ok(val) = i.to_string().parse::<isize>() {
                            if val >= 0 {
                                order.push(val as usize);
                                continue;
                            }
                        }
                    }
                }
                diagnostics.push(Diagnostic {
                    range: text_range_to_lsp(expr_text_range(expr), source),
                    severity: Some(DiagnosticSeverity::ERROR),
                    code: None,
                    code_description: None,
                    source: Some("shapels".into()),
                    message: "Invalid permute index".into(),
                    related_information: None,
                    tags: None,
                    data: None,
                });
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
        diagnostics.push(Diagnostic {
            range: text_range_to_lsp(whole_range, source),
            severity: Some(DiagnosticSeverity::ERROR),
            code: None,
            code_description: None,
            source: Some("shapels".into()),
            message: if !matches!(transpose, Transpose::Permute) {
                "Invalid transpose dimensions".into()
            } else {
                "Invalid permute dimensions".into()
            },
            related_information: None,
            tags: None,
            data: None,
        });
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
