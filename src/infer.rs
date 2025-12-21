//! Specialized inference of `Shape`s for the various implemented operations.
//!
//! Word of caution: the `dim` argument in torch methods and functions expects
//! a int64_t in the ATen implementation
//! (e.g., [here](https://github.com/pytorch/pytorch/blob/9f7fceb887d0cfa0326a59b887821c63ff11340a/torch/csrc/lazy/core/ops/utils.cpp#L92)).
//! However, in functions like (un)squeeze or reduce ops, shapels parses dim as i16 because
//! i64 is excessive for operations that relate to the number of dimensions and
//! not the dimenions themselves. This should be revisited if bugs come.
#![allow(clippy::too_many_arguments, clippy::needless_option_as_deref)]
use crate::op_groups::{BroadcastOp, RangeOps, TorchOp};
use crate::{
    ClassMap, FuncMap, HoverInfo, Imports, ModuleCache, Shape, VarState, expr_text_range, get_arg,
    get_dtype,
};
use lsp_types::{Diagnostic, DiagnosticSeverity, Range};
use rustpython_parser::ast::{
    self, Constant, Expr, ExprBinOp, ExprCall, ExprSubscript, Identifier, Operator,
};
use rustpython_parser::text_size::TextRange;
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use crate::{infer_expr_shape, lookup_shape, text_range_to_lsp};
#[derive(Debug, Clone, Copy)]
enum IndexKind<'a> {
    NewAxis,
    Ellipsis,
    Keep, // inserted from an expanded ellipsis
    Int,
    Slice(Option<i64>),
    Bool(bool),
    Tensor(&'a Expr),
}

pub fn infer_index(
    base_expr: &Expr,
    slice: &Expr,
    vars: &HashMap<Identifier, VarState>,
    func_map: &FuncMap,
    imports: &Imports,
    class_map: &ClassMap,
    call_stack: &mut Vec<Identifier>,
    diagnostics: &mut Vec<Diagnostic>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    record_hovers: bool,
    source: &str,
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> Option<Shape> {
    let base_shape =
        lookup_shape(base_expr, vars, hover_entries, record_hovers, source).or_else(|| {
            infer_expr_shape(
                base_expr,
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
        })?;

    let mut indices: Vec<IndexKind<'_>> = match slice {
        Expr::Tuple(t) => t.elts.iter().map(parse_index_kind).collect(),
        other => vec![parse_index_kind(other)],
    };

    let consuming_without_ellipsis = indices
        .iter()
        .filter(|k| !matches!(k, IndexKind::Ellipsis))
        .filter(|k| consumes_axis(k))
        .count();
    let mut expanded = Vec::new();
    let mut ellipsis_done = false;
    for kind in indices.drain(..) {
        if matches!(kind, IndexKind::Ellipsis) && !ellipsis_done {
            ellipsis_done = true;
            let keep = base_shape
                .dims
                .len()
                .saturating_sub(consuming_without_ellipsis);
            for _ in 0..keep {
                expanded.push(IndexKind::Keep);
            }
        } else if !matches!(kind, IndexKind::Ellipsis) {
            expanded.push(kind);
        }
    }

    let mut output_dims = Vec::new();
    let mut base_idx = 0usize;
    let mut prefix_len = None;
    let mut advanced_shapes = Vec::new();
    let advanced_positions: Vec<usize> = expanded
        .iter()
        .enumerate()
        .filter_map(|(idx, k)| if is_advanced(k) { Some(idx) } else { None })
        .collect();
    let advanced_front =
        advanced_positions.len() > 1 && !advanced_positions.windows(2).all(|w| w[1] == w[0] + 1);

    for kind in expanded.iter() {
        if is_advanced(kind) {
            if prefix_len.is_none() {
                prefix_len = Some(output_dims.len());
            }
            if let Some(shape) = advanced_shape(
                kind,
                vars,
                func_map,
                imports,
                class_map,
                call_stack,
                diagnostics,
                hover_entries,
                record_hovers,
                source,
                module_cache.as_deref_mut(),
                module_path,
            ) {
                advanced_shapes.push(shape);
            }
            if consumes_axis(kind) {
                base_idx += 1;
            }
            continue;
        }
        match kind {
            IndexKind::NewAxis => output_dims.push("1".to_string()),
            IndexKind::Keep | IndexKind::Slice(_) => {
                if let Some(dim) = base_shape.dims.get(base_idx) {
                    let step = match kind {
                        IndexKind::Slice(step) => *step,
                        _ => None,
                    };
                    output_dims.push(apply_slice(dim, step));
                    base_idx += 1;
                }
            }
            IndexKind::Int => {
                base_idx += 1;
            }
            IndexKind::Ellipsis | IndexKind::Bool(_) | IndexKind::Tensor(_) => {}
        }
    }

    while base_idx < base_shape.dims.len() {
        output_dims.push(base_shape.dims[base_idx].clone());
        base_idx += 1;
    }

    if advanced_shapes.is_empty() {
        return Some(Shape {
            dtype: base_shape.dtype.clone(),
            dims: output_dims,
        });
    }

    let adv_shape = broadcast_shapes(&advanced_shapes)?;
    let insert_pos = if advanced_front {
        0
    } else {
        prefix_len.unwrap_or(0)
    };
    output_dims.splice(insert_pos..insert_pos, adv_shape);

    Some(Shape {
        dtype: base_shape.dtype.clone(),
        dims: output_dims,
    })
}

fn is_advanced(kind: &IndexKind<'_>) -> bool {
    matches!(kind, IndexKind::Bool(_) | IndexKind::Tensor(_))
}

fn consumes_axis(kind: &IndexKind<'_>) -> bool {
    matches!(
        kind,
        IndexKind::Int | IndexKind::Slice(_) | IndexKind::Keep | IndexKind::Tensor(_)
    )
}

fn parse_index_kind(expr: &Expr) -> IndexKind<'_> {
    match expr {
        Expr::Constant(c) => match &c.value {
            ast::Constant::None => IndexKind::NewAxis,
            ast::Constant::Ellipsis => IndexKind::Ellipsis,
            ast::Constant::Bool(b) => IndexKind::Bool(*b),
            ast::Constant::Int(_) => IndexKind::Int,
            _ => IndexKind::Tensor(expr),
        },
        Expr::Name(n) if n.id.as_str() == "Ellipsis" => IndexKind::Ellipsis,
        Expr::Slice(s) => IndexKind::Slice(s.step.as_deref().and_then(expr_to_int)),
        Expr::Tuple(_) => IndexKind::Tensor(expr),
        _ => IndexKind::Tensor(expr),
    }
}

fn expr_to_int(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Constant(c) => match &c.value {
            ast::Constant::Int(i) => i.to_string().parse::<i64>().ok(),
            _ => None,
        },
        _ => None,
    }
}

fn apply_slice(dim: &str, step: Option<i64>) -> String {
    match step {
        Some(s) if s > 1 => format!("{dim}/{s}"),
        Some(s) if s < -1 => format!("{dim}/{}", s.abs()),
        _ => dim.to_string(),
    }
}

fn advanced_shape<'a>(
    kind: &IndexKind<'a>,
    vars: &HashMap<Identifier, VarState>,
    func_map: &FuncMap,
    imports: &Imports,
    class_map: &ClassMap,
    call_stack: &mut Vec<Identifier>,
    diagnostics: &mut Vec<Diagnostic>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    record_hovers: bool,
    source: &str,
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> Option<Vec<String>> {
    match kind {
        IndexKind::Bool(b) => Some(vec![(if *b { "1" } else { "0" }).to_string()]),
        IndexKind::Tensor(expr) => {
            if let Some(shape) = infer_expr_shape(
                expr,
                vars,
                func_map,
                imports,
                class_map,
                call_stack,
                diagnostics,
                hover_entries,
                record_hovers,
                source,
                module_cache.as_deref_mut(),
                module_path,
            ) {
                return Some(shape.dims);
            }
            literal_shape(expr)
        }
        _ => None,
    }
}

fn literal_shape(expr: &Expr) -> Option<Vec<String>> {
    match expr {
        Expr::List(l) => Some(list_shape(&l.elts)),
        Expr::Tuple(t) => Some(list_shape(&t.elts)),
        Expr::Call(call) => {
            if let Expr::Attribute(attr) = call.func.as_ref()
                && attr.attr.as_str() == "tensor"
                && let Some(arg0) = call.args.first()
            {
                return literal_shape(arg0);
            }
            None
        }
        _ => None,
    }
}

fn list_shape(elts: &[Expr]) -> Vec<String> {
    let len = elts.len();
    if let Some(first) = elts.first()
        && matches!(first, Expr::List(_) | Expr::Tuple(_))
    {
        let mut dims = vec![len.to_string()];
        dims.extend(literal_shape(first).unwrap_or_default());
        dims
    } else {
        vec![len.to_string()]
    }
}

fn broadcast_shapes(shapes: &[Vec<String>]) -> Option<Vec<String>> {
    let max_len = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
    if max_len == 0 {
        return None;
    }
    let mut result = Vec::with_capacity(max_len);
    for idx in 0..max_len {
        let mut current: Option<String> = None;
        for shape in shapes {
            let offset = max_len.saturating_sub(shape.len());
            let dim = if idx < offset {
                "1".to_string()
            } else {
                shape
                    .get(idx - offset)
                    .cloned()
                    .unwrap_or_else(|| "1".to_string())
            };
            current = Some(match current {
                None => dim,
                Some(cur) if cur == dim => cur,
                Some(cur) if cur == "1" => dim,
                Some(cur) if dim == "1" => cur,
                Some(cur) if cur == "0" || dim == "0" => "0".to_string(),
                Some(cur) => cur,
            });
        }
        result.push(current.unwrap_or_else(|| "1".to_string()));
    }
    Some(result)
}

/// Inference and matrix multiplication shape inference shared by `@` and `torch.mm`.
/// Keeps all leading dims of left except the last, then appends all trailing dims of right except the first.
pub fn infer_matmul_shapes(
    left: &Expr,
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
) -> Option<Shape> {
    let left_shape = lookup_shape(left, vars, hover_entries, record_hovers, source).or_else(|| {
        infer_expr_shape(
            left,
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
    enforce_one: bool,
) -> Option<Shape> {
    let diag_before = diagnostics.len();
    let base_shape = infer_expr_shape(
        base_expr,
        vars,
        func_map,
        imports,
        class_map,
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
            class_map,
            call_stack,
            diagnostics,
            hover_entries,
            record_hovers,
            source,
            module_cache,
            module_path,
        )
    })?;

    let dims_to_remove = if let Some(dim) = dim_arg {
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
        return Some(Shape {
            dtype: base_shape.dtype.clone(),
            dims,
        });
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
    func_map: &FuncMap,
    imports: &Imports,
    class_map: &ClassMap,
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
                    class_map,
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

pub fn infer_unsqueeze(
    base_expr: &Expr,
    dim_arg: Option<&Expr>,
    vars: &HashMap<Identifier, VarState>,
    func_map: &FuncMap,
    imports: &Imports,
    class_map: &ClassMap,
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
        class_map,
        call_stack,
        diagnostics,
        hover_entries,
        record_hovers,
        source,
        module_cache,
        module_path,
    );
    let base_shape = base_shape?;
    let dim =
        dim_arg.and_then(|expr| expr_to_dim_token(expr, vars, diagnostics, source, &mut false))?;
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
    torch_op: &TorchOp,
    base_hint: Option<Shape>,
    vars: &HashMap<Identifier, VarState>,
    diagnostics: &mut Vec<Diagnostic>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    record_hovers: bool,
    source: &str,
    whole_range: TextRange,
) -> Option<Shape> {
    let base_shape =
        base_hint.or_else(|| lookup_shape(base_expr, vars, hover_entries, record_hovers, source));
    let target_tokens = args
        .iter()
        .filter_map(|e| expr_to_dim_token(e, vars, diagnostics, source, &mut false))
        .collect::<Vec<_>>();
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

pub fn shape_dims_equal(a: &Shape, b: &Shape) -> bool {
    a.dims.iter().zip(b.dims.iter()).all(|(left, right)| {
        match (left.parse::<i32>().is_ok(), right.parse::<i32>().is_ok()) {
            (true, true) => left == right,
            (false, false) => left == right,
            _ => true,
        }
    })
}

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
    if matches!(broadcast_op, BroadcastOp::Bitwise) {
        for (text_range, shape) in [
            (left_range, &left_shape),
            (expr_text_range(right), &right_shape.as_ref()),
        ] {
            if let Some(Shape {
                dtype: Some(dtype), ..
            }) = shape
            {
                // TODO(carrascomj): a bit hacky, should be more systematic like get_dtype
                let dlower = dtype.to_lowercase();
                if !(dlower.contains("int") || dlower.contains("bool")) {
                    diagnostics.push(Diagnostic {
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
                    });
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
        Transpose::Explicit => {
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
                if let Expr::Constant(c) = expr
                    && let ast::Constant::Int(i) = &c.value
                    && let Ok(val) = i.to_string().parse::<isize>()
                    && val >= 0
                {
                    dims.push(val as usize);
                    continue;
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
                if let Expr::Constant(c) = expr
                    && let ast::Constant::Int(i) = &c.value
                    && let Ok(val) = i.to_string().parse::<isize>()
                    && val >= 0
                {
                    order.push(val as usize);
                    continue;
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
    match x {
        Expr::Name(name) => Some(Cow::Borrowed(name.id.as_str())),
        Expr::Constant(constant) => match &constant.value {
            Constant::Int(int) => Some(Cow::Owned(int.to_string())),
            // float is needed for ranges' steps but it's not valid for most dims
            Constant::Float(float) => Some(Cow::Owned(float.to_string())),
            _ => {
                if !*diag_already {
                    diagnostics.push(Diagnostic {
                        range: text_range_to_lsp(constant.range, source),
                        severity: Some(DiagnosticSeverity::INFORMATION),
                        code: None,
                        code_description: None,
                        source: Some("shapels".into()),
                        message: "Dim could be inferred".into(),
                        related_information: None,
                        tags: None,
                        data: None,
                    });
                    *diag_already = true;
                }
                None
            }
        },
        Expr::BinOp(bin) => {
            if matches!(bin.op, Operator::Mult) {
                let l = expr_to_dim_token(&bin.left, vars, diagnostics, source, diag_already)?;
                let r = expr_to_dim_token(&bin.right, vars, diagnostics, source, diag_already)?;
                Some(Cow::Owned(format!("{l}*{r}")))
            } else {
                None
            }
        }
        Expr::UnaryOp(u) => match u.op {
            ast::UnaryOp::USub => {
                expr_to_dim_token(u.operand.as_ref(), vars, diagnostics, source, diag_already)
                    .map(|s| Cow::Owned(format!("-{s}")))
            }
            ast::UnaryOp::UAdd => {
                expr_to_dim_token(u.operand.as_ref(), vars, diagnostics, source, diag_already)
            }
            _ => None,
        },
        // e.g., torch.zeros(x.shape[int])
        Expr::Subscript(ExprSubscript { value, slice, .. }) => {
            if let (Expr::Attribute(attr), Expr::Constant(c)) = (value.as_ref(), slice.as_ref())
                && let Expr::Name(name) = attr.value.as_ref()
                && let Constant::Int(i) = &c.value
                && let Ok(idx) = usize::try_from(i)
                && attr.attr.as_str() == "shape"
            {
                vars.get(&name.id)
                    .and_then(|v| v.annotated.as_ref().or(v.inferred.as_ref()))
                    .and_then(|sh| sh.dims.get(idx).map(|dim| Cow::Borrowed(dim.as_str())))
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
                vars.get(&name.id)
                    .and_then(|v| v.annotated.as_ref().or(v.inferred.as_ref()))
                    .and_then(|sh| sh.dims.get(idx).map(|dim| Cow::Borrowed(dim.as_str())))
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
                None
            }
        }
        expr => {
            if !*diag_already {
                diagnostics.push(Diagnostic {
                    range: text_range_to_lsp(expr_text_range(expr), source),
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

pub fn infer_range_size(
    call: &ExprCall<TextRange>,
    range_op: RangeOps,
    vars: &HashMap<Identifier, VarState>,
    func_map: &FuncMap,
    imports: &Imports,
    class_map: &ClassMap,
    call_stack: &mut Vec<Identifier>,
    diagnostics: &mut Vec<Diagnostic>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    record_hovers: bool,
    source: &str,
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> Option<Shape> {
    // helper function
    let push_diag = |diag: &mut Vec<_>, range, msg: &str| {
        diag.push(Diagnostic {
            range: text_range_to_lsp(range, source),
            severity: Some(DiagnosticSeverity::INFORMATION),
            code: None,
            code_description: None,
            source: Some("shapels".into()),
            message: msg.into(),
            related_information: None,
            tags: None,
            data: None,
        })
    };

    if let Some(dim) = match range_op {
        RangeOps::Randperm => get_arg(call, "n", 0).map(Cow::Borrowed).or_else(|| {
            push_diag(diagnostics, call.range, "Failed shape init: `n` arg");
            None
        }),
        RangeOps::Linspace | RangeOps::Logspace => {
            get_arg(call, "steps", 2).map(Cow::Borrowed).or_else(|| {
                push_diag(diagnostics, call.range, "Failed shape init: `steps`");
                None
            })
        }
        RangeOps::Range | RangeOps::Arange => {
            let names = ["start", "end", "step"];
            let defaults: [Option<String>; 3] = [Some("0".into()), None, Some("1".into())];

            let arg_tuple: [Option<String>; 3] = std::array::from_fn(|i| {
                get_arg(call, names[i], i)
                    .and_then(|expr| {
                        expr_to_dim_token(expr, vars, diagnostics, source, &mut false)
                            .map(Cow::into_owned)
                    })
                    .or_else(|| defaults[i].clone())
            });
            if let [Some(start), Some(end), Some(step)] = arg_tuple {
                let plus_one = if range_op == RangeOps::Arange {
                    1.0
                } else {
                    0.0
                };
                let ident = match (
                    start.parse::<f32>(),
                    end.parse::<f32>(),
                    step.parse::<f32>(),
                ) {
                    (Ok(s), Ok(e), Ok(ste)) => ((e - s) / ste + plus_one).to_string(),
                    (Ok(s), Ok(e), Err(_)) => {
                        let plus_one = if plus_one == 0.0 { "" } else { "+1" };
                        (e - s).to_string() + format!("/{step}{plus_one}").as_str()
                    }
                    _ => {
                        let plus_one = if plus_one == 0.0 { "" } else { "+1" };
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
    }
    .as_deref()
    .and_then(|expr| expr_to_dim_token(expr, vars, diagnostics, source, &mut false))
    .or_else(|| {
        push_diag(
            diagnostics,
            call.range,
            "Argument was not understood as shape",
        );
        None
    }) {
        let dtype = get_arg(call, "dtype", range_op.dtype_arg_pos()).and_then(|expr| {
            // dtype as torch.Tensor.dtype
            let attr_dtype = if matches!(expr, Expr::Attribute(_)) {
                infer_expr_shape(
                    expr,
                    vars,
                    func_map,
                    imports,
                    class_map,
                    call_stack,
                    diagnostics,
                    hover_entries,
                    record_hovers,
                    source,
                    module_cache.as_deref_mut(),
                    module_path,
                )
                .and_then(|shape| shape.dtype)
            } else {
                None
            };
            attr_dtype.or_else(|| get_dtype(expr, imports).map(|x| x.to_string()))
        });
        let dims = vec![dim.into_owned()];
        Some(Shape { dtype, dims })
    } else {
        None
    }
}

pub fn infer_conv(
    base: Shape,
    kernel: Vec<String>,
    call: &ExprCall,
    conv_dim: usize,
    diagnostics: &mut Vec<Diagnostic>,
    source: &str,
) -> Option<Shape> {
    let n_dims = base.dims.len();
    let n_k_dims = kernel.len();
    let incorrect_input = n_dims - conv_dim > 2 || n_dims - conv_dim < 1;
    let incorrect_kernel = n_k_dims - conv_dim > 2 || n_k_dims - conv_dim < 1;
    if incorrect_input || incorrect_kernel {
        let (incorrect_dims, tensor_msg) = if !incorrect_input {
            (n_dims, "Input tensor")
        } else {
            (n_k_dims, "Kernel")
        };
        diagnostics.push(Diagnostic {
            range: text_range_to_lsp(call.range, source),
            severity: Some(DiagnosticSeverity::ERROR),
            code: None,
            code_description: None,
            source: Some("shapels".into()),
            message: format!(
                "{tensor_msg} has incorrect dims for conv{conv_dim}d: {} ∉ {{{},{}}}",
                incorrect_dims,
                conv_dim + 1,
                conv_dim + 2
            ),
            related_information: None,
            tags: None,
            data: None,
        });
        return None;
    } else if (n_k_dims - n_dims) > 1 {
        diagnostics.push(Diagnostic {
            range: text_range_to_lsp(call.range, source),
            severity: Some(DiagnosticSeverity::ERROR),
            code: None,
            code_description: None,
            source: Some("shapels".into()),
            message: format!(
                "Input and kernel dimensions do not match: {n_k_dims} - {n_dims} > 1",
            ),
            related_information: None,
            tags: None,
            data: None,
        });
        return None;
    }
    let out_channels = &kernel[0];
    let stride = get_arg(call, "stride", 3)
        .and_then(expr_to_tuple)
        .unwrap_or(vec!["1".to_string(); conv_dim]);
    let padding = get_arg(call, "padding", 4)
        .and_then(expr_to_tuple)
        .unwrap_or(vec!["0".to_string(); conv_dim]);
    let dilation = get_arg(call, "dilation", 5)
        .and_then(expr_to_tuple)
        .unwrap_or(vec!["1".to_string(); conv_dim]);
    let pos = n_dims - conv_dim;
    Some(Shape {
        dtype: base.dtype,
        dims: base
            .dims
            .into_iter()
            .enumerate()
            .map(|(i, dim)| {
                if i == (pos - 1) {
                    out_channels.clone()
                } else if i >= pos {
                    let spatial_i = i - pos;
                    let kernel_size = &kernel[2 + spatial_i];
                    let stride = stride.get(spatial_i).unwrap_or(&stride[0]);
                    let padding = padding.get(spatial_i).unwrap_or(&padding[0]);
                    let dilation = dilation.get(spatial_i).unwrap_or(&dilation[0]);
                    match (
                        dim.parse::<i32>(),
                        kernel_size.parse::<i32>(),
                        stride.parse::<i32>(),
                        padding.parse::<i32>(),
                        dilation.parse::<i32>(),
                    ) {
                        (Ok(n), Ok(k), Ok(s), Ok(p), Ok(d)) => {
                            (((n + 2 * p - d * (k - 1) - 1) / s) + 1).to_string()
                        }
                        (Err(_), Ok(k), Ok(s), Ok(p), Ok(d)) => {
                            format!("({dim}+{})/{s}+1", 2 * p - d * (k - 1) - 1)
                        }
                        (Ok(n), Ok(k), Err(_), Ok(p), Ok(d)) => {
                            format!("{}/{stride}+1", (n + 2 * p - d * (k - 1) - 1))
                        }
                        (Err(_), Ok(k), Err(_), Ok(p), Ok(d)) => {
                            let num = 2 * p - d * (k - 1) - 1;
                            format!("({dim}+{})/{stride}+1", num)
                        }
                        (_, Ok(k), Err(_), Err(_), Ok(d)) => {
                            // can't compute the numerator fully, but still keep it grouped
                            format!("({dim}+2*{padding}-{})/{stride}+1", d * (k - 1) - 1)
                        }
                        (Ok(n), Err(_), _, Err(_), Err(_)) => {
                            // can't compute the numerator fully, but still keep it grouped
                            format!(
                                "({}+2*{padding}-{dilation}*({kernel_size}-1))/{stride}+1",
                                n - 1
                            )
                        }
                        (Ok(n), Ok(k), _, Err(_), Err(_)) => {
                            // can't compute the numerator fully, but still keep it grouped
                            format!("({}+2*{padding}-{dilation}*{})/{stride}+1", n - 1, k - 1)
                        }
                        _ => {
                            format!(
                                "(({dim}+2*{padding}-{dilation}*({kernel_size}-1)-1)/{stride})+1"
                            )
                        }
                    }
                } else {
                    dim
                }
            })
            .collect(),
    })
}

// TODO(carrascomj): make this more robust by piggybacking on expr_to_dim_token
fn expr_to_tuple(expr: &Expr) -> Option<Vec<String>> {
    match expr {
        Expr::Name(n) => Some(vec![n.id.to_string()]),
        Expr::Constant(c) => match &c.value {
            ast::Constant::Int(i) => Some(vec![i.to_string()]),
            _ => None,
        },
        Expr::Tuple(tup) => Some(
            tup.elts
                .iter()
                .map(|x| match x {
                    Expr::Constant(c) => match &c.value {
                        ast::Constant::Int(i) => i.to_string(),
                        _ => "".to_string(),
                    },
                    Expr::Name(n) => n.id.to_string(),
                    _ => "".to_string(),
                })
                .collect(),
        ),
        _ => None,
    }
}

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

pub fn infer_flatten(
    base: Shape,
    offset: usize,
    call: &ExprCall,
    vars: &HashMap<Identifier, VarState>,
    diagnostics: &mut Vec<Diagnostic>,
    source: &str,
) -> Option<Shape> {
    let base_len = base.dims.len();
    let start_dim = get_arg(call, "start_dim", offset)
        .and_then(|e| expr_to_dim_token(e, vars, diagnostics, source, &mut false))
        .unwrap_or(Cow::Borrowed("0"));
    let end_dim = get_arg(call, "end_dim", offset + 1)
        .and_then(|e| expr_to_dim_token(e, vars, diagnostics, source, &mut false))
        .unwrap_or(Cow::Owned((base_len - 1).to_string()));
    if let (Ok(start), Ok(end)) = (start_dim.parse::<i32>(), end_dim.parse::<i32>()) {
        let start = resolve_dim_in_bounds(start, base_len, diagnostics, source, &call.range)?;
        let end = resolve_dim_in_bounds(end, base_len, diagnostics, source, &call.range)?;
        let mut out_dims = vec![String::new(); base_len - (end - start) as usize];
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
        let (sym_dim, conc_dim) = (start..(end + 1)).fold((String::new(), 1), |(sym, conc), i| {
            let dim = &base.dims[i];
            match dim.parse::<usize>() {
                Ok(d) => (sym, conc * d),
                Err(_) if sym.is_empty() => (dim.to_string(), conc),
                _ => (sym + "*" + dim, conc),
            }
        });
        out_dims[start] = if sym_dim.is_empty() {
            conc_dim.to_string()
        } else if conc_dim > 1 {
            sym_dim + &format!("*{conc_dim}")
        } else {
            sym_dim
        };
        Some(Shape {
            dtype: base.dtype,
            dims: out_dims,
        })
    } else {
        diagnostics.push(Diagnostic {
            range: text_range_to_lsp(call.range, source),
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
