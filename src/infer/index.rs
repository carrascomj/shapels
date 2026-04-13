//! inference for indexing operations.
use crate::context::ContextRef;
use crate::expr_tokens::{SLICE_BOUND_TOKEN_OPTIONS, expr_to_symbolic_token};
use crate::op_groups::SimpleDtype;
use crate::{Shape, expr_text_range};
use lsp_types::DiagnosticSeverity;
use rustpython_parser::ast::{self, Expr};
use rustpython_parser::text_size::TextRange;

use super::expr_to_int;

#[derive(Debug, Clone)]
enum IndexKind<'a> {
    NewAxis,  // [None, ]
    Ellipsis, // [..., ]
    Keep,     // inserted from an expanded ellipsis
    Int {
        value: i64,
        range: TextRange,
    },
    Slice {
        start: Option<String>,
        stop: Option<String>,
        step: Option<i64>,
        range: TextRange,
    },
    Bool(bool),
    Tensor(&'a Expr),
}

/// Infer shape from an index. supporting [`IndexKind`].
///
/// Bool indices cannot be known statically, but can be bounded like `[0:(inferred_upper_bound)]`.
pub fn infer_index(
    base_expr: &Expr,
    slice: &Expr,
    record_hovers: bool,
    mut context: ContextRef,
) -> Option<Shape> {
    let base_shape = context
        .lookup_shape(base_expr, record_hovers)
        .or_else(|| context.infer_shape(base_expr, false))?;

    let mut indices: Vec<IndexKind<'_>> = match slice {
        Expr::Tuple(t) => t
            .elts
            .iter()
            .map(|e| parse_index_kind(e, context.source))
            .collect(),
        other => vec![parse_index_kind(other, context.source)],
    };

    if let Some(shape) =
        infer_boolean_mask_index(&base_shape, &indices, record_hovers, context.reborrow())
    {
        return Some(shape);
    }

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
            if let Some(shape) = advanced_shape(kind, record_hovers, context.reborrow()) {
                advanced_shapes.push(shape);
            }
            if consumes_axis(kind) {
                base_idx += 1;
            }
            continue;
        }
        match kind {
            IndexKind::NewAxis => output_dims.push("1".to_string()),
            IndexKind::Keep | IndexKind::Slice { .. } => {
                if let Some(dim) = base_shape.dims.get(base_idx) {
                    match kind {
                        IndexKind::Slice {
                            start,
                            stop,
                            step,
                            range,
                        } => {
                            if let Some(step_val) = step
                                && *step_val <= 0
                            {
                                context.push_diagnostic_text(
                                    *range,
                                    DiagnosticSeverity::ERROR,
                                    "Slice step must be greater than 0".into(),
                                );
                            }
                            if let Ok(base_len) = dim.parse::<i64>() {
                                let start_num = start.as_ref().and_then(|s| s.parse::<i64>().ok());
                                let stop_num = stop.as_ref().and_then(|s| s.parse::<i64>().ok());
                                let step_num = step.filter(|s| *s > 0);
                                if let Some(len) =
                                    slice_len_from_bounds(base_len, start_num, stop_num, step_num)
                                {
                                    output_dims.push(len.to_string());
                                    base_idx += 1;
                                    continue;
                                }
                            }
                            if let Some(step) = step {
                                output_dims.push(apply_slice(dim, Some(*step)));
                            } else if start.is_none() && stop.is_none() {
                                output_dims.push(dim.clone());
                            } else if start.is_none() {
                                output_dims.push(stop.clone().unwrap_or_else(|| dim.clone()));
                            } else if stop.is_none() {
                                let base_tok = base_shape
                                    .dims
                                    .get(base_idx)
                                    .cloned()
                                    .unwrap_or_else(|| dim.clone());
                                let start_tok = start.clone().unwrap();
                                let new_dim = match (
                                    base_tok.parse::<i64>().ok(),
                                    start_tok.parse::<i64>().ok(),
                                ) {
                                    (Some(b), Some(s)) => (b - s).to_string(),
                                    (Some(b), None) => format!("{b}-{start_tok}"),
                                    _ => {
                                        let (primary, offset) = split_primary_offset(&start_tok);
                                        if let Some(offset) = offset {
                                            format!("{base_tok}-{primary}{offset}")
                                        } else {
                                            format!("{base_tok}-{primary}")
                                        }
                                    }
                                };
                                output_dims.push(new_dim);
                            } else {
                                output_dims.push(stop.clone().unwrap_or_else(|| dim.clone()));
                            }
                            base_idx += 1;
                        }
                        _ => {
                            output_dims.push(dim.clone());
                            base_idx += 1;
                        }
                    }
                }
            }
            IndexKind::Int { value, range } => {
                if let Some(dim) = base_shape.dims.get(base_idx)
                    && let Ok(base_len) = dim.parse::<i64>()
                {
                    let mut idx = *value;
                    if idx < 0 {
                        idx += base_len;
                    }
                    if idx < 0 || idx >= base_len {
                        context.push_diagnostic_text(
                            *range,
                            DiagnosticSeverity::ERROR,
                            format!(
                                "Index {} out of bounds for dimension size {}",
                                value, base_len
                            ),
                        );
                    }
                }
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

fn infer_boolean_mask_index<'a>(
    base_shape: &Shape,
    indices: &[IndexKind<'a>],
    record_hovers: bool,
    mut context: ContextRef,
) -> Option<Shape> {
    if indices.len() != 1 {
        return None;
    }
    let IndexKind::Tensor(mask_expr) = &indices[0] else {
        return None;
    };
    let mask_shape = context
        .lookup_shape(mask_expr, record_hovers)
        .or_else(|| context.infer_shape(mask_expr, false))?;
    let mask_dtype = mask_shape.dtype.as_deref()?;
    if !matches!(SimpleDtype::from(mask_dtype), SimpleDtype::Bool) {
        return None;
    }
    if mask_shape.dims.is_empty() || mask_shape.dims.len() > base_shape.dims.len() {
        context.push_diagnostic_text(
            expr_text_range(mask_expr),
            DiagnosticSeverity::ERROR,
            format!(
                "Boolean mask shape {} is incompatible with indexed tensor shape {}",
                Shape {
                    dtype: mask_shape.dtype.clone(),
                    dims: mask_shape.dims.clone(),
                }
                .render(),
                base_shape.render()
            ),
        );
        return None;
    }
    if !mask_matches_prefix(&mask_shape.dims, &base_shape.dims) {
        context.push_diagnostic_text(
            expr_text_range(mask_expr),
            DiagnosticSeverity::ERROR,
            format!(
                "Boolean mask shape {} must match the leading dimensions of indexed tensor shape {}",
                Shape {
                    dtype: mask_shape.dtype.clone(),
                    dims: mask_shape.dims.clone(),
                }
                .render(),
                base_shape.render()
            ),
        );
        return None;
    }

    let mut dims = vec![bounded_product_dim(&mask_shape.dims)];
    dims.extend_from_slice(&base_shape.dims[mask_shape.dims.len()..]);
    Some(Shape {
        dtype: base_shape.dtype.clone(),
        dims,
    })
}

fn mask_matches_prefix(mask_dims: &[String], base_dims: &[String]) -> bool {
    if mask_dims.len() > base_dims.len() {
        return false;
    }
    mask_dims
        .iter()
        .zip(base_dims.iter())
        .all(|(mask, base)| dims_compatible(mask, base))
}

fn dims_compatible(left: &str, right: &str) -> bool {
    match (left.parse::<i64>().ok(), right.parse::<i64>().ok()) {
        (Some(l), Some(r)) => l == r,
        _ => left == right,
    }
}

fn bounded_product_dim(dims: &[String]) -> String {
    let product = dims.iter().try_fold(1i64, |acc, dim| {
        dim.parse::<i64>().ok().map(|value| acc * value)
    });
    match (dims.len(), product) {
        (_, Some(value)) => format!("0:{value}"),
        (0, _) => "0:0".to_string(),
        (1, _) => format!("0:{}", dims[0]),
        _ => format!("0:({})", dims.join("*")),
    }
}

fn is_advanced(kind: &IndexKind<'_>) -> bool {
    matches!(kind, IndexKind::Bool(_) | IndexKind::Tensor(_))
}

fn consumes_axis(kind: &IndexKind<'_>) -> bool {
    matches!(
        kind,
        IndexKind::Int { .. } | IndexKind::Slice { .. } | IndexKind::Keep | IndexKind::Tensor(_)
    )
}

fn parse_index_kind<'a>(expr: &'a Expr, source: &'a str) -> IndexKind<'a> {
    match expr {
        Expr::Constant(c) => match &c.value {
            ast::Constant::None => IndexKind::NewAxis,
            ast::Constant::Ellipsis => IndexKind::Ellipsis,
            ast::Constant::Bool(b) => IndexKind::Bool(*b),
            ast::Constant::Int(_) => {
                expr_to_int(expr, None).map_or(IndexKind::Tensor(expr), |value| IndexKind::Int {
                    value,
                    range: expr_text_range(expr),
                })
            }
            _ => IndexKind::Tensor(expr),
        },
        Expr::Name(n) if n.id.as_str() == "Ellipsis" => IndexKind::Ellipsis,
        Expr::Slice(s) => IndexKind::Slice {
            start: s.lower.as_deref().map(|e| bound_token(e, source)),
            stop: s.upper.as_deref().map(|e| bound_token(e, source)),
            step: s.step.as_deref().and_then(|x| expr_to_int(x, None)),
            range: expr_text_range(expr),
        },
        Expr::Tuple(_) => IndexKind::Tensor(expr),
        Expr::UnaryOp(_) => {
            expr_to_int(expr, None).map_or(IndexKind::Tensor(expr), |value| IndexKind::Int {
                value,
                range: expr_text_range(expr),
            })
        }
        _ => IndexKind::Tensor(expr),
    }
}

fn bound_token(expr: &Expr, source: &str) -> String {
    expr_to_symbolic_token(expr, SLICE_BOUND_TOKEN_OPTIONS)
        .map(|token| token.into_owned())
        .unwrap_or_else(|| {
            let range = expr_text_range(expr);
            let mut raw = source
                .get(range.start().to_usize()..range.end().to_usize())
                .unwrap_or("")
                .replace(' ', "");
            while raw.starts_with('(') && raw.ends_with(')') && raw.len() >= 2 {
                raw = raw[1..raw.len() - 1].to_string();
            }
            raw
        })
}

fn advanced_shape<'a>(
    kind: &IndexKind<'a>,
    record_hovers: bool,
    mut context: ContextRef,
) -> Option<Vec<String>> {
    match kind {
        IndexKind::Bool(b) => Some(vec![(if *b { "1" } else { "0" }).to_string()]),
        IndexKind::Tensor(expr) => {
            if let Some(shape) = context.infer_shape(expr, record_hovers) {
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

fn split_primary_offset(tok: &str) -> (String, Option<String>) {
    if let Some(idx) = tok
        .char_indices()
        .skip(1)
        .find(|&(_, c)| c == '+' || c == '-')
        .map(|(idx, _)| idx)
    {
        let (primary, offset) = tok.split_at(idx);
        (primary.to_string(), Some(offset.to_string()))
    } else {
        (tok.to_string(), None)
    }
}

fn apply_slice(dim: &str, step: Option<i64>) -> String {
    match step {
        Some(s) if s > 1 => format!("{dim}/{s}"),
        Some(s) if s < -1 => format!("{dim}/{}", s.abs()),
        _ => dim.to_string(),
    }
}

fn normalize_slice_bound(value: i64, len: i64) -> i64 {
    let mut idx = if value < 0 { len + value } else { value };
    if idx < 0 {
        idx = 0;
    } else if idx > len {
        idx = len;
    }
    idx
}

fn slice_len_from_bounds(
    len: i64,
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
) -> Option<i64> {
    let start_idx = start.map(|v| normalize_slice_bound(v, len)).unwrap_or(0);
    let stop_idx = stop.map(|v| normalize_slice_bound(v, len)).unwrap_or(len);
    let mut span = stop_idx - start_idx;
    if span < 0 {
        span = 0;
    }
    match step {
        Some(step) if step > 1 => Some((span + step - 1) / step),
        Some(1) => Some(span),
        Some(_) => None,
        None => Some(span),
    }
}
