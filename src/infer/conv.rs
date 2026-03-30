//! Functions related to convolutions, modules and functional.
use crate::expr_tokens::{CONV_PARAM_TOKEN_OPTIONS, expr_to_symbolic_token};
use crate::torch_nn::ConvParams;
use crate::{Shape, get_arg};
use lsp_types::{Diagnostic, DiagnosticSeverity};
use rustpython_parser::ast::{Expr, ExprCall};
use rustpython_parser::text_size::TextRange;

use super::{concrete_dim_mismatch, expand_conv_params};
use crate::text_range_to_lsp;

pub fn infer_conv_module(
    base: Shape,
    conv_dim: usize,
    in_channels: Option<&str>,
    out_channels: Option<&str>,
    kernel_size: &[String],
    stride: &[String],
    padding: &[String],
    dilation: &[String],
    _groups: Option<&str>,
    range: TextRange,
    diagnostics: &mut Vec<Diagnostic>,
    source: &str,
) -> Option<Shape> {
    let kernel_size = expand_conv_params(kernel_size, conv_dim, "K");
    let stride = expand_conv_params(stride, conv_dim, "1");
    let padding = expand_conv_params(padding, conv_dim, "0");
    let dilation = expand_conv_params(dilation, conv_dim, "1");
    infer_module_conv(
        base,
        conv_dim,
        in_channels,
        out_channels,
        "conv",
        "Conv",
        range,
        diagnostics,
        source,
        |dim, spatial_i| {
            infer_conv_dim(
                &dim,
                &kernel_size[spatial_i],
                &stride[spatial_i],
                &padding[spatial_i],
                &dilation[spatial_i],
            )
        },
    )
}

pub fn infer_conv_transpose(
    base: Shape,
    params: &ConvParams,
    range: TextRange,
    diagnostics: &mut Vec<Diagnostic>,
    source: &str,
) -> Option<Shape> {
    let kernel_size = expand_conv_params(&params.kernel_size, params.dims, "K");
    let stride = expand_conv_params(&params.stride, params.dims, "1");
    let padding = expand_conv_params(&params.padding, params.dims, "0");
    let dilation = expand_conv_params(&params.dilation, params.dims, "1");
    let output_padding = expand_conv_params(&params.output_padding, params.dims, "0");
    infer_module_conv(
        base,
        params.dims,
        params.in_channels.as_deref(),
        params.out_channels.as_deref(),
        "convtranspose",
        "ConvTranspose",
        range,
        diagnostics,
        source,
        |dim, spatial_i| {
            infer_conv_transpose_dim(
                &dim,
                &kernel_size[spatial_i],
                &stride[spatial_i],
                &padding[spatial_i],
                &dilation[spatial_i],
                &output_padding[spatial_i],
            )
        },
    )
}

pub fn infer_conv(
    base: Shape,
    kernel: Vec<String>,
    call: &ExprCall,
    conv_dim: usize,
    diagnostics: &mut Vec<Diagnostic>,
    source: &str,
) -> Option<Shape> {
    let stride = get_arg(call, "stride", 3)
        .and_then(expr_to_tuple)
        .unwrap_or_default();
    let padding = get_arg(call, "padding", 4)
        .and_then(expr_to_tuple)
        .unwrap_or_default();
    let dilation = get_arg(call, "dilation", 5)
        .and_then(expr_to_tuple)
        .unwrap_or_default();
    infer_conv_with_params(
        base,
        kernel,
        &stride,
        &padding,
        &dilation,
        conv_dim,
        call.range,
        diagnostics,
        source,
    )
}

fn infer_conv_with_params(
    base: Shape,
    kernel: Vec<String>,
    stride: &[String],
    padding: &[String],
    dilation: &[String],
    conv_dim: usize,
    range: TextRange,
    diagnostics: &mut Vec<Diagnostic>,
    source: &str,
) -> Option<Shape> {
    let n_dims = base.dims.len();
    let n_k_dims = kernel.len();
    ensure_valid_conv_rank(
        n_dims,
        conv_dim,
        "Input tensor",
        "conv",
        range,
        diagnostics,
        source,
    )?;
    ensure_valid_conv_rank(
        n_k_dims,
        conv_dim,
        "Kernel",
        "conv",
        range,
        diagnostics,
        source,
    )?;
    if n_k_dims > n_dims + 1 {
        diagnostics.push(Diagnostic {
            range: text_range_to_lsp(range, source),
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

    let kernel_size = expand_conv_params(&kernel[2..], conv_dim, "K");
    let stride = expand_conv_params(stride, conv_dim, "1");
    let padding = expand_conv_params(padding, conv_dim, "0");
    let dilation = expand_conv_params(dilation, conv_dim, "1");
    Some(infer_conv_output(
        base,
        conv_dim,
        kernel[0].clone(),
        |dim, spatial_i| {
            infer_conv_dim(
                &dim,
                &kernel_size[spatial_i],
                &stride[spatial_i],
                &padding[spatial_i],
                &dilation[spatial_i],
            )
        },
    ))
}

fn infer_module_conv(
    base: Shape,
    conv_dim: usize,
    in_channels: Option<&str>,
    out_channels: Option<&str>,
    op_name: &str,
    module_name: &str,
    range: TextRange,
    diagnostics: &mut Vec<Diagnostic>,
    source: &str,
    map_spatial_dim: impl FnMut(String, usize) -> String,
) -> Option<Shape> {
    validate_module_input(
        &base,
        conv_dim,
        in_channels,
        op_name,
        module_name,
        range,
        diagnostics,
        source,
    )?;
    Some(infer_conv_output(
        base,
        conv_dim,
        out_channels
            .map(str::to_string)
            .unwrap_or_else(|| "OutChannels".to_string()),
        map_spatial_dim,
    ))
}

fn validate_module_input(
    base: &Shape,
    conv_dim: usize,
    in_channels: Option<&str>,
    op_name: &str,
    module_name: &str,
    range: TextRange,
    diagnostics: &mut Vec<Diagnostic>,
    source: &str,
) -> Option<()> {
    ensure_valid_conv_rank(
        base.dims.len(),
        conv_dim,
        "Input tensor",
        op_name,
        range,
        diagnostics,
        source,
    )?;
    let input_channel = conv_input_channel(base, conv_dim);
    if let (Some(current), Some(expected)) = (input_channel.as_deref(), in_channels)
        && concrete_dim_mismatch(current, expected)
    {
        diagnostics.push(Diagnostic {
            range: text_range_to_lsp(range, source),
            severity: Some(DiagnosticSeverity::ERROR),
            code: None,
            code_description: None,
            source: Some("shapels".into()),
            message: format!(
                "{module_name}{conv_dim}d channel mismatch: expected {expected}, got {current}"
            ),
            related_information: None,
            tags: None,
            data: None,
        });
    }
    Some(())
}

fn infer_conv_output(
    base: Shape,
    conv_dim: usize,
    out_channels: String,
    map_spatial_dim: impl FnMut(String, usize) -> String,
) -> Shape {
    map_conv_output_dims(base, conv_dim, out_channels, map_spatial_dim)
}

fn infer_conv_dim(
    dim: &str,
    kernel_size: &str,
    stride: &str,
    padding: &str,
    dilation: &str,
) -> String {
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
            let offset = 2 * p - d * (k - 1) - 1;
            if s == 1 && offset == -1 {
                dim.to_string()
            } else {
                format!("({dim}+{offset})/{s}+1")
            }
        }
        (Ok(n), Ok(k), Err(_), Ok(p), Ok(d)) => {
            format!("{}/{stride}+1", n + 2 * p - d * (k - 1) - 1)
        }
        (Err(_), Ok(k), Err(_), Ok(p), Ok(d)) => {
            let num = 2 * p - d * (k - 1) - 1;
            format!("({dim}+{num})/{stride}+1")
        }
        (_, Ok(k), Err(_), Err(_), Ok(d)) => {
            format!("({dim}+2*{padding}-{})/{stride}+1", d * (k - 1) - 1)
        }
        (Ok(n), Err(_), _, Err(_), Err(_)) => {
            format!(
                "({}+2*{padding}-{dilation}*({kernel_size}-1))/{stride}+1",
                n - 1
            )
        }
        (Ok(n), Ok(k), _, Err(_), Err(_)) => {
            format!("({}+2*{padding}-{dilation}*{})/{stride}+1", n - 1, k - 1)
        }
        _ => format!("(({dim}+2*{padding}-{dilation}*({kernel_size}-1)-1)/{stride})+1"),
    }
}

fn infer_conv_transpose_dim(
    dim: &str,
    kernel_size: &str,
    stride: &str,
    padding: &str,
    dilation: &str,
    output_padding: &str,
) -> String {
    match (
        dim.parse::<i64>(),
        kernel_size.parse::<i64>(),
        stride.parse::<i64>(),
        padding.parse::<i64>(),
        dilation.parse::<i64>(),
        output_padding.parse::<i64>(),
    ) {
        (Ok(n), Ok(k), Ok(s), Ok(p), Ok(d), Ok(op)) => {
            (((n - 1) * s) - 2 * p + d * (k - 1) + op + 1).to_string()
        }
        (Err(_), Ok(k), Ok(s), Ok(p), Ok(d), Ok(op)) => {
            let offset = -s - 2 * p + d * (k - 1) + op + 1;
            linear_dim_expr(dim, s, offset)
        }
        _ => {
            let mut expr = String::new();
            let mut constant = 1i64;

            match (dim.parse::<i64>(), stride.parse::<i64>()) {
                (Ok(n), Ok(s)) => constant += (n - 1) * s,
                (Ok(n), Err(_)) => push_scaled_expr_term(&mut expr, n - 1, stride),
                (Err(_), Ok(s)) => {
                    push_scaled_expr_term(&mut expr, s, dim);
                    constant -= s;
                }
                (Err(_), Err(_)) => push_expr_term(
                    &mut expr,
                    1,
                    &format!(
                        "({}-1)*{}",
                        maybe_parenthesize(dim),
                        maybe_parenthesize(stride)
                    ),
                ),
            }

            match padding.parse::<i64>() {
                Ok(p) => constant -= 2 * p,
                Err(_) => push_scaled_expr_term(&mut expr, -2, padding),
            }

            match (kernel_size.parse::<i64>(), dilation.parse::<i64>()) {
                (Ok(k), Ok(d)) => constant += d * (k - 1),
                (Ok(k), Err(_)) => push_scaled_expr_term(&mut expr, k - 1, dilation),
                (Err(_), Ok(d)) => {
                    if d == 1 {
                        push_expr_term(
                            &mut expr,
                            1,
                            &format!("{}-1", maybe_parenthesize(kernel_size)),
                        );
                    } else {
                        push_expr_term(&mut expr, 1, &format!("{d}*({kernel_size}-1)"));
                    }
                }
                (Err(_), Err(_)) => push_expr_term(
                    &mut expr,
                    1,
                    &format!(
                        "{}*({}-1)",
                        maybe_parenthesize(dilation),
                        maybe_parenthesize(kernel_size)
                    ),
                ),
            }

            match output_padding.parse::<i64>() {
                Ok(op) => constant += op,
                Err(_) => push_scaled_expr_term(&mut expr, 1, output_padding),
            }

            finish_symbolic_sum(expr, constant)
        }
    }
}

fn has_valid_conv_rank(rank: usize, conv_dim: usize) -> bool {
    matches!(rank.checked_sub(conv_dim), Some(1 | 2))
}

fn ensure_valid_conv_rank(
    rank: usize,
    conv_dim: usize,
    tensor_msg: &str,
    op_name: &str,
    range: TextRange,
    diagnostics: &mut Vec<Diagnostic>,
    source: &str,
) -> Option<()> {
    if has_valid_conv_rank(rank, conv_dim) {
        Some(())
    } else {
        diagnostics.push(Diagnostic {
            range: text_range_to_lsp(range, source),
            severity: Some(DiagnosticSeverity::ERROR),
            code: None,
            code_description: None,
            source: Some("shapels".into()),
            message: format!(
                "{tensor_msg} has incorrect dims for {op_name}{conv_dim}d: {} ∉ {{{},{}}}",
                rank,
                conv_dim + 1,
                conv_dim + 2
            ),
            related_information: None,
            tags: None,
            data: None,
        });
        None
    }
}

fn conv_input_channel(base: &Shape, conv_dim: usize) -> Option<String> {
    base.dims
        .len()
        .checked_sub(conv_dim)
        .and_then(|pos| pos.checked_sub(1))
        .and_then(|idx| base.dims.get(idx))
        .cloned()
}

fn map_conv_output_dims(
    base: Shape,
    conv_dim: usize,
    out_channels: String,
    mut map_spatial_dim: impl FnMut(String, usize) -> String,
) -> Shape {
    let spatial_offset = base.dims.len() - conv_dim;
    Shape {
        dtype: base.dtype,
        dims: base
            .dims
            .into_iter()
            .enumerate()
            .map(|(idx, dim)| {
                if idx == spatial_offset - 1 {
                    out_channels.clone()
                } else if idx >= spatial_offset {
                    map_spatial_dim(dim, idx - spatial_offset)
                } else {
                    dim
                }
            })
            .collect(),
    }
}

fn linear_dim_expr(dim: &str, coeff: i64, offset: i64) -> String {
    let mut expr = String::new();
    push_scaled_expr_term(&mut expr, coeff, dim);
    finish_symbolic_sum(expr, offset)
}

fn format_scaled_token(coeff: i64, token: &str) -> String {
    if coeff == 0 {
        "0".to_string()
    } else if coeff == 1 {
        token.to_string()
    } else {
        format!("{coeff}*{}", maybe_parenthesize(token))
    }
}

fn push_scaled_expr_term(expr: &mut String, coeff: i64, token: &str) {
    if coeff == 0 {
        return;
    }
    let term = format_scaled_token(coeff.abs(), token);
    push_expr_term(expr, coeff.signum(), &term);
}

fn finish_symbolic_sum(mut expr: String, constant: i64) -> String {
    if constant != 0 || expr.is_empty() {
        let constant_abs = constant.abs().to_string();
        push_expr_term(&mut expr, constant.signum(), &constant_abs);
    }
    expr
}

fn push_expr_term(expr: &mut String, sign: i64, term: &str) {
    if sign == 0 || term == "0" {
        return;
    }
    if expr.is_empty() {
        if sign < 0 {
            expr.push('-');
        }
        expr.push_str(term);
        return;
    }
    expr.push(if sign < 0 { '-' } else { '+' });
    expr.push_str(term);
}

fn maybe_parenthesize(token: &str) -> String {
    if token
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
    {
        token.to_string()
    } else {
        format!("({token})")
    }
}

fn expr_to_tuple(expr: &Expr) -> Option<Vec<String>> {
    let elts = match expr {
        Expr::Tuple(tup) => &tup.elts,
        Expr::List(list) => &list.elts,
        other => {
            return expr_to_symbolic_token(other, CONV_PARAM_TOKEN_OPTIONS)
                .map(|token| vec![token.into_owned()]);
        }
    };

    elts.iter()
        .map(|expr| {
            expr_to_symbolic_token(expr, CONV_PARAM_TOKEN_OPTIONS).map(|token| token.into_owned())
        })
        .collect()
}
