//! Functions related to convolutions, modules and functional.
use crate::expr_tokens::{CONV_PARAM_TOKEN_OPTIONS, expr_to_symbolic_token};
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
    let input_channel = base
        .dims
        .len()
        .checked_sub(conv_dim)
        .and_then(|pos| pos.checked_sub(1))
        .and_then(|idx| base.dims.get(idx))
        .cloned();
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
                "Conv{conv_dim}d channel mismatch: expected {expected}, got {current}"
            ),
            related_information: None,
            tags: None,
            data: None,
        });
    }

    let mut kernel = vec![
        out_channels
            .map(str::to_string)
            .unwrap_or_else(|| "OutChannels".to_string()),
        in_channels
            .map(str::to_string)
            .or(input_channel)
            .unwrap_or_else(|| "InChannels".to_string()),
    ];
    kernel.extend(expand_conv_params(kernel_size, conv_dim, "K"));

    infer_conv_with_params(
        base,
        kernel,
        stride,
        padding,
        dilation,
        conv_dim,
        range,
        diagnostics,
        source,
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
    let input_extra = n_dims.checked_sub(conv_dim);
    let kernel_extra = n_k_dims.checked_sub(conv_dim);
    let incorrect_input = !matches!(input_extra, Some(1 | 2));
    let incorrect_kernel = !matches!(kernel_extra, Some(1 | 2));
    if incorrect_input || incorrect_kernel {
        let (incorrect_dims, tensor_msg) = if !incorrect_input {
            (n_dims, "Input tensor")
        } else {
            (n_k_dims, "Kernel")
        };
        diagnostics.push(Diagnostic {
            range: text_range_to_lsp(range, source),
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
    } else if n_k_dims > n_dims + 1 {
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
    let out_channels = &kernel[0];
    let stride = expand_conv_params(stride, conv_dim, "1");
    let padding = expand_conv_params(padding, conv_dim, "0");
    let dilation = expand_conv_params(dilation, conv_dim, "1");
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
                            let offset = 2 * p - d * (k - 1) - 1;
                            if s == 1 && offset == -1 {
                                dim.clone()
                            } else {
                                format!("({dim}+{offset})/{s}+1")
                            }
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
