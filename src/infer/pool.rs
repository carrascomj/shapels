use crate::Shape;
use lsp_types::Diagnostic;
use rustpython_parser::text_size::TextRange;

use super::expand_conv_params;
use crate::infer::push_error_diagnostic;

/// Pooling module families from [`torch.nn`](https://docs.pytorch.org/docs/stable/nn.html#pooling-layers).
#[derive(Debug, Clone, Copy)]
pub enum PoolKind {
    Max,
    Avg,
    Lp,
    FractionalMax,
    AdaptiveMax,
    AdaptiveAvg,
}

impl PoolKind {
    fn module_name(&self, pool_dim: usize) -> String {
        let prefix = match self {
            PoolKind::Max => "MaxPool",
            PoolKind::Avg => "AvgPool",
            PoolKind::Lp => "LPPool",
            PoolKind::FractionalMax => "FractionalMaxPool",
            PoolKind::AdaptiveMax => "AdaptiveMaxPool",
            PoolKind::AdaptiveAvg => "AdaptiveAvgPool",
        };
        format!("{prefix}{pool_dim}d")
    }

    fn returns_indices(&self) -> bool {
        matches!(
            self,
            PoolKind::Max | PoolKind::FractionalMax | PoolKind::AdaptiveMax
        )
    }
}

pub fn infer_pool_module(
    base: Shape,
    pool_dim: usize,
    pool_kind: &PoolKind,
    kernel_size: &[String],
    stride: &[String],
    padding: &[String],
    dilation: &[String],
    output_size: &[Option<String>],
    output_ratio: &[String],
    ceil_mode: bool,
    return_indices: bool,
    range: TextRange,
    diagnostics: &mut Vec<Diagnostic>,
    source: &str,
) -> Option<Shape> {
    if return_indices && pool_kind.returns_indices() {
        return None;
    }

    if !has_valid_pool_rank(base.dims.len(), pool_dim) {
        push_error_diagnostic(
            diagnostics,
            range,
            source,
            format!(
                "Input tensor has incorrect dims for {}",
                pool_kind.module_name(pool_dim)
            ),
        );
        return None;
    }

    match pool_kind {
        PoolKind::AdaptiveMax | PoolKind::AdaptiveAvg => {
            let output_size = expand_required_optional_pool_params(
                output_size,
                pool_dim,
                range,
                diagnostics,
                source,
                &format!("{} requires output_size", pool_kind.module_name(pool_dim)),
            )?;
            Some(map_pool_spatial_dims(base, pool_dim, |dim, spatial_i| {
                output_size[spatial_i].clone().unwrap_or(dim)
            }))
        }
        PoolKind::FractionalMax => {
            let output_size = expand_optional_pool_params(output_size, pool_dim);
            let output_ratio = expand_conv_params(output_ratio, pool_dim, "0.5");
            if output_size.is_empty() && output_ratio.is_empty() {
                push_error_diagnostic(
                    diagnostics,
                    range,
                    source,
                    "FractionalMaxPool requires output_size or output_ratio".to_string(),
                );
                return None;
            }
            let use_output_size = !output_size.is_empty();
            Some(map_pool_spatial_dims(base, pool_dim, |dim, spatial_i| {
                if use_output_size {
                    output_size[spatial_i].clone().unwrap_or(dim)
                } else {
                    fractional_pool_dim(&dim, &output_ratio[spatial_i])
                }
            }))
        }
        PoolKind::Max | PoolKind::Avg | PoolKind::Lp => infer_window_pool_module(
            base,
            pool_dim,
            pool_kind,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        ),
    }
}

fn has_valid_pool_rank(rank: usize, pool_dim: usize) -> bool {
    matches!(rank.checked_sub(pool_dim), Some(1 | 2))
}

fn infer_window_pool_module(
    base: Shape,
    pool_dim: usize,
    pool_kind: &PoolKind,
    kernel_size: &[String],
    stride: &[String],
    padding: &[String],
    dilation: &[String],
    ceil_mode: bool,
) -> Option<Shape> {
    let kernel_size = expand_conv_params(kernel_size, pool_dim, "K");
    let stride = if stride.is_empty() {
        None
    } else {
        Some(expand_conv_params(stride, pool_dim, "1"))
    };
    let stride = stride.as_deref().unwrap_or(&kernel_size);
    let padding = match pool_kind {
        PoolKind::Lp => vec!["0".to_string(); pool_dim],
        _ => expand_conv_params(padding, pool_dim, "0"),
    };
    let dilation = match pool_kind {
        PoolKind::Max => expand_conv_params(dilation, pool_dim, "1"),
        _ => vec!["1".to_string(); pool_dim],
    };
    Some(map_pool_spatial_dims(base, pool_dim, |dim, spatial_i| {
        infer_window_pool_dim(
            &dim,
            &kernel_size[spatial_i],
            &stride[spatial_i],
            &padding[spatial_i],
            &dilation[spatial_i],
            ceil_mode,
        )
    }))
}

fn infer_window_pool_dim(
    dim: &str,
    kernel_size: &str,
    stride: &str,
    padding: &str,
    dilation: &str,
    ceil_mode: bool,
) -> String {
    match (
        dim.parse::<i32>(),
        kernel_size.parse::<i32>(),
        stride.parse::<i32>(),
        padding.parse::<i32>(),
        dilation.parse::<i32>(),
    ) {
        (Ok(n), Ok(k), Ok(s), Ok(p), Ok(d)) => {
            let numerator = n + 2 * p - d * (k - 1) - 1;
            let quotient = if ceil_mode {
                div_ceil_i64(numerator.into(), s.into()) as i32
            } else {
                numerator / s
            };
            (quotient + 1).to_string()
        }
        (Err(_), Ok(k), Ok(s), Ok(p), Ok(d)) if !ceil_mode => {
            let offset = 2 * p - d * (k - 1) - 1;
            if s == 1 && offset == -1 {
                dim.to_string()
            } else {
                format!("({dim}+{offset})/{s}+1")
            }
        }
        (Ok(n), Ok(k), Err(_), Ok(p), Ok(d)) if !ceil_mode => {
            format!("{}/{stride}+1", n + 2 * p - d * (k - 1) - 1)
        }
        (Err(_), Ok(k), Err(_), Ok(p), Ok(d)) if !ceil_mode => {
            let num = 2 * p - d * (k - 1) - 1;
            format!("({dim}+{num})/{stride}+1")
        }
        (_, Ok(k), Err(_), Err(_), Ok(d)) if !ceil_mode => {
            format!("({dim}+2*{padding}-{})/{stride}+1", d * (k - 1) - 1)
        }
        (Ok(n), Err(_), _, Err(_), Err(_)) if !ceil_mode => {
            format!(
                "({}+2*{padding}-{dilation}*({kernel_size}-1))/{stride}+1",
                n - 1
            )
        }
        (Ok(n), Ok(k), _, Err(_), Err(_)) if !ceil_mode => {
            format!("({}+2*{padding}-{dilation}*{})/{stride}+1", n - 1, k - 1)
        }
        _ if ceil_mode => {
            let numerator = pool_numerator_expr(dim, kernel_size, padding, dilation);
            format!("ceil(({numerator})/{stride})+1")
        }
        _ => {
            format!("(({dim}+2*{padding}-{dilation}*({kernel_size}-1)-1)/{stride})+1")
        }
    }
}

fn expand_optional_pool_params(values: &[Option<String>], pool_dim: usize) -> Vec<Option<String>> {
    match values.len() {
        0 => Vec::new(),
        1 => vec![values[0].clone(); pool_dim],
        len if len >= pool_dim => values[..pool_dim].to_vec(),
        _ => {
            let mut expanded = values.to_vec();
            while expanded.len() < pool_dim {
                expanded.push(values[0].clone());
            }
            expanded
        }
    }
}

fn expand_required_optional_pool_params(
    values: &[Option<String>],
    pool_dim: usize,
    range: TextRange,
    diagnostics: &mut Vec<Diagnostic>,
    source: &str,
    message: &str,
) -> Option<Vec<Option<String>>> {
    let expanded = expand_optional_pool_params(values, pool_dim);
    if expanded.is_empty() {
        push_error_diagnostic(diagnostics, range, source, message.to_string());
        None
    } else {
        Some(expanded)
    }
}

fn map_pool_spatial_dims(
    base: Shape,
    pool_dim: usize,
    mut map_spatial_dim: impl FnMut(String, usize) -> String,
) -> Shape {
    let spatial_offset = base.dims.len() - pool_dim;
    Shape {
        dtype: base.dtype,
        dims: base
            .dims
            .into_iter()
            .enumerate()
            .map(|(idx, dim)| {
                if idx < spatial_offset {
                    dim
                } else {
                    map_spatial_dim(dim, idx - spatial_offset)
                }
            })
            .collect(),
    }
}

fn fractional_pool_dim(dim: &str, ratio: &str) -> String {
    match (dim.parse::<i64>(), ratio.parse::<f64>()) {
        (Ok(n), Ok(r)) => ((n as f64 * r).floor() as i64).to_string(),
        _ => format!("floor({dim}*{ratio})"),
    }
}

fn pool_numerator_expr(dim: &str, kernel_size: &str, padding: &str, dilation: &str) -> String {
    match (
        kernel_size.parse::<i64>(),
        padding.parse::<i64>(),
        dilation.parse::<i64>(),
    ) {
        (Ok(k), Ok(p), Ok(d)) => append_signed_offset(dim, 2 * p - d * (k - 1) - 1),
        _ => format!("{dim}+2*{padding}-{dilation}*({kernel_size}-1)-1"),
    }
}

fn append_signed_offset(dim: &str, offset: i64) -> String {
    if offset == 0 {
        dim.to_string()
    } else if offset > 0 {
        format!("{dim}+{offset}")
    } else {
        format!("{dim}{offset}")
    }
}

fn div_ceil_i64(num: i64, den: i64) -> i64 {
    let div = num / den;
    let rem = num % den;
    if rem == 0 { div } else { div + 1 }
}
