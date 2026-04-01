use crate::torch_nn::get_call_arg;
use crate::{ClassMap, FuncMap, HoverInfo, Imports, ModuleCache, Shape, VarState, get_arg};
use lsp_types::{Diagnostic, Range};
use rustpython_parser::ast::{Constant, Expr, ExprCall, ExprConstant, Identifier};
use rustpython_parser::text_size::TextRange;
use std::collections::HashMap;
use std::path::Path;

use super::push_error_diagnostic;
use crate::{dims_equal, infer_expr_shape, lookup_shape};

/// Information required to know the expected inputs and output of a [loss](https://docs.pytorch.org/docs/stable/nn.html#loss-functions.
#[derive(Clone, Debug)]
pub struct LossParams {
    pub reduction: Reduction,
    pub expected: LossExpectedInputs,
}

/// Type of inputs expected by a loss.
#[derive(Clone, Debug, Copy)]
pub enum LossExpectedInputs {
    /// Prediction and target should be of equal shape
    Equal {
        min: usize,
        max: usize,
        inputs: usize,
    },
    /// (N?, C) and (N?) or (N, C, d1, d2, dK) and (N, d1, d2, ..., dk)
    /// See [docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html)
    NllLike,
    /// `torch.nn.CTCLoss`
    /// See [docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html)
    Ctc,
    /// `torch.nn.CosineEmbeddingLoss`: three inputs: (N?, C), (N?, C), (N?,)
    CosineEmbedding,
    /// Single input, of shape (N?, D)
    Triplet,
    /// Single input, of shape (N?, ...)
    TripletDistance,
}

impl LossExpectedInputs {
    pub const fn const_default() -> Self {
        Self::Equal {
            min: 0,
            max: usize::MAX,
            inputs: 2,
        }
    }
    pub const fn equal(min: usize, max: usize, inputs: usize) -> Self {
        Self::Equal { min, max, inputs }
    }
}

/// Reduction argument for a [loss](https://docs.pytorch.org/docs/stable/nn.html#loss-functions).
#[derive(Clone, Debug)]
pub enum Reduction {
    /// "none"
    None,
    /// any other ("mean", "sum")
    Some,
}

impl Reduction {
    pub fn loss_from_args(call: &ExprCall, reduction_arg_pos: usize) -> Reduction {
        if let Some(Expr::Constant(ExprConstant {
            value: Constant::Str(s),
            ..
        })) = get_call_arg(call, "reduction", reduction_arg_pos)
            && s == "none"
        {
            Reduction::None
        } else {
            // default is "mean" (Some)
            Reduction::Some
        }
    }
}

/// Infer a call to a loss (e. g., `torch.nn.MSELoss` or functional), either a NoOp
/// or a reduction to a number.
pub fn infer_loss(
    base_shape: Shape,
    reduction: &Reduction,
    expected: &LossExpectedInputs,
    call: &ExprCall<TextRange>,
    vars: &HashMap<Identifier, VarState>,
    func_map: &FuncMap,
    imports: &Imports,
    class_map: &ClassMap,
    call_stack: &mut Vec<Identifier>,
    diagnostics: &mut Vec<Diagnostic>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    source: &str,
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> Option<Shape> {
    let mut get_arg_shape = |pos: usize, key: &'static str, diagnostics: &mut Vec<Diagnostic>| {
        let expr = get_arg(call, key, pos)?;
        lookup_shape(expr, vars, hover_entries, true, source).or_else(|| {
            infer_expr_shape(
                expr,
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
        })
    };
    let base_len = base_shape.dims.len();
    let mut reported_diagnostics = false;
    let unreduced = match expected {
        LossExpectedInputs::Equal { min, max, inputs } => {
            if &base_len < min || &base_len > max {
                push_error_diagnostic(
                    diagnostics,
                    call.range,
                    source,
                    format!(
                        "Invalid number of dims {} for input ∉ [{min}, {max}]",
                        base_shape.render()
                    ),
                );
                reported_diagnostics = true;
            }

            for pos in 1..(*inputs) {
                let key = match pos {
                    1 => "target",
                    2 => "input2",
                    // this should never happen
                    _ => break,
                };

                if let Some(target) = get_arg_shape(inputs - pos, key, diagnostics) {
                    if target != base_shape {
                        push_error_diagnostic(
                            diagnostics,
                            call.range,
                            source,
                            format!(
                                "Input and {key} must have the same shape, found {} vs. {}",
                                base_shape.render(),
                                target.render()
                            ),
                        );
                        reported_diagnostics = true;
                    }
                } else {
                    push_error_diagnostic(
                        diagnostics,
                        call.range,
                        source,
                        format!("Missing `{key}` input"),
                    );
                    reported_diagnostics = true;
                }
            }
            base_shape.dims.clone()
        }
        LossExpectedInputs::NllLike => {
            let unreduced = match base_shape.dims.len() {
                0 => None,
                1 => Some(Vec::new()),
                _ => {
                    let mut dims = vec![base_shape.dims[0].clone()];
                    dims.extend(base_shape.dims[2..].iter().cloned());
                    Some(dims)
                }
            };
            if unreduced.is_none() {
                push_error_diagnostic(
                    diagnostics,
                    call.range,
                    source,
                    format!(
                        "Invalid number of dims {} for NLL-like loss input",
                        base_shape.render()
                    ),
                );
                reported_diagnostics = true;
            }

            if let Some(target) = get_arg_shape(1, "target", diagnostics) {
                if let Some(expected_target) = unreduced.as_ref()
                    && target != expected_target.as_slice()
                {
                    push_error_diagnostic(
                        diagnostics,
                        call.range,
                        source,
                        format!(
                            "NLL-like loss target must have shape [{}], found {}",
                            expected_target.join(" "),
                            target.render()
                        ),
                    );
                    reported_diagnostics = true;
                }
            } else {
                push_error_diagnostic(
                    diagnostics,
                    call.range,
                    source,
                    "Missing `target` input".to_string(),
                );
                reported_diagnostics = true;
            }

            unreduced.unwrap_or_default()
        }
        LossExpectedInputs::Ctc => {
            let is_batched = match base_len {
                2 => false,
                3 => true,
                _ => {
                    push_error_diagnostic(
                        diagnostics,
                        call.range,
                        source,
                        format!(
                            "CTCLoss expects input of rank 2 or 3, found {}",
                            base_shape.render()
                        ),
                    );
                    reported_diagnostics = true;
                    false
                }
            };

            let length_dims = if is_batched {
                vec![base_shape.dims[1].clone()]
            } else {
                Vec::new()
            };
            let (target, input_lengths, target_lengths) = {
                let mut require_arg_shape = |pos: usize, key: &'static str| {
                    let shape = get_arg_shape(pos, key, diagnostics);
                    if shape.is_none() {
                        push_error_diagnostic(
                            diagnostics,
                            call.range,
                            source,
                            format!("Missing `{key}` input"),
                        );
                        reported_diagnostics = true;
                    }
                    shape
                };
                (
                    require_arg_shape(1, "target"),
                    require_arg_shape(2, "input_lengths"),
                    require_arg_shape(3, "target_lengths"),
                )
            };

            if let Some(target) = target {
                let valid_target_rank = if is_batched {
                    matches!(target.dims.len(), 1 | 2)
                } else {
                    target.dims.len() == 1
                };
                if !valid_target_rank {
                    push_error_diagnostic(
                        diagnostics,
                        call.range,
                        source,
                        format!(
                            "CTCLoss target must be 1D{} , found {}",
                            if is_batched { " or 2D" } else { "" },
                            target.render()
                        ),
                    );
                    reported_diagnostics = true;
                } else if is_batched
                    && target.dims.len() == 2
                    && !dims_equal(&target.dims[..1], &base_shape.dims[1..2])
                {
                    push_error_diagnostic(
                        diagnostics,
                        call.range,
                        source,
                        format!(
                            "CTCLoss padded target batch dim must match input batch dim, found {} vs. [{}]",
                            target.render(),
                            base_shape.dims[1]
                        ),
                    );
                    reported_diagnostics = true;
                }
            }

            for (shape, key) in [
                (&input_lengths, "input_lengths"),
                (&target_lengths, "target_lengths"),
            ] {
                if let Some(shape) = shape.as_ref()
                    && shape != &length_dims.as_slice()
                {
                    push_error_diagnostic(
                        diagnostics,
                        call.range,
                        source,
                        format!(
                            "CTCLoss {key} must have shape [{}], found {}",
                            length_dims.join(" "),
                            shape.render()
                        ),
                    );
                    reported_diagnostics = true;
                }
            }

            if is_batched { length_dims } else { Vec::new() }
        }
        LossExpectedInputs::CosineEmbedding => {
            if !matches!(base_len, 1 | 2) {
                push_error_diagnostic(
                    diagnostics,
                    call.range,
                    source,
                    format!(
                        "CosineEmbeddingLoss expects input rank 1 or 2, found {}",
                        base_shape.render()
                    ),
                );
                reported_diagnostics = true;
            }

            let input2 = get_arg_shape(1, "input2", diagnostics);
            let target = get_arg_shape(2, "target", diagnostics);

            if let Some(input2) = input2 {
                if input2 != base_shape {
                    push_error_diagnostic(
                        diagnostics,
                        call.range,
                        source,
                        format!(
                            "CosineEmbeddingLoss inputs must have the same shape, found {} vs. {}",
                            base_shape.render(),
                            input2.render()
                        ),
                    );
                    reported_diagnostics = true;
                }
            } else {
                push_error_diagnostic(
                    diagnostics,
                    call.range,
                    source,
                    "Missing `input2` input".to_string(),
                );
                reported_diagnostics = true;
            }

            let unreduced = if base_len == 2 {
                vec![base_shape.dims[0].clone()]
            } else {
                Vec::new()
            };
            if let Some(target) = target {
                if target != unreduced.as_slice() {
                    push_error_diagnostic(
                        diagnostics,
                        call.range,
                        source,
                        format!(
                            "CosineEmbeddingLoss target must have shape [{}], found {}",
                            unreduced.join(" "),
                            target.render()
                        ),
                    );
                    reported_diagnostics = true;
                }
            } else {
                push_error_diagnostic(
                    diagnostics,
                    call.range,
                    source,
                    "Missing `target` input".to_string(),
                );
                reported_diagnostics = true;
            }

            unreduced
        }
        LossExpectedInputs::Triplet | LossExpectedInputs::TripletDistance => {
            let (loss_name, unreduced, valid_rank, rank_message) = match expected {
                LossExpectedInputs::Triplet => (
                    "TripletMarginLoss",
                    if base_len == 2 {
                        vec![base_shape.dims[0].clone()]
                    } else {
                        Vec::new()
                    },
                    matches!(base_len, 1 | 2),
                    format!(
                        "TripletMarginLoss expects input rank 1 or 2, found {}",
                        base_shape.render()
                    ),
                ),
                LossExpectedInputs::TripletDistance => (
                    "TripletMarginWithDistanceLoss",
                    base_shape.dims.first().cloned().into_iter().collect(),
                    base_len > 0,
                    "TripletMarginWithDistanceLoss expects at least one input dimension"
                        .to_string(),
                ),
                _ => unreachable!(),
            };

            if !valid_rank {
                push_error_diagnostic(diagnostics, call.range, source, rank_message);
                reported_diagnostics = true;
            }

            {
                let mut check_pair_input = |pos: usize, key: &'static str| {
                    if let Some(arg) = get_arg_shape(pos, key, diagnostics) {
                        if arg != base_shape {
                            push_error_diagnostic(
                                diagnostics,
                                call.range,
                                source,
                                format!(
                                    "{loss_name} {key} must have shape {}, found {}",
                                    base_shape.render(),
                                    arg.render()
                                ),
                            );
                            reported_diagnostics = true;
                        }
                    } else {
                        push_error_diagnostic(
                            diagnostics,
                            call.range,
                            source,
                            format!("Missing `{key}` input"),
                        );
                        reported_diagnostics = true;
                    }
                };

                for (pos, key) in [(1usize, "positive"), (2usize, "negative")] {
                    check_pair_input(pos, key);
                }
            }

            unreduced
        }
    };

    if reported_diagnostics {
        return None;
    }

    match reduction {
        Reduction::None => Some(Shape {
            dtype: base_shape.dtype,
            dims: unreduced,
        }),
        Reduction::Some => Some(Shape {
            dtype: base_shape.dtype,
            dims: Vec::new(),
        }),
    }
}
