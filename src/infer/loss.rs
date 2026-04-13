use crate::context::ContextRef;
use crate::torch_nn::get_call_arg;
use crate::{Shape, dims_equal, get_arg};
use lsp_types::DiagnosticSeverity;
use rustpython_parser::ast::{Constant, Expr, ExprCall, ExprConstant};
use rustpython_parser::text_size::TextRange;

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
            Reduction::Some
        }
    }
}

fn get_arg_shape(
    call: &ExprCall<TextRange>,
    pos: usize,
    key: &'static str,
    context: &mut ContextRef,
) -> Option<Shape> {
    let expr = get_arg(call, key, pos)?;
    context
        .lookup_shape(expr, true)
        .or_else(|| context.infer_shape(expr, false))
}

fn push_error(context: &mut ContextRef, range: TextRange, msg: String) {
    context.push_diagnostic_text(range, DiagnosticSeverity::ERROR, msg);
}

/// Infer a call to a loss (e. g., `torch.nn.MSELoss` or functional), either a NoOp
/// or a reduction to a number.
pub fn infer_loss(
    base_shape: Shape,
    reduction: &Reduction,
    expected: &LossExpectedInputs,
    call: &ExprCall<TextRange>,
    mut context: ContextRef,
) -> Option<Shape> {
    let base_len = base_shape.dims.len();
    let mut reported_diagnostics = false;
    let unreduced = match expected {
        LossExpectedInputs::Equal { min, max, inputs } => {
            if &base_len < min || &base_len > max {
                push_error(
                    &mut context,
                    call.range,
                    format!(
                        "Invalid number of dims {} for input ∉ [{min}, {max}]",
                        base_shape.render()
                    ),
                );
                reported_diagnostics = true;
            }

            for pos in 1..*inputs {
                let key = match pos {
                    1 => "target",
                    2 => "input2",
                    _ => break,
                };

                if let Some(target) = get_arg_shape(call, inputs - pos, key, &mut context) {
                    if target != base_shape {
                        push_error(
                            &mut context,
                            call.range,
                            format!(
                                "Input and {key} must have the same shape, found {} vs. {}",
                                base_shape.render(),
                                target.render()
                            ),
                        );
                        reported_diagnostics = true;
                    }
                } else {
                    push_error(&mut context, call.range, format!("Missing `{key}` input"));
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
                push_error(
                    &mut context,
                    call.range,
                    format!(
                        "Invalid number of dims {} for NLL-like loss input",
                        base_shape.render()
                    ),
                );
                reported_diagnostics = true;
            }

            if let Some(target) = get_arg_shape(call, 1, "target", &mut context) {
                if let Some(expected_target) = unreduced.as_ref()
                    && target != expected_target.as_slice()
                {
                    push_error(
                        &mut context,
                        call.range,
                        format!(
                            "NLL-like loss target must have shape [{}], found {}",
                            expected_target.join(" "),
                            target.render()
                        ),
                    );
                    reported_diagnostics = true;
                }
            } else {
                push_error(
                    &mut context,
                    call.range,
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
                    push_error(
                        &mut context,
                        call.range,
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

            let target = get_arg_shape(call, 1, "target", &mut context);
            if target.is_none() {
                push_error(
                    &mut context,
                    call.range,
                    "Missing `target` input".to_string(),
                );
                reported_diagnostics = true;
            }
            let input_lengths = get_arg_shape(call, 2, "input_lengths", &mut context);
            if input_lengths.is_none() {
                push_error(
                    &mut context,
                    call.range,
                    "Missing `input_lengths` input".to_string(),
                );
                reported_diagnostics = true;
            }
            let target_lengths = get_arg_shape(call, 3, "target_lengths", &mut context);
            if target_lengths.is_none() {
                push_error(
                    &mut context,
                    call.range,
                    "Missing `target_lengths` input".to_string(),
                );
                reported_diagnostics = true;
            }

            if let Some(target) = target {
                let valid_target_rank = if is_batched {
                    matches!(target.dims.len(), 1 | 2)
                } else {
                    target.dims.len() == 1
                };
                if !valid_target_rank {
                    push_error(
                        &mut context,
                        call.range,
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
                    push_error(
                        &mut context,
                        call.range,
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
                    push_error(
                        &mut context,
                        call.range,
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
                push_error(
                    &mut context,
                    call.range,
                    format!(
                        "CosineEmbeddingLoss expects input rank 1 or 2, found {}",
                        base_shape.render()
                    ),
                );
                reported_diagnostics = true;
            }

            let input2 = get_arg_shape(call, 1, "input2", &mut context);
            let target = get_arg_shape(call, 2, "target", &mut context);

            if let Some(input2) = input2 {
                if input2 != base_shape {
                    push_error(
                        &mut context,
                        call.range,
                        format!(
                            "CosineEmbeddingLoss inputs must have the same shape, found {} vs. {}",
                            base_shape.render(),
                            input2.render()
                        ),
                    );
                    reported_diagnostics = true;
                }
            } else {
                push_error(
                    &mut context,
                    call.range,
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
                    push_error(
                        &mut context,
                        call.range,
                        format!(
                            "CosineEmbeddingLoss target must have shape [{}], found {}",
                            unreduced.join(" "),
                            target.render()
                        ),
                    );
                    reported_diagnostics = true;
                }
            } else {
                push_error(
                    &mut context,
                    call.range,
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
                push_error(&mut context, call.range, rank_message);
                reported_diagnostics = true;
            }

            for (pos, key) in [(1usize, "positive"), (2usize, "negative")] {
                if let Some(arg) = get_arg_shape(call, pos, key, &mut context) {
                    if arg != base_shape {
                        push_error(
                            &mut context,
                            call.range,
                            format!(
                                "{loss_name} {key} must have shape {}, found {}",
                                base_shape.render(),
                                arg.render()
                            ),
                        );
                        reported_diagnostics = true;
                    }
                } else {
                    push_error(&mut context, call.range, format!("Missing `{key}` input"));
                    reported_diagnostics = true;
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
