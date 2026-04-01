use phf::{Map, Set};
use phf_macros::{phf_map, phf_set};

use crate::infer::LossExpectedInputs;

/// Builtin `torch.nn.Module`s that are safe to treat as shape no-ops.
///
/// This list intentionally excludes:
/// - modules that change shape or rank, such as pooling/padding/flattening;
/// - modules that return non-tensor structures, such as attention or losses;
/// - modules with dedicated parsing/inference elsewhere, such as `BatchNorm*d`;
/// - wrappers/containers rather than direct tensor operations.
pub static NOOP_NN_MODULES: Set<&'static str> = phf_set![
    "ELU",
    "Hardshrink",
    "Hardsigmoid",
    "Hardtanh",
    "Hardswish",
    "LeakyReLU",
    "LogSigmoid",
    "PReLU",
    "ReLU",
    "ReLU6",
    "RReLU",
    "SELU",
    "CELU",
    "GELU",
    "Sigmoid",
    "SiLU",
    "Mish",
    "Softplus",
    "Softshrink",
    "Softsign",
    "Tanh",
    "Tanhshrink",
    "Threshold",
    "Softmin",
    "Softmax",
    "Softmax2d",
    "LogSoftmax",
    "LazyBatchNorm1d",
    "LazyBatchNorm2d",
    "LazyBatchNorm3d",
    "GroupNorm",
    "SyncBatchNorm",
    "LazyInstanceNorm1d",
    "InstanceNorm1d",
    "LazyInstanceNorm2d",
    "InstanceNorm2d",
    "LazyInstanceNorm3d",
    "InstanceNorm3d",
    "LayerNorm",
    "LocalResponseNorm",
    "RMSNorm",
    "Identity",
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "AlphaDropout",
    "FeatureAlphaDropout",
];

/// [Losses in `torch.nn`](https://docs.pytorch.org/docs/stable/nn.html#loss-functions).
///
/// They're mapped to the position of the argument "reduction" in each case.
pub static LOSS_MODULES: Map<&'static str, (usize, LossExpectedInputs)> = phf_map![
    "L1Loss" => (2, LossExpectedInputs::const_default()),
    "MSELoss" => (2, LossExpectedInputs::const_default()),
    "CrossEntropyLoss" => (4, LossExpectedInputs::NllLike),
    "CTCLoss" => (1, LossExpectedInputs::Ctc),
    "NLLLoss" => (4, LossExpectedInputs::NllLike),
    "PoissonNLLLoss" => (5, LossExpectedInputs::const_default()),
    // this is by named arg only, but that's a task for a generalist LSP
    "GaussianNLLLoss" => (2, LossExpectedInputs::const_default()),
    "KLDivLoss" => (2, LossExpectedInputs::const_default()),
    "BCELoss" => (3, LossExpectedInputs::const_default()),
    "BCEWithLogitsLoss" => (3, LossExpectedInputs::const_default()),
    "MarginRankingLoss" => (3, LossExpectedInputs::equal(0, 1, 3)),
    "HingeEmbeddingLoss" => (3, LossExpectedInputs::const_default()),
    "MultiLabelMarginLoss" => (2, LossExpectedInputs::equal(1, 2, 2)),
    "HuberLoss" => (0, LossExpectedInputs::const_default()),
    "SmoothL1Loss" => (2, LossExpectedInputs::const_default()),
    "SoftMarginLoss" => (2, LossExpectedInputs::const_default()),
    "MultiLabelSoftMarginLoss" => (3, LossExpectedInputs::equal(2, 2, 2)),
    "CosineEmbeddingLoss" => (3, LossExpectedInputs::CosineEmbedding),
    // TODO(carrascomj): without the dK part
    "MultiMarginLoss" => (5, LossExpectedInputs::NllLike),
    "TripletMarginLoss" => (6, LossExpectedInputs::Triplet),
    "TripletMarginWithDistanceLoss" => (3, LossExpectedInputs::TripletDistance),
];
