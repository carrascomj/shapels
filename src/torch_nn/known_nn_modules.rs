use phf::{Map, Set};
use phf_macros::{phf_map, phf_set};

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
pub static LOSS_MODULES: Map<&'static str, usize> = phf_map![
    "L1Loss" => 2,
    "MSELoss" => 2,
    "CrossEntropyLoss" => 4,
    "CTCLoss" => 1,
    "NLLLoss" => 4,
    "PoissonNLLLoss" => 5,
    // this is by named arg only, but that's a task for a generalist LSP
    "GaussianNLLLoss" => 2,
    "KLDivLoss" => 2,
    "BCELoss" => 3,
    "BCEWithLogitsLoss" => 3,
    "MarginRankingLoss" => 3,
    "HingeEmbeddingLoss" => 3,
    "MultiLabelMarginLoss" => 2,
    "HuberLoss" => 0,
    "SmoothL1Loss" => 2,
    "SoftMarginLoss" => 2,
    "MultiLabelSoftMarginLoss" => 3,
    "CosineEmbeddingLoss" => 3,
    "MultiMarginLoss" => 5,
    "TripletMarginLoss" => 6,
    "TripletMarginWithDistanceLoss" => 3,
];
