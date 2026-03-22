use phf::Set;
use phf_macros::phf_set;

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
