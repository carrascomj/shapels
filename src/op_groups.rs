use phf::Set;
use phf_macros::phf_set;

/// Operations that accept an argument dim (integer or sequence),
/// return a single tensor and the provided dims have been reduced
/// from the output tensor.
pub static AGGR_ALIASES: Set<&'static str> = phf_set![
    "sum",
    "mean",
    "prod",
    "amax",
    "amin",
    "std",
    "var",
    "nanmean",
    "nansum",
    "nanprod",
    "nanstd",
    "nanvar",
    // FIXME: quantile and nanquantile only apply iff
    // the q argument is a scalar
    "quantile",
    "nanquantile",
    "argmax",
    "argmin",
    "all",
    "any",
    "count_nonzero",
    "logsumexp",
    "norm",
];

/// Shape-wise NoOp, must have dim dimension
pub static NOOP_DIM_ALIASES: Set<&'static str> = phf_set![
    "softmax",
    "log_softmax",
    // dim is optional
    "argsort",
];

// TODO: Tensor should be separated from the rest
/// Subset of creation ops that accept args in the form
///
/// ```python
/// def zeros(*size, *, out=None, dtype=None):
///     pass
///
/// size can be simply arguments like `zeros(1,2,3,4,5)`
/// or a single argument with a sequence: `zeros([1,2,3,4])`
/// ```
pub static CREATION_SIZE_ALIASES: Set<&'static str> =
    phf_set!["Tensor", "zeros", "ones", "empty", "full"];

// TODO(carrascomj): separate this into torch.ATTR_NAME only functions
// e.g., relu is both a tensor and and a top-level function but contiguous is not
/// Shape-wise NoOp
pub static NOOP_ALIASES: Set<&'static str> = phf_set! {
    "relu",
    "contiguous",
    "xlogy",
    "relu6",
    "sigmoid",
    "tanh",
    "silu",
    "swish",
    "mish",
    "hardswish",
    "hardtanh",
    "elu",
    "celu",
    "selu",
    "softplus",
    "softsign",
    "gelu",
    "leaky_relu",
    "prelu",
    "rrelu",
    "threshold",
    "clamp",
    "clip",
    "hardshrink",
    "softshrink",
    "tanhshrink",
    "round",
    "floor",
    "ceil",
    "trunc",
    "frac",
    "reciprocal",
    "abs",
    "sqrt",
    "rsqrt",
    "square",
    "sign",
    "neg",
    "exp",
    "log",
    "log1p",
    "expm1",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "asin",
    "acos",
    "atan",
    "atanh",
    "erf",
    "erfc",
    "erfinv",
    // torch.Tensor only
    "cpu",
    "cuda",
};

/// Shape-wise NoOp, changes dtype, do not accept arguments.
pub static TO_NOARG_ALIASES: Set<&'static str> = phf_set! {
    "float",
    "long",
    "int",
    "byte",
    "bfloat16",
    "cfloat",
    "bool",
};

pub static TORCH_DTYPES: Set<&'static str> = phf_set![
    "float32",
    "float64",
    "float16",
    "bfloat16",
    "complex32",
    "complex64",
    "complex128",
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e4m3fnuz",
    "float8_e5m2fnuz",
    "float8_e8m0fnu",
    "float4_e2m1fn_x2",
    "uint8",
    "int8",
    "uint16",
    "int16",
    "uint32",
    "int32",
    "uint64",
    "int64",
    "bool",
];
