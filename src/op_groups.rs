use phf::Set;
use phf_macros::phf_set;
use rustpython_parser::ast::Identifier;

use crate::{Imports, infer::Transpose, is_alias_of};

/// Torch functions and operations whose inference is supported.
///
/// This is used to match a torch function (imported, from a qualified or as a tensor method)
/// against a supported tensor impolement.
pub enum TorchOp {
    /// `torch.mm`
    MatMul,
    /// Aggregate over a dimension or the whole tensor: `tensor.sum`, `tensor.max`, etc.
    Aggr,
    /// Shape-wise noops, but accept a dim argument like `softmax`.
    NoopDim,
    /// Shape-wise noops.
    Noop,
    /// Tensor initialization, with variadic arguments or *_like operations.
    Creation { is_size: bool },
    /// Tensor initialization from a range: `tensor.arange`, `tensor.linspace`, etc.
    RangeOp(RangeOps),
    /// Only change the dtype.
    NoArg { can_be_function: bool },
    /// Element-wise, broadcastable operations: `torch.mul`, `torch.eq`, etc..
    Broadcastable(BroadcastOp),
    /// `torch.view` and `torch.reshape`
    View,
    /// Permute-like operations.
    Transpose(Transpose),
    /// `torch.unsqueeze`.
    Unsqueeze,
    /// `torch.squeeze`.
    Squeeze,
    /// The operation is not supported or not properly indicated by the user.
    Unknown,
}

impl TorchOp {
    pub fn from_attr(attr_name: &str) -> Self {
        if attr_name == "mm" {
            Self::MatMul
        } else if attr_name == "view" || attr_name == "reshape" {
            Self::View
        } else if attr_name == "permute" {
            // only permute is checked since transpose
            Self::Transpose(Transpose::Permute)
        } else if attr_name == "transpose" {
            Self::Transpose(Transpose::Explicit)
        } else if attr_name == "t" {
            Self::Transpose(Transpose::T)
        } else if attr_name == "unsqueeze" {
            Self::Unsqueeze
        } else if attr_name == "squeeze" {
            Self::Squeeze
        } else if AGGR_ALIASES.contains(attr_name) {
            Self::Aggr
        } else if NOOP_DIM_ALIASES.contains(attr_name) {
            Self::NoopDim
        } else if NOOP_ALIASES.contains(attr_name) {
            Self::Noop
        } else if CREATION_SIZE_ALIASES.contains(attr_name) {
            Self::Creation { is_size: true }
        } else if CREATION_LIKE_ALIASES.contains(attr_name) {
            Self::Creation { is_size: false }
        } else if let Ok(op) = RangeOps::try_from(attr_name) {
            Self::RangeOp(op)
        } else if TO_NOARG_ALIASES.contains(attr_name) {
            Self::NoArg {
                can_be_function: attr_name == "to",
            }
        } else if let Some(op) = BroadcastOp::try_from_attr(attr_name) {
            Self::Broadcastable(op)
        } else {
            Self::Unknown
        }
    }
    pub(crate) fn as_call(func_name_id: &Identifier, imports: &Imports) -> Self {
        if is_alias_of("mm", func_name_id, imports) {
            Self::MatMul
        } else if let Some(op) = BroadcastOp::try_from_alias(func_name_id, imports) {
            Self::Broadcastable(op)
        } else if is_alias_of("view", func_name_id, imports)
            || is_alias_of("reshape", func_name_id, imports)
        {
            Self::View
        } else if is_alias_of("permute", func_name_id, imports) {
            Self::Transpose(Transpose::Permute)
        } else if is_alias_of("transpose", func_name_id, imports) {
            Self::Transpose(Transpose::Explicit)
        } else if is_alias_of("t", func_name_id, imports) {
            Self::Transpose(Transpose::T)
        } else if is_alias_of("unsqueeze", func_name_id, imports) {
            Self::Unsqueeze
        } else if is_alias_of("squeeze", func_name_id, imports) {
            Self::Squeeze
        } else if is_alias_of("sum", func_name_id, imports) {
            Self::Aggr
        } else if is_alias_of("softmax", func_name_id, imports) {
            Self::NoopDim
        } else if is_alias_of("noop", func_name_id, imports) {
            Self::Noop
        } else if is_alias_of("Tensor", func_name_id, imports) {
            Self::Creation { is_size: true }
        } else if is_alias_of("like", func_name_id, imports) {
            Self::Creation { is_size: false }
        } else if let Some(Ok(range_op)) = CREATION_RANGE_ALIASES
            .iter()
            .filter(|x| is_alias_of(x, func_name_id, imports))
            .map(|&x| RangeOps::try_from(x))
            .next()
        {
            Self::RangeOp(range_op)
        } else {
            Self::Unknown
        }
    }
}

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
    "bitwise_not",
    "logical_not",
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
    phf_set!["Tensor", "zeros", "ones", "empty", "full", "rand", "randn"];

/// `*_like`, accepting a tensor as input.
pub static CREATION_LIKE_ALIASES: Set<&'static str> = phf_set![
    "empty_like",
    "zeros_like",
    "ones_like",
    "rand_like",
    "randn_like",
    "randint_like",
    // full_like does require an extra arg "fill_value", but
    // catching that is the job of a generalist LSP
    "full_like",
];

pub static CREATION_RANGE_ALIASES: Set<&'static str> =
    phf_set!["randperm", "linspace", "logspace", "arange", "range"];

#[derive(PartialEq)]
pub enum RangeOps {
    /// Single position argument is the shape.
    Randperm,
    /// Steps is the shape
    Linspace,
    /// Steps is the shape
    Logspace,
    /// (end - start) / steps is the shape
    Range,
    /// (end - start) / steps - 1 is the shape
    Arange,
}

impl TryFrom<&'_ str> for RangeOps {
    type Error = ();

    fn try_from(value: &'_ str) -> Result<Self, Self::Error> {
        match value {
            "randperm" => Ok(Self::Randperm),
            "linspace" => Ok(Self::Linspace),
            "logspace" => Ok(Self::Logspace),
            "arange" => Ok(Self::Arange),
            "range" => Ok(Self::Range),
            _ => Err(()),
        }
    }
}

impl RangeOps {
    pub fn dtype_arg_pos(&self) -> usize {
        match self {
            RangeOps::Randperm => 3,
            RangeOps::Range | RangeOps::Arange | RangeOps::Linspace => 4,
            RangeOps::Logspace => 5,
        }
    }
}

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
    "detach",
};

/// Shape-wise NoOp, changes dtype, do not accept arguments.
pub static TO_NOARG_ALIASES: Set<&'static str> = phf_set! {
    "to",
    "float",
    "long",
    "int",
    "byte",
    "bfloat16",
    "cfloat",
    "bool",
    "neg",
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

/// Functional equivalent to broadcastable operators (+, *, -, etc.).
pub static BROADCASTABLE_ALIASES: Set<&'static str> = phf_set![
    "add",
    "sub",
    "mul",
    "div",
    "true_divide",
    "floor_divide",
    "remainder",
    "fmod",
    "pow",
    "positive",
];

/// Functional equivalent to bitwise broadcastable operators (&, >>, etc.).
pub static BITWISE_ALIASES: Set<&'static str> = phf_set![
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_left_shift",
    "bitwise_right_shift",
];

/// Functional equivalent to bitwise broadcastable operators (&, >>, etc.).
pub static EQ_BROADCAST_ALIASES: Set<&'static str> = phf_set![
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "logical_or",
    "logical_and",
    "logical_xor",
];

/// Broadcastable operations.
pub enum BroadcastOp {
    /// +, -, /, etc.
    Arithmetic,
    /// &, >>, etc.: constrained to input of dtype bool/int
    Bitwise,
    /// ==, !=, etc.: returns a bool dtype
    Eq,
}

impl BroadcastOp {
    pub(crate) fn try_from_alias(func_name_id: &Identifier, imports: &Imports) -> Option<Self> {
        if is_alias_of("broadcast", func_name_id, imports) {
            Some(Self::Arithmetic)
        } else if is_alias_of("bitwise", func_name_id, imports) {
            Some(Self::Bitwise)
        } else if is_alias_of("broadcast_eq", func_name_id, imports) {
            Some(Self::Eq)
        } else {
            None
        }
    }

    pub(crate) fn try_from_attr(attr_name: &str) -> Option<Self> {
        if BROADCASTABLE_ALIASES.contains(attr_name) {
            Some(Self::Arithmetic)
        } else if BITWISE_ALIASES.contains(attr_name) {
            Some(Self::Bitwise)
        } else if EQ_BROADCAST_ALIASES.contains(attr_name) {
            Some(Self::Eq)
        } else {
            None
        }
    }
}
