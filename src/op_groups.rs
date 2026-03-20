use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use phf::Set;
use phf_macros::phf_set;
use rustpython_parser::ast::{Identifier, Stmt, StmtImportFrom};

use crate::{infer::Transpose, is_alias_of};

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
    NoArg { predef_dtype: Option<&'static str> },
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
    /// `torch.Tensor.expand`
    Expand,
    /// Conv1d, Conv2d, Conv3d
    Conv(usize),
    /// `torch.repeat`
    Repeat,
    /// `torch.repeat_interleave`
    RepeatInterleave,
    /// `torch.flatten`, `torch.ravel`
    Flatten,
    /// `torch.quantile`,`torch.nan_quantile`
    Quantile,
    /// `torch.where`
    Condition,
    /// `torch.take`
    Take,
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
        } else if attr_name == "expand" {
            Self::Expand
        } else if attr_name == "conv1d" {
            Self::Conv(1)
        } else if attr_name == "conv2d" {
            Self::Conv(2)
        } else if attr_name == "conv3d" {
            Self::Conv(3)
        } else if attr_name == "repeat" {
            Self::Repeat
        } else if attr_name == "where" {
            Self::Condition
        } else if attr_name == "take" {
            Self::Take
        } else if attr_name == "repeat_interleave" {
            Self::RepeatInterleave
        } else if AGGR_ALIASES.contains(attr_name) {
            Self::Aggr
        } else if QUANTILE_ALIASES.contains(attr_name) {
            Self::Quantile
        } else if NOOP_DIM_ALIASES.contains(attr_name) {
            Self::NoopDim
        } else if NOOP_ALIASES.contains(attr_name) {
            Self::Noop
        } else if CREATION_SIZE_ALIASES.contains(attr_name) {
            Self::Creation { is_size: true }
        } else if CREATION_LIKE_ALIASES.contains(attr_name) {
            Self::Creation { is_size: false }
        } else if CREATION_NEW_LIKE_ALIASES.contains(attr_name) {
            Self::Creation { is_size: true }
        } else if let Ok(op) = RangeOps::try_from(attr_name) {
            Self::RangeOp(op)
        } else if FLATTEN_ALIASES.contains(attr_name) {
            Self::Flatten
        } else if TO_NOARG_ALIASES.contains(attr_name) {
            Self::NoArg {
                predef_dtype: match attr_name {
                    "to" => None,
                    key => TO_NOARG_ALIASES.get_key(key).copied(),
                },
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
        } else if is_alias_of("where", func_name_id, imports) {
            Self::Condition
        } else if is_alias_of("take", func_name_id, imports) {
            Self::Take
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
        } else if is_alias_of("conv1d", func_name_id, imports) {
            Self::Conv(1)
        } else if is_alias_of("conv2d", func_name_id, imports) {
            Self::Conv(2)
        } else if is_alias_of("conv3d", func_name_id, imports) {
            Self::Conv(3)
        } else if is_alias_of("softmax", func_name_id, imports) {
            Self::NoopDim
        } else if is_alias_of("noop", func_name_id, imports) {
            Self::Noop
        } else if is_alias_of("quantile", func_name_id, imports) {
            Self::Quantile
        } else if is_alias_of("Tensor", func_name_id, imports) {
            Self::Creation { is_size: true }
        } else if is_alias_of("like", func_name_id, imports) {
            Self::Creation { is_size: false }
        } else if is_alias_of("to", func_name_id, imports) {
            Self::NoArg { predef_dtype: None }
        } else if is_alias_of("flatten", func_name_id, imports) {
            Self::Flatten
        } else if is_alias_of("repeat", func_name_id, imports) {
            Self::Repeat
        } else if is_alias_of("repeat_interleave", func_name_id, imports) {
            Self::RepeatInterleave
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
    "argmax",
    "argmin",
    "all",
    "any",
    "count_nonzero",
    "logsumexp",
    "norm",
];

/// Quantiles: q is a scalar, dim is at position 2.
/// https://docs.pytorch.org/docs/stable/generated/torch.quantile.html
pub static QUANTILE_ALIASES: Set<&'static str> = phf_set!["quantile", "nanquantile",];

/// Shape-wise NoOp, must have dim dimension
pub static NOOP_DIM_ALIASES: Set<&'static str> = phf_set![
    "softmax",
    "log_softmax",
    // dim is optional
    "argsort",
    "bitwise_not",
    "logical_not",
    "flip",
    "rot90",
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
pub static CREATION_NEW_LIKE_ALIASES: Set<&'static str> =
    phf_set!["new_zeros", "new_ones", "new_empty", "new_full"];

/// `*_like`, accepting a tensor as input.
pub static CREATION_LIKE_ALIASES: Set<&'static str> = phf_set![
    "tensor",
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
    "fliplr",
    "flipud",
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

/// Flatten, product of dimensions is preserved.
pub static FLATTEN_ALIASES: Set<&'static str> = phf_set!["flatten", "ravel"];

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
    "long",
];

/// These are non-official dtypes used to check that operations are valid.
pub(crate) enum SimpleDtype {
    Float,
    Int { long: bool },
    Bool,
    Unknown,
}

impl From<&str> for SimpleDtype {
    fn from(value: &str) -> SimpleDtype {
        // need to include common user-defined dtypes loke Float, Int, etc. from jaxtyping
        match value.to_lowercase().as_str() {
            "f" | "float" | "float32" | "float64" | "float16" | "bfloat16" | "complex32"
            | "complex64" | "complex128" | "float8_e4m3fn" | "float8_e5m2" | "float8_e4m3fnuz"
            | "float8_e5m2fnuz" | "float8_e8m0fnu" | "float4_e2m1fn_x2" => SimpleDtype::Float,
            "i" | "uint8" | "int8" | "uint16" | "int16" | "uint32" | "int32" | "uint64" => {
                SimpleDtype::Int { long: false }
            }
            "l" | "long" | "int64" => SimpleDtype::Int { long: true },
            "b" | "bool" => SimpleDtype::Bool,
            _ => SimpleDtype::Unknown,
        }
    }
}

impl SimpleDtype {
    pub fn is_long(&self) -> bool {
        matches!(self, SimpleDtype::Int { long: true })
    }
}

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
    "logaddexp",
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

/// Aliases for [transpose](https://docs.pytorch.org/docs/stable/generated/torch.transpose.html#torch.transpose).
pub static TRANSPOSE_ALIASES: Set<&'static str> = phf_set!["transpose", "swapaxes", "swapdims"];

/// Broadcastable operations.
pub enum BroadcastOp {
    /// +, -, /, etc.
    Arithmetic,
    /// &, >>, etc.: constrained to input of dtype bool/int
    Bitwise { only_right: bool },
    /// ==, !=, etc.: returns a bool dtype
    Eq,
}

impl BroadcastOp {
    pub(crate) fn try_from_alias(func_name_id: &Identifier, imports: &Imports) -> Option<Self> {
        if is_alias_of("broadcast", func_name_id, imports) {
            Some(Self::Arithmetic)
        } else if is_alias_of("bitwise", func_name_id, imports) {
            Some(Self::Bitwise { only_right: false })
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
            Some(Self::Bitwise { only_right: false })
        } else if "masked_fill" == attr_name {
            Some(Self::Bitwise { only_right: true })
        } else if EQ_BROADCAST_ALIASES.contains(attr_name) {
            Some(Self::Eq)
        } else {
            None
        }
    }
}

#[derive(Default, Clone)]
pub struct Imports {
    pub torch_aliases: HashSet<Identifier>,
    // e.g., `import torch.nn.functional as F`
    pub torch_nn_functional_aliases: HashSet<Identifier>,
    /// Maps simple function name (e.g., "mm") to all aliases in scope.
    pub func_aliases: HashMap<&'static str, HashSet<Identifier>>,
    /// Module alias mapping for `import foo as bar` style.
    pub module_aliases: HashMap<Identifier, String>,
    /// Symbol imports mapping alias -> (module, original name).
    pub from_imports: HashMap<Identifier, (String, Identifier)>,
}

/// Map all known operations to their importing aliases, for instance:
///
/// ```python
/// import torch as t
/// from torch import sum as torch_sum
/// ```
///
/// In that example, shapels has to keep track that `torch_sum` is
/// an alias to `torch.sum` and `t` of `torch` to identify this
/// functions in the scope and perform shape inference.
pub fn collect_imports(
    module: &[Stmt],
    module_path: Option<&Path>,
    project_root: Option<&Path>,
) -> Imports {
    let mut imports = Imports::default();
    // seed known function names
    for fname in [
        "mm",
        "view",
        "reshape",
        "sum",
        "permute",
        "t",
        "softmax",
        "Tensor",
        "randperm",
        "linspace",
        "logspace",
        "arange",
        "range",
        "to",
        "conv1d",
        "conv2d",
        "conv3d",
        "repeat_interleave",
        "quantile",
        "where",
        "take",
        "masked_fill",
    ] {
        imports
            .func_aliases
            .entry(fname)
            .or_insert_with(HashSet::new);
    }
    imports
        .torch_aliases
        .insert(Identifier::from("torch".to_string()));
    imports
        .torch_nn_functional_aliases
        .insert(Identifier::from("torch.nn.functional".to_string()));

    for stmt in module {
        match stmt {
            Stmt::Import(import) => {
                for alias in &import.names {
                    let name = alias.name.as_str();
                    let as_id = alias
                        .asname
                        .clone()
                        .unwrap_or_else(|| Identifier::from(name));
                    imports
                        .module_aliases
                        .insert(as_id.clone(), name.to_string());
                    if name == "torch" {
                        imports.torch_aliases.insert(as_id.clone());
                    } else if name == "torch.nn.functional" {
                        imports.torch_nn_functional_aliases.insert(as_id.clone());
                    }
                    if let Some(val) = imports.func_aliases.get_mut(name) {
                        val.insert(as_id);
                    } else if AGGR_ALIASES.contains(name) {
                        imports.func_aliases.entry("sum").or_default().insert(as_id);
                    }
                }
            }
            Stmt::ImportFrom(f) => {
                let resolved_module = resolve_from_module(f, module_path, project_root);
                if let Some(module) = &resolved_module
                    && (module == "torch" || module == "torch.nn.functional")
                {
                    for alias in &f.names {
                        let name = alias.name.as_str();
                        if let Some(val) = imports.func_aliases.get_mut(name) {
                            let id = alias
                                .asname
                                .clone()
                                .unwrap_or_else(|| Identifier::from(name));
                            val.insert(id);
                        } else {
                            for (container, key) in [
                                (&AGGR_ALIASES, "sum"),
                                (&NOOP_DIM_ALIASES, "softmax"),
                                (&NOOP_ALIASES, "noop"),
                                (&CREATION_SIZE_ALIASES, "Tensor"),
                                (&BROADCASTABLE_ALIASES, "broadcast"),
                                (&BITWISE_ALIASES, "bitwise"),
                                (&EQ_BROADCAST_ALIASES, "broadcast_eq"),
                                (&CREATION_LIKE_ALIASES, "like"),
                                (&TRANSPOSE_ALIASES, "transpose"),
                                (&FLATTEN_ALIASES, "flatten"),
                                (&QUANTILE_ALIASES, "quantile"),
                            ] {
                                if container.contains(name) {
                                    let id = alias
                                        .asname
                                        .clone()
                                        .unwrap_or_else(|| Identifier::from(name));
                                    imports.func_aliases.entry(key).or_default().insert(id);
                                }
                            }
                        }
                    }
                } else if let Some(module) = &resolved_module {
                    for alias in &f.names {
                        let id = alias
                            .asname
                            .clone()
                            .unwrap_or_else(|| Identifier::from(alias.name.as_str()));
                        imports
                            .from_imports
                            .insert(id, (module.to_string(), alias.name.clone()));
                    }
                }
            }
            _ => {}
        }
    }
    imports
}

fn module_name_from_path(path: &Path, project_root: Option<&Path>) -> Option<String> {
    let mut dir = path.parent()?;
    let mut parts = Vec::new();
    loop {
        if dir.join("__init__.py").exists() {
            if let Some(name) = dir.file_name().and_then(|s| s.to_str()) {
                parts.push(name.to_string());
            }
        } else {
            break;
        }
        if let Some(root) = project_root
            && dir == root
        {
            break;
        }
        if let Some(parent) = dir.parent() {
            dir = parent;
        } else {
            break;
        }
    }
    if parts.is_empty() {
        None
    } else {
        parts.reverse();
        Some(parts.join("."))
    }
}

fn resolve_from_module(
    f: &StmtImportFrom,
    module_path: Option<&Path>,
    project_root: Option<&Path>,
) -> Option<String> {
    // Absolute import
    let level_val = f.level.map(|i| i.to_usize()).unwrap_or(0);
    if level_val == 0 {
        return f.module.as_ref().map(|m| m.to_string());
    }
    let base_pkg = module_path
        .and_then(|p| module_name_from_path(p, project_root))
        .unwrap_or_default();
    if base_pkg.is_empty() {
        return f.module.as_ref().map(|m| m.to_string());
    }
    let mut parts: Vec<String> = base_pkg.split('.').map(|s| s.to_string()).collect();
    if level_val > 0 {
        let pops = level_val.saturating_sub(1);
        for _ in 0..pops {
            if parts.pop().is_none() {
                break;
            }
        }
    }
    if let Some(mod_name) = &f.module {
        for p in mod_name.as_str().split('.') {
            parts.push(p.to_string());
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("."))
    }
}
