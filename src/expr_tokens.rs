//! Shared parsing of small symbolic expressions into token strings.

use rustpython_parser::ast::{Constant, Expr, Operator};
use std::borrow::Cow;

/// Toggle which expression forms may be collapsed into a symbolic token.
#[derive(Clone, Copy)]
pub(crate) struct ExprTokenOptions {
    pub(crate) allow_string_literals: bool,
    pub(crate) allow_attributes: bool,
    pub(crate) allow_add_sub: bool,
    pub(crate) allow_mul: bool,
    pub(crate) allow_div: bool,
    pub(crate) allow_int_cast: bool,
}

/// Conservative dim parsing used by variadic shape arguments (used at [`crate::infer`]).
pub(crate) const DIM_TOKEN_OPTIONS: ExprTokenOptions = ExprTokenOptions {
    allow_string_literals: false,
    allow_attributes: false,
    allow_add_sub: false,
    allow_mul: true,
    allow_div: false,
    allow_int_cast: false,
};

/// Broader parsing used for builtin module constructor arguments (used at [`crate::torch_nn`]).
///
/// Builtin `torch.nn` modules often accept symbolic scalar expressions, tuple
/// members, attribute leaves such as `dtype`, and trivial `int(...)` wrappers.
pub(crate) const MODULE_ARG_TOKEN_OPTIONS: ExprTokenOptions = ExprTokenOptions {
    allow_string_literals: true,
    allow_attributes: true,
    allow_add_sub: true,
    allow_mul: true,
    allow_div: true,
    allow_int_cast: true,
};

/// Token parsing for conv stride/padding/dilation style arguments.
///
/// These parameters are scalar-or-tuple numeric expressions, so string literals
/// and plain attribute leaves stay disabled even though module constructor
/// parsing is otherwise broader.
pub(crate) const CONV_PARAM_TOKEN_OPTIONS: ExprTokenOptions = ExprTokenOptions {
    allow_string_literals: false,
    allow_attributes: false,
    allow_add_sub: true,
    allow_mul: true,
    allow_div: true,
    allow_int_cast: true,
};

/// Minimal scalar parsing for slice bounds before falling back to raw source.
///
/// Slice expressions preserve arbitrary source text as a last resort, so this
/// preset only recognizes the simple scalar cases that are safe to normalize.
pub(crate) const SLICE_BOUND_TOKEN_OPTIONS: ExprTokenOptions = ExprTokenOptions {
    allow_string_literals: false,
    allow_attributes: false,
    allow_add_sub: false,
    allow_mul: false,
    allow_div: false,
    allow_int_cast: true,
};

/// Convert a scalar expression into a symbolic token according to `options`.
///
/// The returned `Cow` borrows where possible to keep the common identifier and
/// string-literal cases cheap, and allocates only when normalization is needed
/// (for example `-N` or `A*B`).
pub(crate) fn expr_to_symbolic_token<'a>(
    expr: &'a Expr,
    options: ExprTokenOptions,
) -> Option<Cow<'a, str>> {
    match expr {
        Expr::Name(name) => Some(Cow::Borrowed(name.id.as_str())),
        Expr::Constant(constant) => match &constant.value {
            Constant::Int(int) => Some(Cow::Owned(int.to_string())),
            Constant::Float(float) => Some(Cow::Owned(float.to_string())),
            Constant::Str(string) if options.allow_string_literals => {
                Some(Cow::Borrowed(string.as_str()))
            }
            _ => None,
        },
        Expr::Attribute(attr) if options.allow_attributes => {
            Some(Cow::Borrowed(attr.attr.as_str()))
        }
        Expr::UnaryOp(unary) => match unary.op {
            rustpython_parser::ast::UnaryOp::USub => {
                expr_to_symbolic_token(unary.operand.as_ref(), options)
                    .map(|token| Cow::Owned(format!("-{token}")))
            }
            rustpython_parser::ast::UnaryOp::UAdd => {
                expr_to_symbolic_token(unary.operand.as_ref(), options)
            }
            _ => None,
        },
        Expr::BinOp(bin) => {
            let op = match bin.op {
                Operator::Add if options.allow_add_sub => "+",
                Operator::Sub if options.allow_add_sub => "-",
                Operator::Mult if options.allow_mul => "*",
                Operator::Div | Operator::FloorDiv if options.allow_div => "/",
                _ => return None,
            };
            let left = expr_to_symbolic_token(bin.left.as_ref(), options)?;
            let right = expr_to_symbolic_token(bin.right.as_ref(), options)?;
            Some(Cow::Owned(format!("{left}{op}{right}")))
        }
        Expr::Call(call) if options.allow_int_cast => {
            if let Expr::Name(name) = call.func.as_ref()
                && name.id.as_str() == "int"
                && call.keywords.is_empty()
            {
                return call
                    .args
                    .first()
                    .and_then(|expr| expr_to_symbolic_token(expr, options));
            }
            None
        }
        _ => None,
    }
}

/// Parse either a scalar expression or a tuple/list of scalar expressions into
/// owned token strings.
pub(crate) fn expr_to_symbolic_tokens(
    expr: &Expr,
    options: ExprTokenOptions,
) -> Option<Vec<String>> {
    match expr {
        Expr::Tuple(tuple) => tuple
            .elts
            .iter()
            .map(|expr| expr_to_symbolic_token(expr, options).map(Cow::into_owned))
            .collect(),
        Expr::List(list) => list
            .elts
            .iter()
            .map(|expr| expr_to_symbolic_token(expr, options).map(Cow::into_owned))
            .collect(),
        other => expr_to_symbolic_token(other, options)
            .map(Cow::into_owned)
            .map(|token| vec![token]),
    }
}
