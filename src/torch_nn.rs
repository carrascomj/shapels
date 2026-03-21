use crate::infer::{infer_batchnorm_module, infer_conv_module, infer_linear_module};
use crate::module_resolution::{
    ClassMap, ClassRef, FuncMap, ModuleCache, ResolvedModule, imported_class_ref,
};
use crate::op_groups::Imports;
use crate::{HoverInfo, Shape, VarState, infer_resolved_module_shape};
use lsp_types::{Diagnostic, Range};
use rustpython_parser::ast::{Expr, ExprCall, Identifier, Operator};
use rustpython_parser::text_size::TextRange;
use std::collections::HashMap;
use std::path::Path;

/// Attribute in __init__ parsed into a builtin `torch.nn.Module`.
#[derive(Debug, Clone)]
pub(crate) enum TorchNNModule {
    Linear {
        in_features: Option<String>,
        out_features: Option<String>,
    },
    BatchNorm {
        dims: usize,
        num_features: Option<String>,
    },
    Conv {
        dims: usize,
        in_channels: Option<String>,
        out_channels: Option<String>,
        kernel_size: Vec<String>,
        stride: Vec<String>,
        padding: Vec<String>,
        dilation: Vec<String>,
        groups: Option<String>,
    },
    Sequential(Vec<ResolvedModule>),
    /// Any unknown identifier from torch.nn.*, is parsed as Noop.
    Noop,
}

pub(crate) fn module_from_constructor_call(
    call_expr: &Expr,
    class_map: &ClassMap,
    imports: &Imports,
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> Option<ResolvedModule> {
    let Expr::Call(call) = call_expr else {
        return None;
    };

    match call.func.as_ref() {
        Expr::Name(name) => {
            if class_map.contains_key(&name.id) {
                return Some(ResolvedModule::User(ClassRef {
                    name: name.id.clone(),
                    module: None,
                }));
            }
            if let Some((module_name, original)) = imports.from_imports.get(&name.id) {
                if module_name == "torch.nn" {
                    if original.as_str() == "Parameter" {
                        return None;
                    }
                    return Some(ResolvedModule::Builtin(
                        TorchNNModule::from_constructor_call(
                            call_expr,
                            class_map,
                            imports,
                            module_cache,
                            module_path,
                        ),
                    ));
                }
                if let Some(class_ref) = imported_class_ref(
                    module_name,
                    original,
                    module_cache.as_deref_mut(),
                    module_path,
                ) {
                    return Some(ResolvedModule::User(class_ref));
                }
            }
        }
        Expr::Attribute(attr) => {
            if is_torch_nn_constructor(call_expr, imports) {
                if attr.attr.as_str() == "Parameter" {
                    return None;
                }
                return Some(ResolvedModule::Builtin(
                    TorchNNModule::from_constructor_call(
                        call_expr,
                        class_map,
                        imports,
                        module_cache,
                        module_path,
                    ),
                ));
            }
            if let Expr::Name(module_ident) = attr.value.as_ref()
                && let Some(module_name) = imports.module_aliases.get(&module_ident.id)
                && let Some(class_ref) =
                    imported_class_ref(module_name, &attr.attr, module_cache, module_path)
            {
                return Some(ResolvedModule::User(class_ref));
            }
        }
        _ => {}
    }

    None
}

impl TorchNNModule {
    pub(crate) fn from_constructor_call(
        call_expr: &Expr,
        class_map: &ClassMap,
        imports: &Imports,
        mut module_cache: Option<&mut ModuleCache>,
        module_path: Option<&Path>,
    ) -> Self {
        let Expr::Call(call) = call_expr else {
            return Self::Noop;
        };
        let Some(constructor_name) = torch_nn_constructor_name(call_expr, imports) else {
            return Self::Noop;
        };
        let normalized = constructor_name.to_ascii_lowercase();

        if normalized == "linear" {
            return Self::Linear {
                in_features: get_call_arg(call, "in_features", 0).and_then(expr_to_token),
                out_features: get_call_arg(call, "out_features", 1).and_then(expr_to_token),
            };
        }

        if normalized.starts_with("batchnorm")
            && let Some(dims) = parse_module_dims(&normalized, "batchnorm")
        {
            return Self::BatchNorm {
                dims,
                num_features: get_call_arg(call, "num_features", 0).and_then(expr_to_token),
            };
        }

        if normalized.starts_with("conv")
            && let Some(dims) = parse_module_dims(&normalized, "conv")
        {
            return Self::Conv {
                dims,
                in_channels: get_call_arg(call, "in_channels", 0).and_then(expr_to_token),
                out_channels: get_call_arg(call, "out_channels", 1).and_then(expr_to_token),
                kernel_size: get_call_arg(call, "kernel_size", 2)
                    .and_then(expr_to_tokens)
                    .unwrap_or_default(),
                stride: get_call_arg(call, "stride", 3)
                    .and_then(expr_to_tokens)
                    .unwrap_or_default(),
                padding: get_call_arg(call, "padding", 4)
                    .and_then(expr_to_tokens)
                    .unwrap_or_default(),
                dilation: get_call_arg(call, "dilation", 5)
                    .and_then(expr_to_tokens)
                    .unwrap_or_default(),
                groups: get_call_arg(call, "groups", 6).and_then(expr_to_token),
            };
        }

        if normalized == "sequential" {
            let mut modules = Vec::new();
            for element in sequential_elements(call) {
                if let Some(module) = module_from_constructor_call(
                    element,
                    class_map,
                    imports,
                    module_cache.as_deref_mut(),
                    module_path,
                ) {
                    modules.push(module);
                }
            }
            return Self::Sequential(modules);
        }

        Self::Noop
    }

    pub(crate) fn infer_builtin_module(
        &self,
        base_shape: Shape,
        vars: &HashMap<Identifier, VarState>,
        func_map: &FuncMap,
        imports: &Imports,
        class_map: &ClassMap,
        call_stack: &mut Vec<Identifier>,
        diagnostics: &mut Vec<Diagnostic>,
        hover_entries: &mut Vec<(Range, HoverInfo)>,
        source: &str,
        range: TextRange,
        mut module_cache: Option<&mut ModuleCache>,
        module_path: Option<&Path>,
    ) -> Option<Shape> {
        match self {
            Self::Linear {
                in_features,
                out_features,
            } => infer_linear_module(
                base_shape,
                in_features.as_deref(),
                out_features.as_deref(),
                range,
                diagnostics,
                source,
            ),
            Self::BatchNorm { dims, num_features } => infer_batchnorm_module(
                base_shape,
                *dims,
                num_features.as_deref(),
                range,
                diagnostics,
                source,
            ),
            Self::Conv {
                dims,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => infer_conv_module(
                base_shape,
                *dims,
                in_channels.as_deref(),
                out_channels.as_deref(),
                kernel_size,
                stride,
                padding,
                dilation,
                groups.as_deref(),
                range,
                diagnostics,
                source,
            ),
            Self::Sequential(modules) => {
                let mut current = base_shape;
                for module in modules {
                    current = infer_resolved_module_shape(
                        module,
                        current,
                        vars,
                        func_map,
                        imports,
                        class_map,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        source,
                        range,
                        module_cache.as_deref_mut(),
                        module_path,
                    )?;
                }
                Some(current)
            }
            Self::Noop => Some(base_shape),
        }
    }
}

fn get_call_arg<'a>(call: &'a ExprCall, name_arg: &str, as_positional: usize) -> Option<&'a Expr> {
    call.args.get(as_positional).or_else(|| {
        call.keywords
            .iter()
            .find(|kw| kw.arg.as_deref() == Some(name_arg))
            .map(|kw| &kw.value)
    })
}

fn is_torch_nn_constructor(call_expr: &Expr, imports: &Imports) -> bool {
    torch_nn_constructor_name(call_expr, imports).is_some()
}

fn torch_nn_constructor_name(call_expr: &Expr, imports: &Imports) -> Option<String> {
    let Expr::Call(call) = call_expr else {
        return None;
    };
    match call.func.as_ref() {
        Expr::Name(name) => imports
            .from_imports
            .get(&name.id)
            .filter(|(module_name, _)| module_name == "torch.nn")
            .map(|(_, original)| original.to_string()),
        Expr::Attribute(attr) => match attr.value.as_ref() {
            Expr::Name(module_ident) => imports
                .module_aliases
                .get(&module_ident.id)
                .filter(|module_name| module_name.as_str() == "torch.nn")
                .map(|_| attr.attr.to_string()),
            Expr::Attribute(nn_attr) if nn_attr.attr.as_str() == "nn" => {
                if let Expr::Name(torch_name) = nn_attr.value.as_ref()
                    && imports.torch_aliases.contains(&torch_name.id)
                {
                    return Some(attr.attr.to_string());
                }
                None
            }
            _ => None,
        },
        _ => None,
    }
}

fn parse_module_dims(name: &str, prefix: &str) -> Option<usize> {
    name.strip_prefix(prefix)?
        .trim_end_matches('d')
        .parse::<usize>()
        .ok()
}

fn sequential_elements(call: &ExprCall) -> Vec<&Expr> {
    if call.args.len() == 1 {
        match &call.args[0] {
            Expr::Tuple(tuple) => return tuple.elts.iter().collect(),
            Expr::List(list) => return list.elts.iter().collect(),
            _ => {}
        }
    }
    call.args.iter().collect()
}

fn expr_to_tokens(expr: &Expr) -> Option<Vec<String>> {
    match expr {
        Expr::Tuple(tuple) => tuple.elts.iter().map(expr_to_token).collect(),
        Expr::List(list) => list.elts.iter().map(expr_to_token).collect(),
        other => expr_to_token(other).map(|token| vec![token]),
    }
}

fn expr_to_token(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Name(name) => Some(name.id.to_string()),
        Expr::Constant(constant) => match &constant.value {
            rustpython_parser::ast::Constant::Int(int) => Some(int.to_string()),
            rustpython_parser::ast::Constant::Float(float) => Some(float.to_string()),
            rustpython_parser::ast::Constant::Str(string) => Some(string.to_string()),
            _ => None,
        },
        Expr::Attribute(attr) => Some(attr.attr.to_string()),
        Expr::UnaryOp(unary) => match unary.op {
            rustpython_parser::ast::UnaryOp::USub => {
                expr_to_token(unary.operand.as_ref()).map(|token| format!("-{token}"))
            }
            rustpython_parser::ast::UnaryOp::UAdd => expr_to_token(unary.operand.as_ref()),
            _ => None,
        },
        Expr::BinOp(bin) => {
            let left = expr_to_token(bin.left.as_ref())?;
            let right = expr_to_token(bin.right.as_ref())?;
            let op = match bin.op {
                Operator::Add => "+",
                Operator::Sub => "-",
                Operator::Mult => "*",
                Operator::Div | Operator::FloorDiv => "/",
                _ => return None,
            };
            Some(format!("{left}{op}{right}"))
        }
        Expr::Call(call) => {
            if let Expr::Name(name) = call.func.as_ref()
                && name.id.as_str() == "int"
            {
                return call.args.first().and_then(expr_to_token);
            }
            None
        }
        _ => None,
    }
}
