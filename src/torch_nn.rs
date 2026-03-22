use crate::infer::{infer_batchnorm_module, infer_conv_module, infer_linear_module};
use crate::module_resolution::{
    ClassMap, ClassRef, FuncMap, ModuleCache, ResolvedModule, imported_class_ref,
    is_torch_nn_namespace,
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

    if let Some(class_ref) = constructor_class_ref(
        call.func.as_ref(),
        class_map,
        imports,
        module_cache.as_deref_mut(),
        module_path,
    ) {
        return Some(ResolvedModule::User(class_ref));
    }

    let constructor_name = torch_nn_constructor_name(call_expr, imports)?;
    if constructor_name == "Parameter" {
        return None;
    }
    Some(ResolvedModule::Builtin(
        TorchNNModule::from_constructor_call(
            call_expr,
            class_map,
            imports,
            module_cache,
            module_path,
        ),
    ))
}

impl TorchNNModule {
    pub(crate) fn from_constructor_call(
        call_expr: &Expr,
        class_map: &ClassMap,
        imports: &Imports,
        module_cache: Option<&mut ModuleCache>,
        module_path: Option<&Path>,
    ) -> Self {
        let Expr::Call(call) = call_expr else {
            return Self::Noop;
        };
        let Some(constructor_name) = torch_nn_constructor_name(call_expr, imports) else {
            return Self::Noop;
        };
        Self::from_named_constructor(
            call,
            &constructor_name,
            class_map,
            imports,
            module_cache,
            module_path,
        )
    }

    fn from_named_constructor(
        call: &ExprCall,
        constructor_name: &str,
        class_map: &ClassMap,
        imports: &Imports,
        mut module_cache: Option<&mut ModuleCache>,
        module_path: Option<&Path>,
    ) -> Self {
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
            return Self::Sequential(
                sequential_elements(call)
                    .into_iter()
                    .filter_map(|element| {
                        module_from_constructor_call(
                            element,
                            class_map,
                            imports,
                            module_cache.as_deref_mut(),
                            module_path,
                        )
                    })
                    .collect(),
            );
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

fn constructor_class_ref(
    constructor: &Expr,
    class_map: &ClassMap,
    imports: &Imports,
    module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> Option<ClassRef> {
    match constructor {
        Expr::Name(name) => {
            if class_map.contains_key(&name.id) {
                return Some(ClassRef {
                    name: name.id.clone(),
                    module: None,
                });
            }
            let (module_name, original) = imports.from_imports.get(&name.id)?;
            imported_class_ref(module_name, original, module_cache, module_path)
        }
        Expr::Attribute(attr) => {
            let Expr::Name(module_ident) = attr.value.as_ref() else {
                return None;
            };
            let module_name = imports.module_aliases.get(&module_ident.id)?;
            imported_class_ref(module_name, &attr.attr, module_cache, module_path)
        }
        _ => None,
    }
}

fn torch_nn_constructor_name(call_expr: &Expr, imports: &Imports) -> Option<String> {
    let Expr::Call(call) = call_expr else {
        return None;
    };
    match call.func.as_ref() {
        Expr::Name(name) => imports
            .imported_symbol_from(&name.id, "torch.nn")
            .map(ToString::to_string),
        Expr::Attribute(attr) if is_torch_nn_namespace(attr.value.as_ref(), imports) => {
            Some(attr.attr.to_string())
        }
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
