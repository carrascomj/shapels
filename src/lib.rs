use lsp_types::{Diagnostic, DiagnosticSeverity, Position, Range};
use rustpython_parser::Parse;
use rustpython_parser::ast::{self, Arguments, Expr, ExprBinOp, Identifier, Operator, Stmt, Suite};
use rustpython_parser::text_size::{TextRange, TextSize};
use std::collections::{HashMap, HashSet};
mod ops;
use ops::{infer_matmul, infer_squeeze, infer_unsqueeze, infer_view_like, shape_dims_equal};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub dtype: Option<String>,
    pub dims: Vec<String>,
}

impl Shape {
    pub fn render(&self) -> String {
        self.dims.join(" ")
    }
}

#[derive(Debug, Clone)]
pub struct HoverInfo {
    pub shape: Option<Shape>,
}

#[derive(Debug, Default)]
pub struct Analysis {
    pub diagnostics: Vec<Diagnostic>,
    pub hover_entries: Vec<(Range, HoverInfo)>,
}

#[derive(Debug, Clone, Default)]
struct VarState {
    annotated: Option<Shape>,
    inferred: Option<Shape>,
}

/// Operations that accept an argument dim (integer or sequence),
/// return a single tensor and the provided dims have been reduced
/// from the output tensor.
pub const AGGR_ALIASES: [&'static str; 21] = [
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

#[derive(Default)]
struct Imports {
    torch_aliases: HashSet<Identifier>,
    /// Maps simple function name (e.g., "mm") to all aliases in scope.
    func_aliases: HashMap<&'static str, HashSet<Identifier>>,
}

pub fn analyze_source(source: &str) -> Analysis {
    let mut analysis = Analysis::default();
    match Suite::parse(source, "<memory>") {
        Ok(module) => {
            let imports = collect_imports(&module);
            // collect function definitions first
            let mut func_map: HashMap<Identifier, (Box<Arguments>, Vec<Stmt>)> = HashMap::new();
            for stmt in &module {
                if let Stmt::FunctionDef(func) = stmt {
                    func_map.insert(func.name.clone(), (func.args.clone(), func.body.clone()));
                }
            }

            for stmt in module {
                if let Stmt::FunctionDef(func) = stmt {
                    let mut func_analysis = analyze_function(
                        &func.args,
                        &func.body,
                        source,
                        &func_map,
                        &imports,
                        &mut Vec::new(),
                    );
                    analysis.diagnostics.append(&mut func_analysis.diagnostics);
                    analysis
                        .hover_entries
                        .append(&mut func_analysis.hover_entries);
                }
            }
        }
        Err(err) => {
            analysis.diagnostics.push(Diagnostic {
                range: default_range(),
                severity: Some(DiagnosticSeverity::ERROR),
                code: None,
                code_description: None,
                source: Some("shapels".into()),
                message: format!("Parse error: {err}"),
                related_information: None,
                tags: None,
                data: None,
            });
        }
    }
    analysis
}

fn analyze_function(
    args: &Arguments,
    body: &[Stmt],
    source: &str,
    func_map: &HashMap<Identifier, (Box<Arguments>, Vec<Stmt>)>,
    imports: &Imports,
    call_stack: &mut Vec<Identifier>,
) -> Analysis {
    let (diagnostics, hover_entries, _) = simulate_function(
        args,
        body,
        source,
        func_map,
        imports,
        call_stack,
        HashMap::new(),
        true,
    );

    Analysis {
        diagnostics,
        hover_entries,
    }
}

#[allow(clippy::too_many_arguments)]
fn simulate_function(
    args: &Arguments,
    body: &[Stmt],
    source: &str,
    func_map: &HashMap<Identifier, (Box<Arguments>, Vec<Stmt>)>,
    imports: &Imports,
    call_stack: &mut Vec<Identifier>,
    mut initial_vars: HashMap<Identifier, VarState>,
    record_hovers: bool,
) -> (Vec<Diagnostic>, Vec<(Range, HoverInfo)>, Option<Shape>) {
    let mut diagnostics = Vec::new();
    let mut hover_entries = Vec::new();
    let mut vars: HashMap<Identifier, VarState> = HashMap::new();

    seed_args_from_annotations(
        args,
        source,
        &mut vars,
        &mut hover_entries,
        &mut diagnostics,
        Some(&mut initial_vars),
        record_hovers,
    );

    let mut return_shape = None;

    for stmt in body {
        match stmt {
            Stmt::AnnAssign(assign) => {
                if let Some(name) = name_from_expr(&assign.target) {
                    let ann_shape = parse_shape_annotation(&assign.annotation);
                    let range = text_range_to_lsp(expr_text_range(&assign.target), source);
                    let mut inferred = None;
                    if let Some(val) = &assign.value {
                        inferred = infer_expr_shape(
                            val,
                            &vars,
                            func_map,
                            imports,
                            call_stack,
                            &mut diagnostics,
                            &mut hover_entries,
                            record_hovers,
                            source,
                        );
                    }
                    if let (Some(ann), Some(inf)) = (ann_shape.clone(), inferred.clone())
                        && !shape_dims_equal(&ann, &inf)
                    {
                        diagnostics.push(Diagnostic {
                            range,
                            severity: Some(DiagnosticSeverity::ERROR),
                            code: None,
                            code_description: None,
                            source: Some("shapels".into()),
                            message: format!(
                                "Shape mismatch: annotation {} vs inferred {}",
                                ann.render(),
                                inf.render()
                            ),
                            related_information: None,
                            tags: None,
                            data: None,
                        });
                    }
                    let chosen_shape = ann_shape.clone().or(inferred.clone());
                    if let Some(shape) = chosen_shape {
                        vars.insert(
                            name.clone(),
                            VarState {
                                annotated: ann_shape.clone(),
                                inferred,
                            },
                        );
                        if record_hovers {
                            hover_entries.push((range, HoverInfo { shape: Some(shape) }));
                        }
                    }
                }
            }
            Stmt::Assign(assign) => {
                if assign.targets.len() == 1
                    && let Some(name) = name_from_expr(&assign.targets[0])
                {
                    let range = text_range_to_lsp(expr_text_range(&assign.targets[0]), source);
                    let diag_before = diagnostics.len();
                    let mut shape = infer_expr_shape(
                        &assign.value,
                        &vars,
                        func_map,
                        imports,
                        call_stack,
                        &mut diagnostics,
                        &mut hover_entries,
                        record_hovers,
                        source,
                    );
                    // fallback for squeeze/unsqueeze when inference failed
                    if shape.is_none()
                        && diagnostics.len() == diag_before
                        && let Expr::Call(call) = &*assign.value
                        && let Expr::Attribute(attr) = call.func.as_ref()
                    {
                        let attr_name: &str = attr.attr.as_ref();
                        if attr_name == "squeeze" {
                            let (base, dim_arg) = if is_torch_base(&attr.value, imports) {
                                let base = call.args.first();
                                let dim_kw = call
                                    .keywords
                                    .iter()
                                    .find(|kw| kw.arg.as_deref() == Some("dim"))
                                    .map(|kw| &kw.value);
                                let dim_pos = call.args.get(1);
                                (base, dim_kw.or(dim_pos))
                            } else {
                                let dim_kw = call
                                    .keywords
                                    .iter()
                                    .find(|kw| kw.arg.as_deref() == Some("dim"))
                                    .map(|kw| &kw.value);
                                let dim_pos = call.args.first();
                                (Some(attr.value.as_ref()), dim_kw.or(dim_pos))
                            };
                            if let Some(base_expr) = base {
                                shape = infer_squeeze(
                                    base_expr,
                                    dim_arg,
                                    &vars,
                                    func_map,
                                    imports,
                                    call_stack,
                                    &mut diagnostics,
                                    &mut hover_entries,
                                    record_hovers,
                                    source,
                                    call.range,
                                    true,
                                );
                            }
                        } else if attr_name == "unsqueeze" {
                            shape = infer_unsqueeze(
                                &attr.value,
                                call.args.first(),
                                &vars,
                                func_map,
                                imports,
                                call_stack,
                                &mut diagnostics,
                                &mut hover_entries,
                                record_hovers,
                                source,
                            );
                        }
                    }
                    if let Some(shape) = shape {
                        vars.insert(
                            name.clone(),
                            VarState {
                                annotated: None,
                                inferred: Some(shape.clone()),
                            },
                        );
                        if record_hovers {
                            hover_entries.push((range, HoverInfo { shape: Some(shape) }));
                        }
                    }
                }
            }
            Stmt::Return(ret) => {
                if let Some(val) = &ret.value {
                    return_shape = infer_expr_shape(
                        val,
                        &vars,
                        func_map,
                        imports,
                        call_stack,
                        &mut diagnostics,
                        &mut hover_entries,
                        record_hovers,
                        source,
                    )
                    .or(return_shape);
                    if record_hovers && let Some(shape) = return_shape.clone() {
                        let range = text_range_to_lsp(expr_text_range(val), source);
                        hover_entries.push((range, HoverInfo { shape: Some(shape) }));
                    }
                }
            }
            _ => {}
        }
    }

    (diagnostics, hover_entries, return_shape)
}

#[allow(clippy::too_many_arguments)]
fn infer_expr_shape(
    expr: &Expr,
    vars: &HashMap<Identifier, VarState>,
    func_map: &HashMap<Identifier, (Box<Arguments>, Vec<Stmt>)>,
    imports: &Imports,
    call_stack: &mut Vec<Identifier>,
    diagnostics: &mut Vec<Diagnostic>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    record_hovers: bool,
    source: &str,
) -> Option<Shape> {
    match expr {
        Expr::BinOp(ExprBinOp {
            left,
            op,
            right,
            range: expr_range,
        }) => {
            if matches!(op, Operator::MatMult) {
                return infer_matmul_shapes(
                    left,
                    right,
                    vars,
                    func_map,
                    imports,
                    call_stack,
                    diagnostics,
                    hover_entries,
                    record_hovers,
                    source,
                    *expr_range,
                );
            }
            None
        }
        Expr::Call(call) => {
            if let Expr::Name(func_name) = call.func.as_ref() {
                if is_alias_of("mm", &func_name.id, imports)
                    && let (Some(arg0), Some(arg1)) = (call.args.first(), call.args.get(1))
                {
                    return infer_matmul_shapes(
                        arg0,
                        arg1,
                        vars,
                        func_map,
                        imports,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        call.range,
                    );
                }
                if (is_alias_of("view", &func_name.id, imports)
                    || is_alias_of("reshape", &func_name.id, imports))
                    && let Some(arg0) = call.args.first()
                {
                    let args = call.args.iter().skip(1).collect::<Vec<_>>();
                    return infer_view_like(
                        arg0,
                        &args,
                        vars,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        call.range,
                    );
                }
                if is_alias_of("unsqueeze", &func_name.id, imports)
                    && let Some(arg0) = call.args.first()
                {
                    return infer_unsqueeze(
                        arg0,
                        call.args.get(1),
                        vars,
                        func_map,
                        imports,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                    );
                }
                if is_alias_of("squeeze", &func_name.id, imports)
                    && let Some(arg0) = call.args.first()
                {
                    return infer_squeeze(
                        arg0,
                        call.args.get(1),
                        vars,
                        func_map,
                        imports,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        call.range,
                        true,
                    );
                }
                if is_alias_of("sum", &func_name.id, imports)
                    && let Some(arg0) = call.args.first()
                {
                    let dim_arg = call
                        .keywords
                        .iter()
                        .find(|kw| kw.arg.as_deref() == Some("dim"))
                        .map(|kw| &kw.value)
                        .or_else(|| call.args.get(1));
                    return infer_squeeze(
                        arg0,
                        dim_arg,
                        vars,
                        func_map,
                        imports,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        call.range,
                        false,
                    );
                }
                if let Some((callee_args, callee_body)) = func_map.get(&func_name.id) {
                    // avoid infinite recursion
                    if call_stack.iter().any(|id| id == &func_name.id) {
                        return None;
                    }
                    // build argument binding map
                    let mut arg_shapes: HashMap<Identifier, VarState> = HashMap::new();
                    for (idx, param) in callee_args.args.iter().enumerate() {
                        if let Some(arg_expr) = call.args.get(idx)
                            && let Some(shape) = infer_expr_shape(
                                arg_expr,
                                vars,
                                func_map,
                                imports,
                                call_stack,
                                diagnostics,
                                hover_entries,
                                record_hovers,
                                source,
                            )
                        {
                            arg_shapes.insert(
                                param.def.arg.clone(),
                                VarState {
                                    annotated: None,
                                    inferred: Some(shape),
                                },
                            );
                        }
                    }
                    call_stack.push(func_name.id.clone());
                    let (mut diag, mut hovers, ret_shape) = simulate_function(
                        callee_args.as_ref(),
                        callee_body,
                        source,
                        func_map,
                        imports,
                        call_stack,
                        arg_shapes,
                        false,
                    );
                    diagnostics.append(&mut diag);
                    if record_hovers {
                        hover_entries.append(&mut hovers);
                    }
                    call_stack.pop();
                    return ret_shape;
                }
            }
            if let Expr::Attribute(attr) = call.func.as_ref() {
                let attr_name: &str = attr.attr.as_ref();
                if attr_name == "mm"
                    && is_torch_base(&attr.value, imports)
                    && let (Some(arg0), Some(arg1)) = (call.args.first(), call.args.get(1))
                {
                    return infer_matmul_shapes(
                        arg0,
                        arg1,
                        vars,
                        func_map,
                        imports,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        call.range,
                    );
                }
                if attr_name == "view" || attr_name == "reshape" {
                    if lookup_shape(&attr.value, vars, hover_entries, record_hovers, source)
                        .is_some()
                    {
                        let args = call.args.iter().collect::<Vec<_>>();
                        return infer_view_like(
                            &attr.value,
                            &args,
                            vars,
                            diagnostics,
                            hover_entries,
                            record_hovers,
                            source,
                            call.range,
                        );
                    }
                } else if attr_name == "unsqueeze" {
                    return infer_unsqueeze(
                        &attr.value,
                        call.args.first(),
                        vars,
                        func_map,
                        imports,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                    );
                } else if attr_name == "squeeze" {
                    // tensor.squeeze(...) vs torch.squeeze(tensor, ...)
                    let (base, dim_arg) = if is_torch_base(&attr.value, imports) {
                        // first positional is the tensor, dim is 2nd positional or keyword
                        let base = call.args.first()?;
                        let dim_kw = call
                            .keywords
                            .iter()
                            .find(|kw| kw.arg.as_deref() == Some("dim"))
                            .map(|kw| &kw.value);
                        let dim_pos = call.args.get(1);
                        (base, dim_kw.or(dim_pos))
                    } else {
                        let base = attr.value.as_ref();
                        let dim_kw = call
                            .keywords
                            .iter()
                            .find(|kw| kw.arg.as_deref() == Some("dim"))
                            .map(|kw| &kw.value);
                        let dim_pos = call.args.first();
                        (base, dim_kw.or(dim_pos))
                    };
                    return infer_squeeze(
                        base,
                        dim_arg,
                        vars,
                        func_map,
                        imports,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        call.range,
                        true,
                    );
                } else if AGGR_ALIASES.contains(&attr_name) {
                    // tensor.sum(...) vs torch.sum(tensor, ...)
                    let (base, dim_source) = if is_torch_base(&attr.value, imports) {
                        let base = call.args.first()?;
                        let dim_kw = call
                            .keywords
                            .iter()
                            .find(|kw| kw.arg.as_deref() == Some("dim"))
                            .map(|kw| &kw.value);
                        let dim_pos = call.args.get(1);
                        (base, dim_kw.or(dim_pos))
                    } else {
                        let base = attr.value.as_ref();
                        let dim_kw = call
                            .keywords
                            .iter()
                            .find(|kw| kw.arg.as_deref() == Some("dim"))
                            .map(|kw| &kw.value);
                        let dim_pos = call.args.first();
                        (base, dim_kw.or(dim_pos))
                    };
                    return infer_squeeze(
                        base,
                        dim_source,
                        vars,
                        func_map,
                        imports,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        call.range,
                        false,
                    );
                }
            }
            None
        }
        Expr::Name(expr_name) => {
            let shape = vars
                .get(&expr_name.id)
                .and_then(|v| v.annotated.clone().or_else(|| v.inferred.clone()));
            if record_hovers && let Some(s) = shape.clone() {
                let range = text_range_to_lsp(expr_text_range(expr), source);
                hover_entries.push((range, HoverInfo { shape: Some(s) }));
            }
            shape
        }
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_matmul_shapes(
    left: &Expr,
    right: &Expr,
    vars: &HashMap<Identifier, VarState>,
    func_map: &HashMap<Identifier, (Box<Arguments>, Vec<Stmt>)>,
    imports: &Imports,
    call_stack: &mut Vec<Identifier>,
    diagnostics: &mut Vec<Diagnostic>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    record_hovers: bool,
    source: &str,
    whole_range: TextRange,
) -> Option<Shape> {
    let left_shape = lookup_shape(left, vars, hover_entries, record_hovers, source).or_else(|| {
        infer_expr_shape(
            left,
            vars,
            func_map,
            imports,
            call_stack,
            diagnostics,
            hover_entries,
            false,
            source,
        )
    });
    let right_shape =
        lookup_shape(right, vars, hover_entries, record_hovers, source).or_else(|| {
            infer_expr_shape(
                right,
                vars,
                func_map,
                imports,
                call_stack,
                diagnostics,
                hover_entries,
                false,
                source,
            )
        });
    match (left_shape, right_shape) {
        (Some(l), Some(r)) => match infer_matmul(&l, &r) {
            Ok(shape) => Some(shape),
            Err(msg) => {
                diagnostics.push(Diagnostic {
                    range: text_range_to_lsp(whole_range, source),
                    severity: Some(DiagnosticSeverity::ERROR),
                    code: None,
                    code_description: None,
                    source: Some("shapels".into()),
                    message: msg,
                    related_information: None,
                    tags: None,
                    data: None,
                });
                None
            }
        },
        _ => None,
    }
}

fn lookup_shape(
    expr: &Expr,
    vars: &HashMap<Identifier, VarState>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    record_hovers: bool,
    source: &str,
) -> Option<Shape> {
    match expr {
        Expr::Name(expr_name) => {
            let shape = vars
                .get(&expr_name.id)
                .and_then(|v| v.annotated.clone().or_else(|| v.inferred.clone()));
            if record_hovers && let Some(s) = shape.clone() {
                let range = text_range_to_lsp(expr_text_range(expr), source);
                hover_entries.push((range, HoverInfo { shape: Some(s) }));
            }
            shape
        }
        _ => None,
    }
}

fn is_torch_base(expr: &Expr, imports: &Imports) -> bool {
    match expr {
        Expr::Name(n) => n.id.to_string() == "torch" || imports.torch_aliases.contains(&n.id),
        _ => false,
    }
}

fn is_alias_of(canonical: &str, ident: &Identifier, imports: &Imports) -> bool {
    imports
        .func_aliases
        .get(canonical)
        .map(|set| set.contains(ident))
        .unwrap_or(false)
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
fn collect_imports(module: &[Stmt]) -> Imports {
    let mut imports = Imports::default();
    // seed known function names
    for fname in ["mm", "view", "reshape", "sum"] {
        imports
            .func_aliases
            .entry(fname)
            .or_insert_with(HashSet::new);
    }
    imports
        .torch_aliases
        .insert(Identifier::from("torch".to_string()));

    for stmt in module {
        match stmt {
            Stmt::Import(import) => {
                for alias in &import.names {
                    let name = alias.name.as_str();
                    let as_id = alias
                        .asname
                        .clone()
                        .unwrap_or_else(|| Identifier::from(name));
                    if name == "torch" {
                        imports.torch_aliases.insert(as_id.clone());
                    }
                    if let Some(val) = imports.func_aliases.get_mut(name) {
                        val.insert(as_id);
                    } else if AGGR_ALIASES.contains(&name) {
                        imports.func_aliases.entry("sum").or_default().insert(as_id);
                    }
                }
            }
            Stmt::ImportFrom(f) => {
                if let Some(module) = &f.module
                    && module == "torch"
                {
                    for alias in &f.names {
                        let name = alias.name.as_str();
                        if let Some(val) = imports.func_aliases.get_mut(name) {
                            let id = alias
                                .asname
                                .clone()
                                .unwrap_or_else(|| Identifier::from(name));
                            val.insert(id);
                        } else if AGGR_ALIASES.contains(&name) {
                            let id = alias
                                .asname
                                .clone()
                                .unwrap_or_else(|| Identifier::from(name));
                            imports.func_aliases.entry("sum").or_default().insert(id);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    imports
}

fn parse_shape_annotation(expr: &Expr) -> Option<Shape> {
    if let Expr::Subscript(sub) = expr {
        let dtype = name_like(&sub.value);
        let components: Vec<&Expr> = match &*sub.slice {
            ast::Expr::Tuple(t) => t.elts.iter().collect(),
            other => vec![other],
        };
        if components.len() >= 2
            && let Some(raw) = string_literal_value(components[1])
        {
            let dims = normalize_shape_tokens(&raw);
            return Some(Shape { dtype, dims });
        }
    }
    None
}

fn name_like(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Name(name) => Some(name.id.to_string()),
        Expr::Attribute(attr) => {
            let mut base = name_like(&attr.value)?;
            base.push('.');
            base.push_str(attr.attr.as_ref());
            Some(base)
        }
        _ => None,
    }
}

fn string_literal_value(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Constant(c) => {
            if let ast::Constant::Str(s) = &c.value {
                Some(s.clone())
            } else {
                None
            }
        }
        Expr::JoinedStr(js) => {
            let mut buf = String::new();
            for val in &js.values {
                if let Expr::Constant(c) = val
                    && let ast::Constant::Str(s) = &c.value
                {
                    buf.push_str(s);
                }
            }
            if buf.is_empty() { None } else { Some(buf) }
        }
        _ => None,
    }
}

fn normalize_shape_tokens(raw: &str) -> Vec<String> {
    raw.split_whitespace()
        .map(|s| s.trim_matches('"').to_string())
        .collect()
}

fn name_from_expr(expr: &Expr) -> Option<Identifier> {
    match expr {
        Expr::Name(n) => Some(n.id.clone()),
        _ => None,
    }
}

fn seed_args_from_annotations(
    args: &Arguments,
    source: &str,
    vars: &mut HashMap<Identifier, VarState>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    diagnostics: &mut Vec<Diagnostic>,
    provided: Option<&mut HashMap<Identifier, VarState>>,
    record_hovers: bool,
) {
    for arg in &args.args {
        let ann_shape = arg
            .def
            .annotation
            .as_ref()
            .and_then(|expr| parse_shape_annotation(expr.as_ref()));
        let range = text_range_to_lsp(arg.def.range, source);
        let mut provided_state = provided.as_ref().and_then(|p| p.get(&arg.def.arg)).cloned();

        if let (Some(ann), Some(inf_state)) = (
            ann_shape.clone(),
            provided_state.as_ref().and_then(|s| s.inferred.clone()),
        ) && ann.dims != inf_state.dims
        {
            diagnostics.push(Diagnostic {
                range,
                severity: Some(DiagnosticSeverity::ERROR),
                code: None,
                code_description: None,
                source: Some("shapels".into()),
                message: format!(
                    "Shape mismatch: annotation {} vs inferred {}",
                    ann.render(),
                    inf_state.render()
                ),
                related_information: None,
                tags: None,
                data: None,
            });
        }

        let state = VarState {
            annotated: ann_shape.clone(),
            inferred: provided_state.take().and_then(|s| s.inferred).or(None),
        };

        if state.annotated.is_some() || state.inferred.is_some() {
            vars.insert(arg.def.arg.clone(), state.clone());
            if record_hovers {
                hover_entries.push((
                    range,
                    HoverInfo {
                        shape: state.annotated.or(state.inferred),
                    },
                ));
            }
        }
    }
}

fn text_range_to_lsp(range: TextRange, source: &str) -> Range {
    Range {
        start: offset_to_position(source, range.start().to_usize()),
        end: offset_to_position(source, range.end().to_usize()),
    }
}

fn expr_text_range(expr: &Expr) -> TextRange {
    match expr {
        Expr::Name(n) => n.range,
        Expr::BinOp(b) => b.range,
        Expr::Call(c) => c.range,
        Expr::Subscript(s) => s.range,
        Expr::Attribute(a) => a.range,
        Expr::Constant(c) => c.range,
        Expr::JoinedStr(j) => j.range,
        Expr::Tuple(t) => t.range,
        _ => TextRange::new(TextSize::from(0), TextSize::from(0)),
    }
}

fn offset_to_position(source: &str, offset: usize) -> Position {
    let mut line = 0u32;
    let mut col = 0u32;
    let mut count = 0usize;
    for ch in source.chars() {
        if count == offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
        count += ch.len_utf8();
    }
    Position {
        line,
        character: col,
    }
}

fn default_range() -> Range {
    Range {
        start: Position {
            line: 0,
            character: 0,
        },
        end: Position {
            line: 0,
            character: 1,
        },
    }
}

impl Analysis {
    pub fn hover(&self, position: Position) -> Option<&HoverInfo> {
        // Prefer exact containment with smallest span.
        if let Some((_, info)) = self
            .hover_entries
            .iter()
            .filter(|(range, _)| within(range, &position))
            .min_by_key(|(range, _)| range_span(range, &position))
        {
            return Some(info);
        }

        // Fallback: nearest entry on the same line to the left.
        if let Some((_, info)) = self
            .hover_entries
            .iter()
            .filter(|(range, _)| range.start.line == position.line)
            .min_by_key(|(range, _)| {
                let a = range.start.character as i64;
                let b = position.character as i64;
                (a - b).abs()
            })
        {
            return Some(info);
        }
        None
    }
}

fn within(range: &Range, pos: &Position) -> bool {
    (pos.line > range.start.line
        || (pos.line == range.start.line && pos.character >= range.start.character))
        && (pos.line < range.end.line
            || (pos.line == range.end.line && pos.character <= range.end.character))
}

fn range_span(range: &Range, _pos: &Position) -> u32 {
    // Prioritize entries that wrap the position tightly.
    let line_span = (range.end.line as i64 - range.start.line as i64).unsigned_abs() as u32;
    let char_span = if range.start.line == range.end.line {
        (range.end.character as i64 - range.start.character as i64).unsigned_abs() as u32
    } else {
        1000
    };
    line_span * 1000 + char_span
}
