use lsp_types::{Diagnostic, DiagnosticSeverity, Position, Range};
use rustpython_parser::Parse;
use rustpython_parser::ast::{
    self, Arguments, Expr, ExprBinOp, ExprCall, Identifier, Operator, Stmt, Suite,
};
use rustpython_parser::text_size::{TextRange, TextSize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
mod infer;
mod op_groups;
use crate::infer::{
    Transpose, infer_broadcastable_poswise, infer_matmul_shapes, infer_noop, infer_permute,
    infer_squeeze, infer_unsqueeze, infer_view_like, shape_dims_equal,
};
pub use crate::op_groups::AGGR_ALIASES;
use crate::op_groups::{NOOP_ALIASES, NOOP_DIM_ALIASES};

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

#[derive(Debug, Clone, Default)]
struct VarState {
    annotated: Option<Shape>,
    inferred: Option<Shape>,
}

#[derive(Default, Clone)]
struct Imports {
    torch_aliases: HashSet<Identifier>,
    /// Maps simple function name (e.g., "mm") to all aliases in scope.
    func_aliases: HashMap<&'static str, HashSet<Identifier>>,
    /// Module alias mapping for `import foo as bar` style.
    module_aliases: HashMap<Identifier, String>,
    /// Symbol imports mapping alias -> (module, original name).
    from_imports: HashMap<Identifier, (String, Identifier)>,
}

#[derive(Clone)]
pub(crate) struct CachedModule {
    path: PathBuf,
    source: String,
    func_map: HashMap<Identifier, (Box<Arguments>, Vec<Stmt>)>,
    imports: Imports,
}

pub(crate) struct ModuleCache {
    modules: HashMap<String, CachedModule>,
    project_root: Option<PathBuf>,
}

impl ModuleCache {
    fn new(current_file: &Path) -> Self {
        let project_root = find_project_root(current_file);
        Self {
            modules: HashMap::new(),
            project_root,
        }
    }

    fn project_root(&self) -> Option<&Path> {
        self.project_root.as_deref()
    }

    /// Return a cloned module entry, loading and parsing it if necessary.
    fn get_module(&mut self, module_name: &str, current_file: &Path) -> Option<CachedModule> {
        if let Some(cached) = self.modules.get(module_name) {
            return Some(cached.clone());
        }
        let path = resolve_module_path(module_name, current_file, self.project_root.as_deref())?;
        let source = fs::read_to_string(&path).ok()?;
        let module = Suite::parse(&source, module_name).ok()?;
        let imports = collect_imports(&module, Some(&path), self.project_root());
        let mut func_map: HashMap<Identifier, (Box<Arguments>, Vec<Stmt>)> = HashMap::new();
        for stmt in &module {
            if let Stmt::FunctionDef(func) = stmt {
                func_map.insert(func.name.clone(), (func.args.clone(), func.body.clone()));
            }
        }
        let cached = CachedModule {
            path,
            source,
            func_map,
            imports,
        };
        self.modules.insert(module_name.to_string(), cached.clone());
        Some(cached)
    }
}

fn find_project_root(start: &Path) -> Option<PathBuf> {
    let mut dir = start.parent();
    let home = std::env::var_os("HOME").map(PathBuf::from);
    while let Some(current) = dir {
        if current.join(".git").exists()
            || current.join("pyproject.toml").exists()
            || current.join("setup.py").exists()
            || current.join("setup.cfg").exists()
        {
            return Some(current.to_path_buf());
        }
        if Some(current.to_path_buf()) == home {
            break;
        }
        dir = current.parent();
    }
    None
}

fn resolve_module_path(
    module: &str,
    current_file: &Path,
    project_root: Option<&Path>,
) -> Option<PathBuf> {
    let mut search_roots: Vec<PathBuf> = Vec::new();

    if let Some(parent) = current_file.parent() {
        search_roots.push(parent.to_path_buf());
    }
    if let Some(prj) = project_root {
        search_roots.push(prj.to_path_buf());
        let src_dir = prj.join("src");
        if src_dir.exists() {
            search_roots.push(src_dir);
        }
    }

    // Search for virtual environment markers in PATH.
    if let Ok(path_var) = std::env::var("PATH") {
        for entry in path_var.split(':') {
            if !entry.contains(".venv") {
                continue;
            }
            let mut p = PathBuf::from(entry);
            while let Some(parent) = p.parent() {
                if let Some(name) = parent.file_name()
                    && name.to_string_lossy().contains(".venv")
                {
                    let venv_dir = parent.to_path_buf();
                    // parent of .venv might be the project root
                    if let Some(parent_parent) = venv_dir.parent() {
                        search_roots.push(parent_parent.to_path_buf());
                    }
                    // site-packages paths
                    let lib_dir = venv_dir.join("lib");
                    if lib_dir.exists() {
                        if let Ok(entries) = fs::read_dir(&lib_dir) {
                            for entry in entries.flatten() {
                                let fname = entry.file_name();
                                if fname.to_string_lossy().starts_with("python") {
                                    let sp = entry.path().join("site-packages");
                                    if sp.exists() {
                                        search_roots.push(sp);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                p = parent.to_path_buf();
            }
        }
    }

    // Convert module name to path components.
    let parts: Vec<&str> = module.split('.').collect();
    for root in search_roots {
        let mut base = root.clone();
        for part in &parts {
            base.push(part);
        }
        let file_candidate = base.with_extension("py");
        if file_candidate.exists() {
            return Some(file_candidate);
        }
        let init_candidate = base.join("__init__.py");
        if init_candidate.exists() {
            return Some(init_candidate);
        }
        // Heuristic: if root already points at the top-level package (e.g., root ends with parts[0]),
        // try resolving without repeating the first component to avoid example_python/example_python duplication.
        if let Some(root_name) = root.file_name()
            && root_name
                == parts
                    .first()
                    .map(|s| std::ffi::OsStr::new(s))
                    .unwrap_or_else(|| std::ffi::OsStr::new(""))
            && parts.len() > 1
        {
            let mut base = root.clone();
            for part in parts.iter().skip(1) {
                base.push(part);
            }
            let file_candidate = base.with_extension("py");
            if file_candidate.exists() {
                return Some(file_candidate);
            }
            let init_candidate = base.join("__init__.py");
            if init_candidate.exists() {
                return Some(init_candidate);
            }
        }
    }
    None
}

pub fn analyze_source(source: &str) -> Analysis {
    analyze_source_internal(source, None, None)
}

/// Analyze in-memory source but anchored at a file path so imports can resolve.
pub fn analyze_source_at_path(source: &str, path: &Path) -> Analysis {
    let mut cache = ModuleCache::new(path);
    analyze_source_internal(source, Some(path), Some(&mut cache))
}

/// Analyze a python file with module resolution enabled.
pub fn analyze_file(path: &Path) -> Analysis {
    match fs::read_to_string(path) {
        Ok(src) => {
            let mut cache = ModuleCache::new(path);
            analyze_source_internal(&src, Some(path), Some(&mut cache))
        }
        Err(err) => Analysis {
            diagnostics: vec![Diagnostic {
                range: default_range(),
                severity: Some(DiagnosticSeverity::ERROR),
                code: None,
                code_description: None,
                source: Some("shapels".into()),
                message: format!("Failed to read file: {err}"),
                related_information: None,
                tags: None,
                data: None,
            }],
            hover_entries: Vec::new(),
        },
    }
}

fn analyze_source_internal<'a>(
    source: &'a str,
    current_path: Option<&'a Path>,
    mut module_cache: Option<&mut ModuleCache>,
) -> Analysis {
    let mut analysis = Analysis::default();
    let parse_name = current_path
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "<memory>".to_string());
    match Suite::parse(source, &parse_name) {
        Ok(module) => {
            let imports = collect_imports(
                &module,
                current_path,
                module_cache.as_deref().and_then(|c| c.project_root()),
            );
            // collect function definitions first
            let mut func_map: HashMap<Identifier, (Box<Arguments>, Vec<Stmt>)> = HashMap::new();
            for stmt in &module {
                if let Stmt::FunctionDef(func) = stmt {
                    func_map.insert(func.name.clone(), (func.args.clone(), func.body.clone()));
                }
            }

            // analyze top-level statements (outside functions)
            let empty_args = Arguments {
                range: ast::OptionalRange::from(TextRange::new(
                    TextSize::from(0),
                    TextSize::from(0),
                )),
                posonlyargs: Vec::new(),
                args: Vec::new(),
                vararg: None,
                kwonlyargs: Vec::new(),
                kwarg: None,
            };
            let (mut top_diags, mut top_hovers, _) = simulate_function(
                &empty_args,
                &module,
                source,
                &func_map,
                &imports,
                &mut Vec::new(),
                HashMap::new(),
                true,
                module_cache.as_deref_mut(),
                current_path,
            );
            analysis.diagnostics.append(&mut top_diags);
            analysis.hover_entries.append(&mut top_hovers);

            for stmt in &module {
                if let Stmt::FunctionDef(func) = stmt {
                    let mut func_analysis = analyze_function(
                        &func.args,
                        &func.body,
                        source,
                        &func_map,
                        &imports,
                        &mut Vec::new(),
                        module_cache.as_deref_mut(),
                        current_path,
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
    module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
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
        module_cache,
        module_path,
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
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
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
                if assignment_shape_checks(
                    &assign.target,
                    assign.value.as_deref().unwrap_or(assign.target.as_ref()),
                    &mut vars,
                    &mut diagnostics,
                    &mut hover_entries,
                    record_hovers,
                    source,
                ) {
                    continue;
                }
                if let Some(name) = name_from_expr(&assign.target) {
                    let ann_shape = parse_shape_annotation(&assign.annotation);
                    let range = text_range_to_lsp(expr_text_range(&assign.target), source);
                    let mut inferred = None;
                    if let Some(val) = &assign.value {
                        if assignment_shape_checks(
                            val,
                            val,
                            &mut vars,
                            &mut diagnostics,
                            &mut hover_entries,
                            record_hovers,
                            source,
                        ) {
                            inferred = vars
                                .get(&name)
                                .and_then(|v| v.annotated.clone().or(v.inferred.clone()));
                        } else {
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
                                module_cache.as_deref_mut(),
                                module_path,
                            );
                        }
                    }
                    if let (Some(ann), Some(inf)) = (ann_shape.clone(), inferred.clone())
                        && !shape_dims_equal(&ann, &inf)
                    {
                        // If annotation is a shape-unroll, treat it as a rename rather than mismatch.
                        if let Expr::Subscript(_sub) = &*assign.annotation
                            && ann.dims.len() == inf.dims.len()
                            && matches!(
                                assign.value.as_deref(),
                                Some(Expr::Name(_)) | Some(Expr::Attribute(_))
                            )
                        {
                            let mut renamed = inf.clone();
                            renamed.dims = ann.dims.clone();
                            vars.insert(
                                name.clone(),
                                VarState {
                                    annotated: ann_shape.clone(),
                                    inferred: Some(renamed.clone()),
                                },
                            );
                            if record_hovers {
                                hover_entries.push((
                                    range,
                                    HoverInfo {
                                        shape: Some(renamed),
                                    },
                                ));
                            }
                            continue;
                        }
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
                // handle tuple destructuring of `.shape`
                if assign.targets.len() == 1
                    && assignment_shape_checks(
                        &assign.targets[0],
                        &assign.value,
                        &mut vars,
                        &mut diagnostics,
                        &mut hover_entries,
                        record_hovers,
                        source,
                    )
                {
                    continue;
                }
                if assign.targets.len() == 1
                    && let Some(name) = name_from_expr(&assign.targets[0])
                {
                    let range = text_range_to_lsp(expr_text_range(&assign.targets[0]), source);
                    let shape = infer_expr_shape(
                        &assign.value,
                        &vars,
                        func_map,
                        imports,
                        call_stack,
                        &mut diagnostics,
                        &mut hover_entries,
                        record_hovers,
                        source,
                        module_cache.as_deref_mut(),
                        module_path,
                    );
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
                        module_cache.as_deref_mut(),
                        module_path,
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
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> Option<Shape> {
    match expr {
        Expr::BinOp(ExprBinOp {
            left,
            op,
            right,
            range: expr_range,
        }) => match op {
            Operator::Mult | Operator::Add | Operator::Sub | Operator::Div => {
                return infer_broadcastable_poswise(
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
                    module_cache.as_deref_mut(),
                    module_path,
                );
            }
            Operator::MatMult => {
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
                    module_cache.as_deref_mut(),
                    module_path,
                );
            }
            _ => return None,
        },
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
                        module_cache.as_deref_mut(),
                        module_path,
                    );
                }
                if let Some((module_name, original)) = imports.from_imports.get(&func_name.id) {
                    if let (Some(cache), Some(cur_path)) =
                        (module_cache.as_deref_mut(), module_path)
                        && let Some(module) = cache.get_module(module_name, cur_path)
                    {
                        if let Some((callee_args, callee_body)) = module.func_map.get(original) {
                            // avoid infinite recursion
                            if call_stack.iter().any(|id| id == &func_name.id) {
                                return None;
                            }
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
                                        module_cache.as_deref_mut(),
                                        module_path,
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
                                &module.source,
                                &module.func_map,
                                &module.imports,
                                call_stack,
                                arg_shapes,
                                false,
                                module_cache.as_deref_mut(),
                                Some(module.path.as_path()),
                            );
                            diagnostics.append(&mut diag);
                            if record_hovers {
                                hover_entries.append(&mut hovers);
                            }
                            call_stack.pop();
                            return ret_shape;
                        }
                    }
                }
                if (is_alias_of("view", &func_name.id, imports)
                    || is_alias_of("reshape", &func_name.id, imports))
                    && let Some(arg0) = call.args.first()
                {
                    let args = call.args.iter().skip(1).collect::<Vec<_>>();
                    return infer_view_like(
                        arg0,
                        &args,
                        None,
                        vars,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        call.range,
                    );
                } else if is_alias_of("permute", &func_name.id, imports)
                    && let Some(base) = call.args.first()
                {
                    let order_args: Vec<&Expr> = if call.args.len() >= 2 {
                        if let Some(Expr::Tuple(t)) = call.args.get(1) {
                            t.elts.iter().collect()
                        } else {
                            call.args.iter().skip(1).collect()
                        }
                    } else {
                        Vec::new()
                    };
                    return infer_permute(
                        base,
                        &order_args,
                        Transpose::Permute,
                        None,
                        vars,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        call.range,
                    );
                } else if is_alias_of("transpose", &func_name.id, imports)
                    && let Some(base) = call.args.first()
                {
                    let order_args: Vec<&Expr> = if call.args.len() >= 3 {
                        call.args.iter().skip(1).take(2).collect()
                    } else if call.args.len() == 2 {
                        if let Some(Expr::Tuple(t)) = call.args.get(1) {
                            t.elts.iter().collect()
                        } else {
                            call.args.iter().skip(1).collect()
                        }
                    } else {
                        Vec::new()
                    };
                    return infer_permute(
                        base,
                        &order_args,
                        Transpose::Transpose,
                        None,
                        vars,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        call.range,
                    );
                } else if is_alias_of("t", &func_name.id, imports)
                    && let Some(base) = call.args.first()
                {
                    let order_args: Vec<&Expr> = if call.args.len() >= 3 {
                        call.args.iter().skip(1).take(2).collect()
                    } else if call.args.len() == 2 {
                        if let Some(Expr::Tuple(t)) = call.args.get(1) {
                            t.elts.iter().collect()
                        } else {
                            call.args.iter().skip(1).collect()
                        }
                    } else {
                        Vec::new()
                    };
                    return infer_permute(
                        base,
                        &order_args,
                        Transpose::T,
                        None,
                        vars,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        call.range,
                    );
                } else if is_alias_of("unsqueeze", &func_name.id, imports)
                    && let Some(arg0) = call.args.first()
                {
                    return infer_unsqueeze(
                        arg0,
                        get_arg(call, "dim", 1),
                        vars,
                        func_map,
                        imports,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        module_cache.as_deref_mut(),
                        module_path,
                    );
                } else if is_alias_of("squeeze", &func_name.id, imports)
                    && let Some(arg0) = call.args.first()
                {
                    return infer_squeeze(
                        arg0,
                        get_arg(call, "dim", 1),
                        vars,
                        func_map,
                        imports,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        call.range,
                        module_cache.as_deref_mut(),
                        module_path,
                        true,
                    );
                } else if is_alias_of("sum", &func_name.id, imports)
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
                        module_cache.as_deref_mut(),
                        module_path,
                        false,
                    );
                } else if is_alias_of("softmax", &func_name.id, imports) {
                    let base_hint = infer_expr_shape(
                        call.args.first()?,
                        vars,
                        func_map,
                        imports,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        false,
                        source,
                        module_cache.as_deref_mut(),
                        module_path,
                    );
                    return infer_noop(
                        base_hint,
                        get_arg(&call, "dim", 1),
                        diagnostics,
                        source,
                        call.range,
                        // TODO(carrascomj): hack: just treat argsort different
                        func_name.id.contains("soft"),
                    );
                } else if is_alias_of("noop", &func_name.id, imports) {
                    return infer_expr_shape(
                        call.args.first()?,
                        vars,
                        func_map,
                        imports,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        false,
                        source,
                        module_cache.as_deref_mut(),
                        module_path,
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
                                module_cache.as_deref_mut(),
                                module_path,
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
                        module_cache.as_deref_mut(),
                        module_path,
                    );
                    diagnostics.append(&mut diag);
                    if record_hovers {
                        hover_entries.append(&mut hovers);
                    }
                    call_stack.pop();
                    return ret_shape;
                }
            }
            // methods are functions with attributes
            if let Expr::Attribute(attr) = call.func.as_ref() {
                let attr_name: &str = attr.attr.as_ref();
                if let Expr::Name(module_ident) = attr.value.as_ref()
                    && let Some(module_name) = imports.module_aliases.get(&module_ident.id)
                    && module_name != "torch"
                    && let (Some(cache), Some(cur_path)) =
                        (module_cache.as_deref_mut(), module_path)
                    && let Some(module) = cache.get_module(module_name, cur_path)
                    && let Some((callee_args, callee_body)) = module.func_map.get(&attr.attr)
                {
                    if call_stack.iter().any(|id| id == &attr.attr) {
                        return None;
                    }
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
                                module_cache.as_deref_mut(),
                                module_path,
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
                    call_stack.push(attr.attr.clone());
                    let (mut diag, mut hovers, ret_shape) = simulate_function(
                        callee_args.as_ref(),
                        callee_body,
                        &module.source,
                        &module.func_map,
                        &module.imports,
                        call_stack,
                        arg_shapes,
                        false,
                        module_cache.as_deref_mut(),
                        Some(module.path.as_path()),
                    );
                    diagnostics.append(&mut diag);
                    if record_hovers {
                        hover_entries.append(&mut hovers);
                    }
                    call_stack.pop();
                    return ret_shape;
                }
                // two cases: torch.ATTR_NAME(torch.Tensor, ...) or torch.Tensor.ATTR_NAME(...)
                let in_torch = is_torch_base(&attr.value, imports);
                let offset = if in_torch { 1 } else { 0 };
                if attr_name == "mm" {
                    let args = match (in_torch, call.args.get(0), call.args.get(1)) {
                        (true, Some(arg0), Some(arg1)) => Some((arg0, arg1)),
                        (false, Some(arg1), _) => Some((attr.value.as_ref(), arg1)),
                        _ => None,
                    };
                    if let Some((arg0, arg1)) = args {
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
                            module_cache.as_deref_mut(),
                            module_path,
                        );
                    }
                }
                if attr_name == "view" || attr_name == "reshape" {
                    let base_hint =
                        lookup_shape(&attr.value, vars, hover_entries, record_hovers, source)
                            .or_else(|| {
                                infer_expr_shape(
                                    &attr.value,
                                    vars,
                                    func_map,
                                    imports,
                                    call_stack,
                                    diagnostics,
                                    hover_entries,
                                    false,
                                    source,
                                    module_cache.as_deref_mut(),
                                    module_path,
                                )
                            });
                    let args = call.args.iter().collect::<Vec<_>>();
                    return infer_view_like(
                        &attr.value,
                        &args,
                        base_hint,
                        vars,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        call.range,
                    );
                } else if attr_name == "permute" {
                    let (base, order_args): (&Expr, Vec<&Expr>) =
                        if is_torch_base(&attr.value, imports) {
                            let base = call.args.first()?;
                            let rest = if call.args.len() >= 2 {
                                if let Some(Expr::Tuple(t)) = call.args.get(1) {
                                    t.elts.iter().collect()
                                } else {
                                    call.args.iter().skip(1).collect()
                                }
                            } else {
                                Vec::new()
                            };
                            (base, rest)
                        } else {
                            let rest = if call.args.len() == 1 {
                                if let Some(Expr::Tuple(t)) = call.args.first() {
                                    t.elts.iter().collect()
                                } else {
                                    call.args.iter().collect()
                                }
                            } else {
                                call.args.iter().collect()
                            };
                            (&attr.value, rest)
                        };
                    return infer_permute(
                        base,
                        &order_args,
                        Transpose::Permute,
                        None,
                        vars,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        call.range,
                    );
                } else if attr_name == "transpose" || attr_name == "t" {
                    let (base, order_args): (&Expr, Vec<&Expr>) =
                        if is_torch_base(&attr.value, imports) {
                            let base = call.args.first()?;
                            let rest = if call.args.len() >= 3 {
                                call.args.iter().skip(1).take(2).collect()
                            } else if call.args.len() == 2 {
                                if let Some(Expr::Tuple(t)) = call.args.get(1) {
                                    t.elts.iter().collect()
                                } else {
                                    call.args.iter().skip(1).collect()
                                }
                            } else {
                                Vec::new()
                            };
                            (base, rest)
                        } else {
                            let rest = if call.args.len() >= 2 {
                                call.args.iter().take(2).collect()
                            } else if call.args.len() == 1 {
                                if let Some(Expr::Tuple(t)) = call.args.first() {
                                    t.elts.iter().collect()
                                } else {
                                    call.args.iter().collect()
                                }
                            } else {
                                Vec::new()
                            };
                            (&attr.value, rest)
                        };
                    let trans_type = match attr_name {
                        "transpose" => Transpose::Transpose,
                        _ => Transpose::T,
                    };
                    let base_hint = infer_expr_shape(
                        base,
                        vars,
                        func_map,
                        imports,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        false,
                        source,
                        module_cache.as_deref_mut(),
                        module_path,
                    );
                    return infer_permute(
                        base,
                        &order_args,
                        trans_type,
                        base_hint,
                        vars,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        call.range,
                    );
                } else if attr_name == "unsqueeze" {
                    let base = if in_torch {
                        // torch.unsqueeze(torch.Tensor, ...)
                        call.args.first()?
                    } else {
                        attr.value.as_ref()
                    };
                    let dim_arg = get_arg(&call, "dim", offset);
                    return infer_unsqueeze(
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
                        module_cache.as_deref_mut(),
                        module_path,
                    );
                } else if attr_name == "squeeze" {
                    // tensor.squeeze(...) vs torch.squeeze(tensor, ...)
                    let base = if in_torch {
                        call.args.first()?
                    } else {
                        attr.value.as_ref()
                    };

                    return infer_squeeze(
                        base,
                        get_arg(&call, "dim", offset),
                        vars,
                        func_map,
                        imports,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        call.range,
                        module_cache.as_deref_mut(),
                        module_path,
                        true,
                    );
                } else if AGGR_ALIASES.contains(&attr_name) {
                    // tensor.sum(...) vs torch.sum(tensor, ...)
                    let base = if in_torch {
                        call.args.first()?
                    } else {
                        attr.value.as_ref()
                    };
                    return infer_squeeze(
                        base,
                        get_arg(&call, "dim", offset),
                        vars,
                        func_map,
                        imports,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        call.range,
                        module_cache.as_deref_mut(),
                        module_path,
                        false,
                    );
                } else if NOOP_DIM_ALIASES.contains(&attr_name) {
                    let base = if in_torch {
                        call.args.first()?
                    } else {
                        attr.value.as_ref()
                    };
                    let base_hint = infer_expr_shape(
                        base,
                        vars,
                        func_map,
                        imports,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        false,
                        source,
                        module_cache.as_deref_mut(),
                        module_path,
                    );
                    return infer_noop(
                        base_hint,
                        get_arg(&call, "dim", offset),
                        diagnostics,
                        source,
                        call.range,
                        attr_name != "argsort", // dim optional for argsort
                    );
                } else if NOOP_ALIASES.contains(&attr_name) {
                    return infer_expr_shape(
                        if in_torch {
                            call.args.first()?
                        } else {
                            attr.value.as_ref()
                        },
                        vars,
                        func_map,
                        imports,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        false,
                        source,
                        module_cache.as_deref_mut(),
                        module_path,
                    );
                }
            }
            None
        }
        // attributes of a tensor, not a method!
        Expr::Attribute(attr) => {
            let attr_name: &str = attr.attr.as_ref();
            if attr_name == "T" {
                let base_hint =
                    lookup_shape(&attr.value, vars, hover_entries, record_hovers, source).or_else(
                        || {
                            infer_expr_shape(
                                &attr.value,
                                vars,
                                func_map,
                                imports,
                                call_stack,
                                diagnostics,
                                hover_entries,
                                false,
                                source,
                                module_cache.as_deref_mut(),
                                module_path,
                            )
                        },
                    );
                let order_args: Vec<&Expr> = Vec::new();
                let res = infer_permute(
                    &attr.value,
                    &order_args,
                    Transpose::T,
                    base_hint,
                    vars,
                    diagnostics,
                    hover_entries,
                    record_hovers,
                    source,
                    attr.range,
                );
                if record_hovers {
                    if let Some(s) = res.clone() {
                        let range = text_range_to_lsp(attr.range, source);
                        hover_entries.push((range, HoverInfo { shape: Some(s) }));
                    }
                }
                return res;
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
        Expr::Name(n) => n.id.as_str() == "torch" || imports.torch_aliases.contains(&n.id),
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
fn collect_imports(
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
        "transpose",
        "t",
        "softmax",
        "noop",
    ] {
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
                    imports
                        .module_aliases
                        .insert(as_id.clone(), name.to_string());
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
                let resolved_module = resolve_from_module(f, module_path, project_root);
                if let Some(module) = &resolved_module
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
                        } else if NOOP_DIM_ALIASES.contains(&name) {
                            let id = alias
                                .asname
                                .clone()
                                .unwrap_or_else(|| Identifier::from(name));
                            imports
                                .func_aliases
                                .entry("softmax")
                                .or_default()
                                .insert(id);
                        } else if NOOP_ALIASES.contains(&name) {
                            let id = alias
                                .asname
                                .clone()
                                .unwrap_or_else(|| Identifier::from(name));
                            imports.func_aliases.entry("noop").or_default().insert(id);
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
        if let Some(root) = project_root {
            if dir == root {
                break;
            }
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

fn assignment_shape_checks(
    target: &Expr,
    value: &Expr,
    vars: &mut HashMap<Identifier, VarState>,
    diagnostics: &mut Vec<Diagnostic>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    record_hovers: bool,
    source: &str,
) -> bool {
    let Expr::Tuple(tup) = target else {
        return false;
    };
    let dims: Vec<Identifier> = tup.elts.iter().filter_map(name_from_expr).collect();
    if dims.len() != tup.elts.len() || dims.is_empty() {
        return false;
    }
    let Expr::Attribute(attr) = value else {
        return false;
    };
    if attr.attr.as_str() != "shape" {
        return false;
    }
    let Some(base_id) = name_from_expr(&attr.value) else {
        return false;
    };

    let range = text_range_to_lsp(expr_text_range(value), source);
    let existing_state = vars.get(&base_id);
    let existing_shape = existing_state.and_then(|v| v.annotated.clone().or(v.inferred.clone()));

    if let Some(shape) = existing_shape {
        if shape.dims.len() == dims.len() {
            let mut new_shape = shape.clone();
            new_shape.dims = dims.iter().map(|d| d.to_string()).collect();
            vars.insert(
                base_id.clone(),
                VarState {
                    annotated: None,
                    inferred: Some(new_shape.clone()),
                },
            );
            if record_hovers {
                let hrange = text_range_to_lsp(expr_text_range(&attr.value), source);
                hover_entries.push((
                    hrange,
                    HoverInfo {
                        shape: Some(new_shape),
                    },
                ));
            }
        } else {
            diagnostics.push(Diagnostic {
                range,
                severity: Some(DiagnosticSeverity::ERROR),
                code: None,
                code_description: None,
                source: Some("shapels".into()),
                message: "Cannot unroll shape with different rank".into(),
                related_information: None,
                tags: None,
                data: None,
            });
        }
    } else {
        let new_shape = Shape {
            dtype: None,
            dims: dims.iter().map(|d| d.to_string()).collect(),
        };
        vars.insert(
            base_id.clone(),
            VarState {
                annotated: None,
                inferred: Some(new_shape.clone()),
            },
        );
        if record_hovers {
            let hrange = text_range_to_lsp(expr_text_range(&attr.value), source);
            hover_entries.push((
                hrange,
                HoverInfo {
                    shape: Some(new_shape),
                },
            ));
        }
    }
    true
}

fn resolve_from_module(
    f: &ast::StmtImportFrom,
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

fn get_arg<'a, R>(
    call: &'a ExprCall<R>,
    name_arg: &str,
    as_positional: usize,
) -> Option<&'a Expr<R>> {
    // first check for named args
    let dim_keyword = call
        .keywords
        .iter()
        .find(|kw| kw.arg.as_deref() == Some(name_arg))
        .map(|kw| &kw.value);
    dim_keyword.or(call.args.get(as_positional))
}
