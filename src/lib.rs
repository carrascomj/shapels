#![allow(clippy::too_many_arguments)]

use lsp_types::{Diagnostic, DiagnosticSeverity, Position, Range};
use rustpython_parser::Parse;
use rustpython_parser::ast::{
    self, Arguments, Constant, Expr, ExprBinOp, ExprCall, ExprCompare, ExprList, ExprTuple,
    Identifier, Operator, Stmt, Suite,
};
use rustpython_parser::text_size::{TextRange, TextSize};
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
mod infer;
pub mod op_groups;
use crate::infer::{
    ShapeOrExpr, Transpose, infer_broadcastable_poswise, infer_conv, infer_creation_size,
    infer_flatten, infer_index, infer_matmul_shapes, infer_noop, infer_permute, infer_range_size,
    infer_repeat, infer_repeat_interleave, infer_squeeze, infer_to, infer_unsqueeze,
    infer_view_like, shape_dims_equal,
};
pub use crate::op_groups::AGGR_ALIASES;
use crate::op_groups::{BroadcastOp, Imports, TORCH_DTYPES, TorchOp, collect_imports};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub dtype: Option<String>,
    pub dims: Vec<String>,
}

impl Shape {
    pub fn render(&self) -> String {
        format!("[{}]", self.dims.join(" "))
    }
    pub fn dim_string(&self) -> String {
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
    class_ref: Option<ClassRef>,
}

/// Output of [`simulate_function`], such that return type
/// can be recorded and matched against tuple destructuring on
/// assignment.
#[derive(Debug, Clone, Default)]
enum ReturnValue {
    Single(Shape),
    Tuple(Vec<Option<Shape>>),
    #[default]
    None,
}

impl ReturnValue {
    fn from_shape(shape: Option<Shape>) -> Self {
        shape.map(Self::Single).unwrap_or(Self::None)
    }

    fn from_tuple(shapes: Vec<Option<Shape>>) -> Self {
        Self::Tuple(shapes)
    }

    fn first(&self) -> Option<&Shape> {
        match self {
            Self::Single(shape) => Some(shape),
            Self::Tuple(tuple) => tuple.first().and_then(|x| x.as_ref()),
            Self::None => None,
        }
    }

    fn tuple(&self) -> Option<&[Option<Shape>]> {
        match self {
            Self::Tuple(tuple) => Some(tuple.as_ref()),
            _ => None,
        }
    }

    fn is_some(&self) -> bool {
        match self {
            Self::Single(_) => true,
            Self::Tuple(tup) => !tup.is_empty(),
            Self::None => false,
        }
    }
}

#[derive(Clone)]
struct FunctionInfo {
    args: Box<Arguments>,
    body: Vec<Stmt>,
    returns: Option<Box<Expr>>,
}

type FuncMap = HashMap<Identifier, FunctionInfo>;

#[derive(Debug, Clone)]
struct ClassRef {
    name: Identifier,
    module: Option<String>,
}

#[derive(Clone)]
struct ClassInfo {
    is_torch_module: bool,
    forward: Option<FunctionInfo>,
    methods: HashMap<Identifier, FunctionInfo>,
}

type ClassMap = HashMap<Identifier, ClassInfo>;

#[derive(Clone)]
pub(crate) struct CachedModule {
    path: PathBuf,
    source: String,
    func_map: FuncMap,
    imports: Imports,
    class_map: ClassMap,
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
        let normalized = normalize_return_annotations(&source);
        let source = normalized.into_owned();
        let module = Suite::parse(&source, module_name).ok()?;
        let imports = collect_imports(&module, Some(&path), self.project_root());
        let mut func_map: FuncMap = HashMap::new();
        collect_function_defs(&module, &mut func_map);
        let class_map = collect_class_defs(&module, &imports);
        let cached = CachedModule {
            path,
            source,
            func_map,
            imports,
            class_map,
        };
        self.modules.insert(module_name.to_string(), cached.clone());
        Some(cached)
    }
}

/// Normalize return annotations so `-> A, B` parses as `-> (A, B)`.
fn normalize_return_annotations<'a>(source: &'a str) -> Cow<'a, str> {
    let mut changed = false;
    let mut out = String::with_capacity(source.len());
    let mut first = true;
    for line in source.lines() {
        if !first {
            out.push('\n');
        }
        first = false;
        if let (Some(arrow_idx), Some(colon_idx)) = (line.find("->"), line.rfind(':'))
            && arrow_idx < colon_idx
        {
            let ann = &line[arrow_idx + 2..colon_idx];
            let ann_trim = ann.trim();
            if ann_trim.contains(',') && !ann_trim.starts_with('(') {
                changed = true;
                out.push_str(&line[..arrow_idx + 2]);
                out.push(' ');
                out.push('(');
                out.push_str(ann_trim);
                out.push(')');
                out.push_str(&line[colon_idx..]);
                continue;
            }
        }
        out.push_str(line);
    }
    if changed {
        Cow::Owned(out)
    } else {
        Cow::Borrowed(source)
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

/// Resolve a class reference across local/module context and run a callback with its info.
fn with_class_info<R, F>(
    class_ref: &ClassRef,
    source: &str,
    func_map: &FuncMap,
    imports: &Imports,
    class_map: &ClassMap,
    module_cache: &mut Option<&mut ModuleCache>,
    module_path: Option<&Path>,
    f: F,
) -> Option<R>
where
    F: FnOnce(
        &ClassInfo,
        &str,
        &FuncMap,
        &Imports,
        &ClassMap,
        Option<&Path>,
        &mut Option<&mut ModuleCache>,
    ) -> R,
{
    match &class_ref.module {
        Some(module_name) => {
            let module = {
                let (Some(cache), Some(cur_path)) = (module_cache.as_deref_mut(), module_path)
                else {
                    return None;
                };
                cache.get_module(module_name, cur_path)?
            };
            let class_info = module.class_map.get(&class_ref.name)?;
            Some(f(
                class_info,
                &module.source,
                &module.func_map,
                &module.imports,
                &module.class_map,
                Some(module.path.as_path()),
                module_cache,
            ))
        }
        None => {
            let class_info = class_map.get(&class_ref.name)?;
            Some(f(
                class_info,
                source,
                func_map,
                imports,
                class_map,
                module_path,
                module_cache,
            ))
        }
    }
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
                    if lib_dir.exists()
                        && let Ok(entries) = fs::read_dir(&lib_dir)
                    {
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
                    .map(std::ffi::OsStr::new)
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
    let normalized = normalize_return_annotations(source);
    let source = normalized.as_ref();
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
            let mut func_map: FuncMap = HashMap::new();
            collect_function_defs(&module, &mut func_map);
            let class_map = collect_class_defs(&module, &imports);

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
                &class_map,
                &mut Vec::new(),
                HashMap::new(),
                true,
                module_cache.as_deref_mut(),
                current_path,
            );
            analysis.diagnostics.append(&mut top_diags);
            analysis.hover_entries.append(&mut top_hovers);

            analyze_function_bodies(
                &module,
                source,
                &func_map,
                &imports,
                &class_map,
                &mut analysis,
                module_cache,
                current_path,
            );
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

fn collect_function_defs(body: &[Stmt], func_map: &mut FuncMap) {
    for stmt in body {
        if let Stmt::FunctionDef(func) = stmt {
            func_map.insert(
                func.name.clone(),
                FunctionInfo {
                    args: func.args.clone(),
                    body: func.body.clone(),
                    returns: func.returns.clone(),
                },
            );
        }
    }
}

struct ClassDefInfo {
    forward: Option<FunctionInfo>,
    methods: HashMap<Identifier, FunctionInfo>,
    base_names: Vec<Identifier>,
    direct_torch: bool,
}

fn is_torch_nn_module_base(expr: &Expr, imports: &Imports) -> bool {
    let Expr::Attribute(attr) = expr else {
        return false;
    };
    if attr.attr.as_str() != "Module" {
        return false;
    }
    match attr.value.as_ref() {
        Expr::Attribute(nn_attr) if nn_attr.attr.as_str() == "nn" => {
            if let Expr::Name(torch_name) = nn_attr.value.as_ref() {
                return imports.torch_aliases.contains(&torch_name.id);
            }
            false
        }
        Expr::Name(nn_name) => imports
            .module_aliases
            .get(&nn_name.id)
            .map(|module| module == "torch.nn")
            .unwrap_or(false),
        _ => false,
    }
}

fn collect_class_defs(body: &[Stmt], imports: &Imports) -> ClassMap {
    let mut defs: HashMap<Identifier, ClassDefInfo> = HashMap::new();
    for stmt in body {
        if let Stmt::ClassDef(class_def) = stmt {
            let mut methods = HashMap::new();
            let mut forward = None;
            for stmt in class_def.body.iter() {
                if let Stmt::FunctionDef(func) = stmt {
                    let info = FunctionInfo {
                        args: func.args.clone(),
                        body: func.body.clone(),
                        returns: func.returns.clone(),
                    };
                    if func.name.as_str() == "forward" {
                        forward = Some(info.clone());
                    }
                    methods.insert(func.name.clone(), info);
                }
            }
            let base_names = class_def
                .bases
                .iter()
                .filter_map(|base| match base {
                    Expr::Name(name) => Some(name.id.clone()),
                    _ => None,
                })
                .collect();
            let direct_torch = class_def
                .bases
                .iter()
                .any(|base| is_torch_nn_module_base(base, imports));
            defs.insert(
                class_def.name.clone(),
                ClassDefInfo {
                    forward,
                    methods,
                    base_names,
                    direct_torch,
                },
            );
        }
    }

    let mut is_torch: HashMap<Identifier, bool> = defs
        .iter()
        .map(|(name, info)| (name.clone(), info.direct_torch))
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for (name, info) in defs.iter() {
            if !is_torch.get(name).copied().unwrap_or(false)
                && info
                    .base_names
                    .iter()
                    .any(|base| is_torch.get(base).copied().unwrap_or(false))
            {
                is_torch.insert(name.clone(), true);
                changed = true;
            }
        }
    }

    let mut class_map = HashMap::new();
    for (name, info) in defs {
        class_map.insert(
            name.clone(),
            ClassInfo {
                is_torch_module: is_torch.get(&name).copied().unwrap_or(false),
                forward: info.forward,
                methods: info.methods,
            },
        );
    }
    class_map
}

/// Iterate over the function and classes of a module `body`.
///
/// The base case is a function, where shape inference is run. For classes,
/// it's called recursively, treating the class as a module where each method is a function.
fn analyze_function_bodies(
    body: &[Stmt],
    source: &str,
    func_map: &FuncMap,
    imports: &Imports,
    class_map: &ClassMap,
    analysis: &mut Analysis,
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) {
    for stmt in body {
        match stmt {
            Stmt::FunctionDef(func) => {
                let mut func_analysis = analyze_function(
                    &func.args,
                    &func.body,
                    source,
                    func_map,
                    imports,
                    class_map,
                    &mut Vec::new(),
                    module_cache.as_deref_mut(),
                    module_path,
                );
                analysis.diagnostics.append(&mut func_analysis.diagnostics);
                analysis
                    .hover_entries
                    .append(&mut func_analysis.hover_entries);
            }
            Stmt::ClassDef(class_def) => analyze_function_bodies(
                &class_def.body,
                source,
                func_map,
                imports,
                class_map,
                analysis,
                module_cache.as_deref_mut(),
                module_path,
            ),
            _ => {}
        }
    }
}

fn analyze_function(
    args: &Arguments,
    body: &[Stmt],
    source: &str,
    func_map: &FuncMap,
    imports: &Imports,
    class_map: &ClassMap,
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
        class_map,
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

/// Initializes inputs of a function and wraps around [`simulate_block`]
/// that may run recursively carrying the initialized inputs.
fn simulate_function(
    args: &Arguments,
    body: &[Stmt],
    source: &str,
    func_map: &FuncMap,
    imports: &Imports,
    class_map: &ClassMap,
    call_stack: &mut Vec<Identifier>,
    mut initial_vars: HashMap<Identifier, VarState>,
    record_hovers: bool,
    module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> (Vec<Diagnostic>, Vec<(Range, HoverInfo)>, ReturnValue) {
    let mut diagnostics = Vec::new();
    let mut hover_entries = Vec::new();
    let mut vars: HashMap<Identifier, VarState> = HashMap::new();

    seed_args_from_annotations(
        args,
        source,
        &mut vars,
        &mut hover_entries,
        Some(&mut initial_vars),
        record_hovers,
        imports,
        class_map,
    );

    let mut return_value = ReturnValue::default();

    let _ = simulate_block(
        body,
        &mut vars,
        &mut diagnostics,
        &mut hover_entries,
        &mut return_value,
        source,
        func_map,
        imports,
        class_map,
        call_stack,
        record_hovers,
        module_cache,
        module_path,
        false,
    );

    (diagnostics, hover_entries, return_value)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockFlow {
    None,
    Break,
    Continue,
    Return,
}

/// Run static shape inference on assignments and return types.
fn simulate_block(
    body: &[Stmt],
    vars: &mut HashMap<Identifier, VarState>,
    diagnostics: &mut Vec<Diagnostic>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    return_value: &mut ReturnValue,
    source: &str,
    func_map: &FuncMap,
    imports: &Imports,
    class_map: &ClassMap,
    call_stack: &mut Vec<Identifier>,
    record_hovers: bool,
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
    in_loop: bool,
) -> BlockFlow {
    for stmt in body {
        match stmt {
            Stmt::AnnAssign(assign) => {
                if assignment_shape_checks(
                    &assign.target,
                    assign.value.as_deref().unwrap_or(assign.target.as_ref()),
                    vars,
                    diagnostics,
                    hover_entries,
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
                            vars,
                            diagnostics,
                            hover_entries,
                            record_hovers,
                            source,
                        ) {
                            inferred = vars
                                .get(&name)
                                .and_then(|v| v.annotated.clone().or(v.inferred.clone()));
                        } else {
                            inferred = infer_expr_shape(
                                val,
                                vars,
                                func_map,
                                imports,
                                class_map,
                                call_stack,
                                diagnostics,
                                hover_entries,
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
                                    class_ref: None,
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
                                class_ref: None,
                            },
                        );
                        if record_hovers {
                            hover_entries.push((range, HoverInfo { shape: Some(shape) }));
                        }
                    } else if let Some(val) = &assign.value
                        && let Some(class_ref) = class_ref_from_constructor_call(
                            val,
                            class_map,
                            imports,
                            module_cache.as_deref_mut(),
                            module_path,
                        )
                    {
                        vars.insert(
                            name.clone(),
                            VarState {
                                annotated: ann_shape.clone(),
                                inferred: None,
                                class_ref: Some(class_ref),
                            },
                        );
                    }
                }
            }
            Stmt::Assign(assign) => {
                // handle tuple destructuring of `.shape`
                if assign.targets.len() == 1
                    && assignment_shape_checks(
                        &assign.targets[0],
                        &assign.value,
                        vars,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                    )
                {
                    continue;
                }
                if assign.targets.len() == 1
                    && matches!(assign.targets[0], Expr::Tuple(_))
                    && let Expr::Tuple(target_tuple) = &assign.targets[0]
                    && let Some(tuple_shapes) = infer_tuple_elements(
                        &assign.value,
                        vars,
                        func_map,
                        imports,
                        class_map,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        module_cache.as_deref_mut(),
                        module_path,
                    )
                {
                    for (target_expr, shape_opt) in
                        target_tuple.elts.iter().zip(tuple_shapes.into_iter())
                    {
                        if let (Some(name), Some(shape)) = (name_from_expr(target_expr), shape_opt)
                        {
                            vars.insert(
                                name.clone(),
                                VarState {
                                    annotated: None,
                                    inferred: Some(shape.clone()),
                                    class_ref: None,
                                },
                            );
                            if record_hovers {
                                let range = text_range_to_lsp(expr_text_range(target_expr), source);
                                hover_entries.push((range, HoverInfo { shape: Some(shape) }));
                            }
                        }
                    }
                    continue;
                }
                if assign.targets.len() == 1
                    && let Some(name) = name_from_expr(&assign.targets[0])
                {
                    let range = text_range_to_lsp(expr_text_range(&assign.targets[0]), source);
                    let shape = infer_expr_shape(
                        &assign.value,
                        vars,
                        func_map,
                        imports,
                        class_map,
                        call_stack,
                        diagnostics,
                        hover_entries,
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
                                class_ref: None,
                            },
                        );
                        if record_hovers {
                            hover_entries.push((range, HoverInfo { shape: Some(shape) }));
                        }
                    } else if let Some(class_ref) = class_ref_from_constructor_call(
                        &assign.value,
                        class_map,
                        imports,
                        module_cache.as_deref_mut(),
                        module_path,
                    ) {
                        vars.insert(
                            name.clone(),
                            VarState {
                                annotated: None,
                                inferred: None,
                                class_ref: Some(class_ref),
                            },
                        );
                    }
                }
            }
            Stmt::For(for_stmt) => {
                let flow = simulate_block(
                    &for_stmt.body,
                    vars,
                    diagnostics,
                    hover_entries,
                    return_value,
                    source,
                    func_map,
                    imports,
                    class_map,
                    call_stack,
                    record_hovers,
                    module_cache.as_deref_mut(),
                    module_path,
                    true,
                );
                if flow == BlockFlow::Return {
                    return BlockFlow::Return;
                }
                if flow != BlockFlow::Break {
                    let else_flow = simulate_block(
                        &for_stmt.orelse,
                        vars,
                        diagnostics,
                        hover_entries,
                        return_value,
                        source,
                        func_map,
                        imports,
                        class_map,
                        call_stack,
                        record_hovers,
                        module_cache.as_deref_mut(),
                        module_path,
                        true,
                    );
                    if else_flow == BlockFlow::Return {
                        return BlockFlow::Return;
                    }
                }
            }
            Stmt::While(while_stmt) => {
                let flow = simulate_block(
                    &while_stmt.body,
                    vars,
                    diagnostics,
                    hover_entries,
                    return_value,
                    source,
                    func_map,
                    imports,
                    class_map,
                    call_stack,
                    record_hovers,
                    module_cache.as_deref_mut(),
                    module_path,
                    true,
                );
                if flow == BlockFlow::Return {
                    return BlockFlow::Return;
                }
                if flow != BlockFlow::Break {
                    let else_flow = simulate_block(
                        &while_stmt.orelse,
                        vars,
                        diagnostics,
                        hover_entries,
                        return_value,
                        source,
                        func_map,
                        imports,
                        class_map,
                        call_stack,
                        record_hovers,
                        module_cache.as_deref_mut(),
                        module_path,
                        true,
                    );
                    if else_flow == BlockFlow::Return {
                        return BlockFlow::Return;
                    }
                }
            }
            Stmt::If(if_stmt) => {
                let mut body_vars = vars.clone();
                let body_flow = simulate_block(
                    &if_stmt.body,
                    &mut body_vars,
                    diagnostics,
                    hover_entries,
                    return_value,
                    source,
                    func_map,
                    imports,
                    class_map,
                    call_stack,
                    record_hovers,
                    module_cache.as_deref_mut(),
                    module_path,
                    in_loop,
                );
                let mut else_vars = vars.clone();
                let else_flow = simulate_block(
                    &if_stmt.orelse,
                    &mut else_vars,
                    diagnostics,
                    hover_entries,
                    return_value,
                    source,
                    func_map,
                    imports,
                    class_map,
                    call_stack,
                    record_hovers,
                    module_cache.as_deref_mut(),
                    module_path,
                    in_loop,
                );
                if body_flow == else_flow
                    && matches!(
                        body_flow,
                        BlockFlow::Break | BlockFlow::Continue | BlockFlow::Return
                    )
                {
                    return body_flow;
                }
            }
            Stmt::Break(_) => {
                if in_loop {
                    return BlockFlow::Break;
                }
            }
            Stmt::Continue(_) => {
                if in_loop {
                    return BlockFlow::Continue;
                }
            }
            Stmt::Return(ret) => {
                if let Some(val) = &ret.value {
                    let new_value = match val.as_ref() {
                        Expr::Tuple(tuple) => {
                            let tuple_shapes: Vec<Option<Shape>> = tuple
                                .elts
                                .iter()
                                .map(|elt| {
                                    infer_expr_shape(
                                        elt,
                                        vars,
                                        func_map,
                                        imports,
                                        class_map,
                                        call_stack,
                                        diagnostics,
                                        hover_entries,
                                        record_hovers,
                                        source,
                                        module_cache.as_deref_mut(),
                                        module_path,
                                    )
                                })
                                .collect();
                            ReturnValue::from_tuple(tuple_shapes)
                        }
                        _ => ReturnValue::from_shape(infer_expr_shape(
                            val,
                            vars,
                            func_map,
                            imports,
                            class_map,
                            call_stack,
                            diagnostics,
                            hover_entries,
                            record_hovers,
                            source,
                            module_cache.as_deref_mut(),
                            module_path,
                        )),
                    };
                    if record_hovers && let Some(shape) = new_value.first() {
                        let range = text_range_to_lsp(expr_text_range(val), source);
                        hover_entries.push((
                            range,
                            HoverInfo {
                                shape: Some(shape.clone()),
                            },
                        ));
                    }
                    if new_value.is_some() {
                        *return_value = new_value;
                    }
                }
                return BlockFlow::Return;
            }
            _ => {}
        }
    }
    BlockFlow::None
}

/// Recursively run static shape inference on an [`Expr`].
///
/// This is the central function for inference, it calls the specialized
/// inference at src/infer.rs depending on type of the expression.
fn infer_expr_shape(
    expr: &Expr,
    vars: &HashMap<Identifier, VarState>,
    func_map: &FuncMap,
    imports: &Imports,
    class_map: &ClassMap,
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
                infer_broadcastable_poswise(
                    &ShapeOrExpr::Expr(left),
                    right,
                    vars,
                    func_map,
                    imports,
                    class_map,
                    call_stack,
                    diagnostics,
                    hover_entries,
                    record_hovers,
                    source,
                    *expr_range,
                    module_cache.as_deref_mut(),
                    module_path,
                    BroadcastOp::Arithmetic,
                )
            }
            Operator::BitAnd
            | Operator::BitXor
            | Operator::BitOr
            | Operator::LShift
            | Operator::RShift => infer_broadcastable_poswise(
                &ShapeOrExpr::Expr(left),
                right,
                vars,
                func_map,
                imports,
                class_map,
                call_stack,
                diagnostics,
                hover_entries,
                record_hovers,
                source,
                *expr_range,
                module_cache.as_deref_mut(),
                module_path,
                BroadcastOp::Bitwise,
            ),
            Operator::MatMult => infer_matmul_shapes(
                left,
                right,
                vars,
                func_map,
                imports,
                class_map,
                call_stack,
                diagnostics,
                hover_entries,
                record_hovers,
                source,
                *expr_range,
                module_cache.as_deref_mut(),
                module_path,
            ),
            _ => None,
        },
        Expr::Compare(ExprCompare {
            left: init_left,
            // can be chained, so we need to compute the pairs left to right
            comparators,
            range: expr_range,
            ..
        }) => {
            // first, infer with two Expr
            let mut iter = comparators.iter();
            let first_right = iter.next()?;
            let init = infer_broadcastable_poswise(
                &ShapeOrExpr::Expr(init_left.as_ref()),
                first_right,
                vars,
                func_map,
                imports,
                class_map,
                call_stack,
                diagnostics,
                hover_entries,
                record_hovers,
                source,
                *expr_range,
                module_cache.as_deref_mut(),
                module_path,
                BroadcastOp::Eq,
            );

            // second, fold with Shape and Expr
            iter.fold(init, |left, right| {
                infer_broadcastable_poswise(
                    &ShapeOrExpr::Shape(left.as_ref()),
                    right,
                    vars,
                    func_map,
                    imports,
                    class_map,
                    call_stack,
                    diagnostics,
                    hover_entries,
                    record_hovers,
                    source,
                    *expr_range,
                    module_cache.as_deref_mut(),
                    module_path,
                    BroadcastOp::Eq,
                )
            })
        }
        Expr::Call(call) => {
            if let Expr::Name(func_name) = call.func.as_ref() {
                if let Some(class_ref) = vars.get(&func_name.id).and_then(|v| v.class_ref.as_ref())
                    && let Some(ret) = infer_class_call_return(
                        call,
                        class_ref,
                        vars,
                        func_map,
                        imports,
                        class_map,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        module_cache.as_deref_mut(),
                        module_path,
                    )
                {
                    return ret.first().cloned();
                }
                let torchop_shape = torch_op_to_shape(
                    vars,
                    func_map,
                    imports,
                    class_map,
                    call_stack,
                    diagnostics,
                    hover_entries,
                    record_hovers,
                    source,
                    &mut module_cache,
                    module_path,
                    call,
                    func_name.id.as_str(),
                    TorchOp::as_call(&func_name.id, imports),
                    TorchOpKind::Function,
                    None,
                );
                if torchop_shape.is_some() {
                    return torchop_shape;
                }

                if let Some((module_name, original)) = imports.from_imports.get(&func_name.id)
                    && let (Some(cache), Some(cur_path)) =
                        (module_cache.as_deref_mut(), module_path)
                    && let Some(module) = cache.get_module(module_name, cur_path)
                    && let Some(callee_info) = module.func_map.get(original)
                    && let Some(ret) = infer_call_return_from_info(
                        call,
                        &func_name.id,
                        callee_info,
                        &module.source,
                        &module.func_map,
                        &module.imports,
                        &module.class_map,
                        vars,
                        func_map,
                        imports,
                        class_map,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        module_cache.as_deref_mut(),
                        Some(module.path.as_path()),
                        0,
                        false,
                    )
                {
                    return ret.first().cloned();
                }
                if let Some(callee_info) = func_map.get(&func_name.id)
                    && let Some(ret) = infer_call_return_from_info(
                        call,
                        &func_name.id,
                        callee_info,
                        source,
                        func_map,
                        imports,
                        class_map,
                        vars,
                        func_map,
                        imports,
                        class_map,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        module_cache.as_deref_mut(),
                        module_path,
                        0,
                        false,
                    )
                {
                    return ret.first().cloned();
                }
            }
            // methods are functions with attributes
            if let Expr::Attribute(attr) = call.func.as_ref() {
                let attr_name: &str = attr.attr.as_ref();
                if let Expr::Name(base_name) = attr.value.as_ref()
                    && let Some(class_ref) =
                        vars.get(&base_name.id).and_then(|v| v.class_ref.as_ref())
                    && let Some(ret) = infer_class_method_call_return(
                        call,
                        class_ref,
                        &attr.attr,
                        vars,
                        func_map,
                        imports,
                        class_map,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        module_cache.as_deref_mut(),
                        module_path,
                    )
                {
                    return ret.first().cloned();
                }
                if let Expr::Name(module_ident) = attr.value.as_ref()
                    && let Some(module_name) = imports.module_aliases.get(&module_ident.id)
                    && module_name != "torch"
                    && let (Some(cache), Some(cur_path)) =
                        (module_cache.as_deref_mut(), module_path)
                    && let Some(module) = cache.get_module(module_name, cur_path)
                    && let Some(callee_info) = module.func_map.get(&attr.attr)
                    && let Some(ret) = infer_call_return_from_info(
                        call,
                        &attr.attr,
                        callee_info,
                        &module.source,
                        &module.func_map,
                        &module.imports,
                        &module.class_map,
                        vars,
                        func_map,
                        imports,
                        class_map,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        module_cache.as_deref_mut(),
                        Some(module.path.as_path()),
                        0,
                        false,
                    )
                {
                    return ret.first().cloned();
                }
                // two cases: torch.ATTR_NAME(torch.Tensor, ...) or torch.Tensor.ATTR_NAME(...)
                let op_kind = function_or_method(&attr.value, imports);
                let aliased_shape = torch_op_to_shape(
                    vars,
                    func_map,
                    imports,
                    class_map,
                    call_stack,
                    diagnostics,
                    hover_entries,
                    record_hovers,
                    source,
                    &mut module_cache,
                    module_path,
                    call,
                    attr_name,
                    TorchOp::from_attr(attr_name),
                    op_kind,
                    Some(attr),
                );
                if aliased_shape.is_some() {
                    return aliased_shape;
                }
            }
            None
        }
        Expr::Subscript(sub) => infer_index(
            &sub.value,
            &sub.slice,
            vars,
            func_map,
            imports,
            class_map,
            call_stack,
            diagnostics,
            hover_entries,
            record_hovers,
            source,
            module_cache.as_deref_mut(),
            module_path,
        ),
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
                                class_map,
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
                if record_hovers && let Some(s) = res.clone() {
                    let range = text_range_to_lsp(attr.range, source);
                    hover_entries.push((range, HoverInfo { shape: Some(s) }));
                }
                return res;
            } else if attr_name == "shape" || attr_name == "dtype" {
                return lookup_shape(&attr.value, vars, hover_entries, record_hovers, source)
                    .or_else(|| {
                        infer_expr_shape(
                            &attr.value,
                            vars,
                            func_map,
                            imports,
                            class_map,
                            call_stack,
                            diagnostics,
                            hover_entries,
                            false,
                            source,
                            module_cache,
                            module_path,
                        )
                    });
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

/// Match an `TorchOp` against all supported torch operations.
///
/// It can be a function or a method, returns None if the operation
/// is not implemented OR if the arguments to the operation are not
/// incorrect.
fn torch_op_to_shape(
    vars: &HashMap<Identifier, VarState>,
    func_map: &HashMap<Identifier, FunctionInfo>,
    imports: &Imports,
    class_map: &HashMap<Identifier, ClassInfo>,
    call_stack: &mut Vec<Identifier>,
    diagnostics: &mut Vec<Diagnostic>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    record_hovers: bool,
    source: &str,
    module_cache: &mut Option<&mut ModuleCache>,
    module_path: Option<&Path>,
    call: &ExprCall,
    attr_name: &str,
    torch_op: TorchOp,
    op_kind: TorchOpKind,
    maybe_attr: Option<&ast::ExprAttribute>,
) -> Option<Shape> {
    use TorchOpKind::*;

    let (may_arg0, may_arg1, offset) = match op_kind {
        Function => (call.args.first(), call.args.get(1), 1),
        Method => (maybe_attr.map(|x| x.value.as_ref()), call.args.first(), 0),
    };
    match (torch_op, may_arg0, may_arg1, &op_kind) {
        (TorchOp::MatMul, Some(arg0), Some(arg1), _) => infer_matmul_shapes(
            arg0,
            arg1,
            vars,
            func_map,
            imports,
            class_map,
            call_stack,
            diagnostics,
            hover_entries,
            record_hovers,
            source,
            call.range,
            module_cache.as_deref_mut(),
            module_path,
        ),
        (op @ (TorchOp::Squeeze | TorchOp::Aggr), Some(arg0), _, _) => infer_squeeze(
            arg0,
            get_arg(call, "dim", offset),
            vars,
            func_map,
            imports,
            class_map,
            call_stack,
            diagnostics,
            hover_entries,
            record_hovers,
            source,
            call.range,
            module_cache.as_deref_mut(),
            module_path,
            matches!(op, TorchOp::Squeeze),
            get_arg(call, "keepdim", offset + 1),
        ),
        (TorchOp::NoopDim, Some(base), _, _) => {
            let base_hint = infer_expr_shape(
                base,
                vars,
                func_map,
                imports,
                class_map,
                call_stack,
                diagnostics,
                hover_entries,
                false,
                source,
                module_cache.as_deref_mut(),
                module_path,
            );
            infer_noop(
                base_hint,
                get_arg(call, "dim", offset),
                vars,
                diagnostics,
                source,
                call.range,
                attr_name.contains("soft"), // HACK: dim not optional for softmax
            )
        }
        (TorchOp::Noop, Some(base), _, _) => infer_expr_shape(
            base,
            vars,
            func_map,
            imports,
            class_map,
            call_stack,
            diagnostics,
            hover_entries,
            false,
            source,
            module_cache.as_deref_mut(),
            module_path,
        ),
        (TorchOp::Creation { is_size }, _, _, Function) => {
            let shape_assign = tensor_or_shape_as_arg(
                is_size,
                vars,
                func_map,
                imports,
                class_map,
                call_stack,
                diagnostics,
                hover_entries,
                record_hovers,
                source,
                module_cache,
                module_path,
                call,
            );
            let dtype = get_arg(call, "dtype", 200).and_then(|expr| {
                let attr_dtype = if matches!(expr, Expr::Attribute(_)) {
                    infer_expr_shape(
                        expr,
                        vars,
                        func_map,
                        imports,
                        class_map,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        module_cache.as_deref_mut(),
                        module_path,
                    )
                    .and_then(|shape| shape.dtype)
                } else {
                    None
                };
                attr_dtype.or_else(|| get_dtype(expr, imports).map(|x| x.to_string()))
            });

            infer_creation_size(
                call,
                vars,
                diagnostics,
                source,
                shape_assign,
                dtype,
                is_size,
            )
        }
        (TorchOp::RangeOp(range_op), _, _, Function) => infer_range_size(
            call,
            range_op,
            vars,
            func_map,
            imports,
            class_map,
            call_stack,
            diagnostics,
            hover_entries,
            record_hovers,
            source,
            module_cache.as_deref_mut(),
            module_path,
        ),
        (TorchOp::NoArg { predef_dtype }, Some(base), _, _)
            // only `.to` (no predef type) is allowed as Function
            if predef_dtype.is_none() || matches!(op_kind, Method) =>
        {
            let base_expr = infer_expr_shape(
                base,
                vars,
                func_map,
                imports,
                class_map,
                call_stack,
                diagnostics,
                hover_entries,
                false,
                source,
                module_cache.as_deref_mut(),
                module_path,
            );
            let dtype_expr = if let Some(dtype) = predef_dtype {
                Some(&Expr::Constant(ast::ExprConstant {
                    range: expr_text_range(base),
                    value: Constant::Str(dtype.to_string()),
                    kind: None,
                }))
            } else {
                get_arg(call, "dtype", 0)
            };
            infer_to(base_expr, dtype_expr, diagnostics, source, imports)
        }
        (TorchOp::Broadcastable(broadcast_op), Some(left), Some(right), _) => {
            infer_broadcastable_poswise(
                &ShapeOrExpr::Expr(left),
                right,
                vars,
                func_map,
                imports,
                class_map,
                call_stack,
                diagnostics,
                hover_entries,
                record_hovers,
                source,
                call.range,
                module_cache.as_deref_mut(),
                module_path,
                broadcast_op,
            )
        }
        (op @ (TorchOp::View | TorchOp::Expand), Some(base), _, kind)
            if matches!(kind, Method) || matches!(op, TorchOp::View) =>
        {
            let base_hint =
                lookup_shape(base, vars, hover_entries, record_hovers, source).or_else(|| {
                    infer_expr_shape(
                        base,
                        vars,
                        func_map,
                        imports,
                        class_map,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        false,
                        source,
                        module_cache.as_deref_mut(),
                        module_path,
                    )
                });
            infer_view_like(
                base,
                &call.args.iter().collect::<Vec<_>>(),
                &op,
                base_hint,
                vars,
                diagnostics,
                hover_entries,
                record_hovers,
                source,
                call.range,
            )
        }
        (TorchOp::Transpose(transpose), Some(base), _, _) => {
            let order_args = match &transpose {
                Transpose::Permute => {
                    match get_arg(call, "dims", offset) {
                        Some(Expr::Tuple(ExprTuple { elts, .. }))
                        | Some(Expr::List(ExprList { elts, .. })) => elts.iter().collect(),
                        // torch.permute does not accept variadic args for dims
                        // but the torch.Tensor.permute does
                        _ if matches!(op_kind, Method) => call.args.iter().collect(),
                        _ => Vec::new(),
                    }
                }
                // torch.transpose does not accept a size-like
                Transpose::Explicit => call.args.iter().skip(offset).take(2).collect(),
                // no args
                Transpose::T => Vec::new(),
            };
            let base_hint = infer_expr_shape(
                base,
                vars,
                func_map,
                imports,
                class_map,
                call_stack,
                diagnostics,
                hover_entries,
                false,
                source,
                module_cache.as_deref_mut(),
                module_path,
            );
            infer_permute(
                base,
                &order_args,
                transpose,
                base_hint,
                vars,
                diagnostics,
                hover_entries,
                record_hovers,
                source,
                call.range,
            )
        }
        (TorchOp::Unsqueeze, Some(base), _, _) => {
            let dim_arg = get_arg(call, "dim", offset);
            infer_unsqueeze(
                base,
                dim_arg,
                vars,
                func_map,
                imports,
                class_map,
                call_stack,
                diagnostics,
                hover_entries,
                record_hovers,
                source,
                module_cache.as_deref_mut(),
                module_path,
            )
        }
        (TorchOp::Conv(d), Some(base), _, Function) => {
            let kernel = lookup_shape(get_arg(call, "weight", 1)?, vars, hover_entries, record_hovers, source)?;
            lookup_shape(base, vars, hover_entries, record_hovers, source).and_then(|shape| infer_conv(shape, kernel.dims, call, d, diagnostics, source))
        }
        (TorchOp::Repeat, Some(base), _, Method) => {
            lookup_shape(base, vars, hover_entries, record_hovers, source).and_then(|shape| infer_repeat(shape, call, vars, diagnostics, source))
        }
        (TorchOp::RepeatInterleave, Some(base), _, _) => {
            lookup_shape(base, vars, hover_entries, record_hovers, source).and_then(|shape| {
                infer_repeat_interleave(shape, call, offset, diagnostics, source)
            })
        }
        (TorchOp::Flatten, Some(base), _, _) => {
            // TODO(carrascomj): ravel is function-only
            lookup_shape(base, vars, hover_entries, record_hovers, source).and_then(|shape| infer_flatten(shape, offset, call, vars, diagnostics, source))
        }
        (TorchOp::Unknown, _, _, Function) => {
            // TODO(carrascomj): check if emitting diagnostics here
            // is not too annoying
            None
        }
        // unsupported or not a torch tensor method, etc.
        _ => None,
    }
}

fn infer_tuple_elements(
    expr: &Expr,
    vars: &HashMap<Identifier, VarState>,
    func_map: &FuncMap,
    imports: &Imports,
    class_map: &ClassMap,
    call_stack: &mut Vec<Identifier>,
    diagnostics: &mut Vec<Diagnostic>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    record_hovers: bool,
    source: &str,
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> Option<Vec<Option<Shape>>> {
    match expr {
        Expr::Tuple(tuple) => Some(
            tuple
                .elts
                .iter()
                .map(|elt| {
                    infer_expr_shape(
                        elt,
                        vars,
                        func_map,
                        imports,
                        class_map,
                        call_stack,
                        diagnostics,
                        hover_entries,
                        record_hovers,
                        source,
                        module_cache.as_deref_mut(),
                        module_path,
                    )
                })
                .collect(),
        ),
        Expr::Call(call) => infer_defined_call_return(
            call,
            vars,
            func_map,
            imports,
            class_map,
            call_stack,
            diagnostics,
            hover_entries,
            record_hovers,
            source,
            module_cache,
            module_path,
        )
        .and_then(|ret| ret.tuple().map(|tuple| tuple.to_vec())),
        _ => None,
    }
}

fn method_param_offset(args: &Arguments) -> usize {
    match args.args.first() {
        Some(param) if param.def.arg.as_str() == "self" => 1,
        _ => 0,
    }
}

fn class_ref_from_constructor_call(
    call_expr: &Expr,
    class_map: &ClassMap,
    imports: &Imports,
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> Option<ClassRef> {
    let Expr::Call(call) = call_expr else {
        return None;
    };
    match call.func.as_ref() {
        Expr::Name(name) => {
            if class_map.contains_key(&name.id) {
                return Some(ClassRef {
                    name: name.id.clone(),
                    module: None,
                });
            }
            if let Some((module_name, original)) = imports.from_imports.get(&name.id)
                && module_name != "torch"
                && !module_name.starts_with("torch.")
                && let (Some(cache), Some(cur_path)) = (module_cache.as_deref_mut(), module_path)
                && let Some(module) = cache.get_module(module_name, cur_path)
                && module.class_map.contains_key(original)
            {
                return Some(ClassRef {
                    name: original.clone(),
                    module: Some(module_name.to_string()),
                });
            }
        }
        Expr::Attribute(attr) => {
            if let Expr::Name(module_ident) = attr.value.as_ref()
                && let Some(module_name) = imports.module_aliases.get(&module_ident.id)
                && module_name != "torch"
                && !module_name.starts_with("torch.")
                && let (Some(cache), Some(cur_path)) = (module_cache, module_path)
                && let Some(module) = cache.get_module(module_name, cur_path)
                && module.class_map.contains_key(&attr.attr)
            {
                return Some(ClassRef {
                    name: attr.attr.clone(),
                    module: Some(module_name.to_string()),
                });
            }
        }
        _ => {}
    }
    None
}

fn class_ref_from_annotation(
    ann: &Expr,
    imports: &Imports,
    class_map: &ClassMap,
) -> Option<ClassRef> {
    if let Expr::Name(name) = ann {
        if class_map.contains_key(&name.id) {
            return Some(ClassRef {
                name: name.id.clone(),
                module: None,
            });
        }
        if let Some((module_name, original)) = imports.from_imports.get(&name.id) {
            return Some(ClassRef {
                name: original.clone(),
                module: Some(module_name.clone()),
            });
        }
    }
    None
}

fn union_members<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) {
    if let Expr::BinOp(ExprBinOp {
        left, op, right, ..
    }) = expr
        && matches!(op, Operator::BitOr)
    {
        union_members(left, out);
        union_members(right, out);
    } else {
        out.push(expr);
    }
}

fn shape_or_class_from_union(
    ann: &Expr,
    imports: &Imports,
    class_map: &ClassMap,
) -> (Option<Shape>, Option<ClassRef>) {
    let mut members = Vec::new();
    union_members(ann, &mut members);
    members
        .into_iter()
        .find_map(|member| {
            parse_shape_annotation(member)
                .map(|shape| (Some(shape), None))
                .or_else(|| {
                    class_ref_from_annotation(member, imports, class_map)
                        .map(|class_ref| (None, Some(class_ref)))
                })
        })
        .unwrap_or((None, None))
}

fn tuple_shapes_from_annotation(
    ann: &Expr,
    imports: &Imports,
    class_map: &ClassMap,
) -> Option<Vec<Option<Shape>>> {
    let elements: Vec<&Expr> = match ann {
        Expr::Tuple(t) => t.elts.iter().collect(),
        Expr::Subscript(sub) => {
            let is_tuple = name_like(&sub.value)
                .map(|n| n.eq_ignore_ascii_case("tuple"))
                .unwrap_or(false);
            if !is_tuple {
                return None;
            }
            match &*sub.slice {
                Expr::Tuple(t) => t.elts.iter().collect(),
                other => vec![other],
            }
        }
        _ => return None,
    };
    let tuple_shapes: Vec<Option<Shape>> = elements
        .iter()
        .map(|elt| {
            parse_shape_annotation(elt)
                .or_else(|| shape_or_class_from_union(elt, imports, class_map).0)
        })
        .collect();
    if tuple_shapes.iter().any(|s| s.is_some()) {
        Some(tuple_shapes)
    } else {
        None
    }
}

/// Infer return shapes for a call, optionally simulating the callee body for diagnostics.
fn infer_call_return_from_info(
    call: &ExprCall<TextRange>,
    callee_name: &Identifier,
    callee_info: &FunctionInfo,
    callee_source: &str,
    callee_func_map: &FuncMap,
    callee_imports: &Imports,
    callee_class_map: &ClassMap,
    vars: &HashMap<Identifier, VarState>,
    func_map: &FuncMap,
    imports: &Imports,
    class_map: &ClassMap,
    call_stack: &mut Vec<Identifier>,
    diagnostics: &mut Vec<Diagnostic>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    record_hovers: bool,
    source: &str,
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
    param_offset: usize,
    emit_body_diagnostics: bool,
) -> Option<ReturnValue> {
    if call_stack.iter().any(|id| id == callee_name) {
        return None;
    }
    let mut arg_shapes: HashMap<Identifier, VarState> = HashMap::new();
    for (idx, param) in callee_info.args.args.iter().enumerate().skip(param_offset) {
        let call_idx = idx.saturating_sub(param_offset);
        if let Some(arg_expr) = call.args.get(call_idx) {
            if let Some(shape) = infer_expr_shape(
                arg_expr,
                vars,
                func_map,
                imports,
                class_map,
                call_stack,
                diagnostics,
                hover_entries,
                record_hovers,
                source,
                module_cache.as_deref_mut(),
                module_path,
            ) {
                if let Some(ann) = param
                    .def
                    .annotation
                    .as_deref()
                    .and_then(parse_shape_annotation)
                {
                    let dims_match = ann.dims.len() == shape.dims.len()
                        && ann.dims.iter().zip(shape.dims.iter()).all(|(left, right)| {
                            // less stringent equality (alpha-equivalence):
                            // behaves as a annotated aliasing; only concrete
                            // dimensions at both sides must match
                            match (left.parse::<i32>().is_ok(), right.parse::<i32>().is_ok()) {
                                (true, true) => left == right,
                                _ => true,
                            }
                        });
                    if !dims_match {
                        diagnostics.push(Diagnostic {
                            range: text_range_to_lsp(expr_text_range(arg_expr), source),
                            severity: Some(DiagnosticSeverity::ERROR),
                            code: None,
                            code_description: None,
                            source: Some("shapels".into()),
                            message: format!(
                                "Shape mismatch: annotation {} vs inferred {}",
                                ann.render(),
                                shape.render()
                            ),
                            related_information: None,
                            tags: None,
                            data: None,
                        });
                    }
                }
                arg_shapes.insert(
                    param.def.arg.clone(),
                    VarState {
                        annotated: None,
                        inferred: Some(shape),
                        class_ref: None,
                    },
                );
            } else if let Expr::Name(name) = arg_expr
                && let Some(class_ref) = vars.get(&name.id).and_then(|v| v.class_ref.clone())
            {
                arg_shapes.insert(
                    param.def.arg.clone(),
                    VarState {
                        annotated: None,
                        inferred: None,
                        class_ref: Some(class_ref),
                    },
                );
            }
        }
    }
    if let Some(ret_ann) = callee_info.returns.as_deref() {
        let annotated_return = if let Some(ret_shape) = parse_shape_annotation(ret_ann) {
            Some(ReturnValue::from_shape(Some(ret_shape)))
        } else if let Some(tuple_shapes) = tuple_shapes_from_annotation(ret_ann, imports, class_map)
        {
            Some(ReturnValue::from_tuple(tuple_shapes))
        } else {
            let (shape_union, _) = shape_or_class_from_union(ret_ann, imports, class_map);
            shape_union.map(|shape| ReturnValue::from_shape(Some(shape)))
        };
        if let Some(ret) = annotated_return {
            if emit_body_diagnostics {
                call_stack.push(callee_name.clone());
                let (mut diag, mut hovers, _) = simulate_function(
                    callee_info.args.as_ref(),
                    &callee_info.body,
                    callee_source,
                    callee_func_map,
                    callee_imports,
                    callee_class_map,
                    call_stack,
                    arg_shapes,
                    record_hovers,
                    module_cache.as_deref_mut(),
                    module_path,
                );
                diagnostics.append(&mut diag);
                if record_hovers {
                    hover_entries.append(&mut hovers);
                }
                call_stack.pop();
            }
            return Some(ret);
        }
    }
    call_stack.push(callee_name.clone());
    let (mut diag, mut hovers, ret_value) = simulate_function(
        callee_info.args.as_ref(),
        &callee_info.body,
        callee_source,
        callee_func_map,
        callee_imports,
        callee_class_map,
        call_stack,
        arg_shapes,
        false,
        module_cache,
        module_path,
    );
    diagnostics.append(&mut diag);
    if record_hovers {
        hover_entries.append(&mut hovers);
    }
    call_stack.pop();
    Some(ret_value)
}

fn infer_class_call_return(
    call: &ExprCall<TextRange>,
    class_ref: &ClassRef,
    vars: &HashMap<Identifier, VarState>,
    func_map: &FuncMap,
    imports: &Imports,
    class_map: &ClassMap,
    call_stack: &mut Vec<Identifier>,
    diagnostics: &mut Vec<Diagnostic>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    record_hovers: bool,
    source: &str,
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> Option<ReturnValue> {
    with_class_info(
        class_ref,
        source,
        func_map,
        imports,
        class_map,
        &mut module_cache,
        module_path,
        |class_info,
         callee_source,
         callee_func_map,
         callee_imports,
         callee_class_map,
         callee_path,
         module_cache| {
            if !class_info.is_torch_module {
                return None;
            }
            let forward = class_info.forward.as_ref()?;
            let param_offset = method_param_offset(&forward.args);
            infer_call_return_from_info(
                call,
                &class_ref.name,
                forward,
                callee_source,
                callee_func_map,
                callee_imports,
                callee_class_map,
                vars,
                func_map,
                imports,
                class_map,
                call_stack,
                diagnostics,
                hover_entries,
                record_hovers,
                source,
                module_cache.as_deref_mut(),
                callee_path,
                param_offset,
                false,
            )
        },
    )
    .and_then(|ret| ret)
}

fn infer_class_method_call_return(
    call: &ExprCall<TextRange>,
    class_ref: &ClassRef,
    method_name: &Identifier,
    vars: &HashMap<Identifier, VarState>,
    func_map: &FuncMap,
    imports: &Imports,
    class_map: &ClassMap,
    call_stack: &mut Vec<Identifier>,
    diagnostics: &mut Vec<Diagnostic>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    record_hovers: bool,
    source: &str,
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> Option<ReturnValue> {
    let callee_name = Identifier::from(format!("{}::{}", class_ref.name, method_name));
    with_class_info(
        class_ref,
        source,
        func_map,
        imports,
        class_map,
        &mut module_cache,
        module_path,
        |class_info,
         callee_source,
         callee_func_map,
         callee_imports,
         callee_class_map,
         callee_path,
         module_cache| {
            let method_info = class_info.methods.get(method_name)?;
            let param_offset = method_param_offset(&method_info.args);
            infer_call_return_from_info(
                call,
                &callee_name,
                method_info,
                callee_source,
                callee_func_map,
                callee_imports,
                callee_class_map,
                vars,
                func_map,
                imports,
                class_map,
                call_stack,
                diagnostics,
                hover_entries,
                record_hovers,
                source,
                module_cache.as_deref_mut(),
                callee_path,
                param_offset,
                false,
            )
        },
    )
    .and_then(|ret| ret)
}

fn infer_defined_call_return(
    call: &ExprCall<TextRange>,
    vars: &HashMap<Identifier, VarState>,
    func_map: &FuncMap,
    imports: &Imports,
    class_map: &ClassMap,
    call_stack: &mut Vec<Identifier>,
    diagnostics: &mut Vec<Diagnostic>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    record_hovers: bool,
    source: &str,
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> Option<ReturnValue> {
    if let Expr::Name(func_name) = call.func.as_ref() {
        if let Some(class_ref) = vars.get(&func_name.id).and_then(|v| v.class_ref.as_ref()) {
            return infer_class_call_return(
                call,
                class_ref,
                vars,
                func_map,
                imports,
                class_map,
                call_stack,
                diagnostics,
                hover_entries,
                record_hovers,
                source,
                module_cache.as_deref_mut(),
                module_path,
            );
        }
        if let Some((module_name, original)) = imports.from_imports.get(&func_name.id)
            && let (Some(cache), Some(cur_path)) = (module_cache.as_deref_mut(), module_path)
            && let Some(module) = cache.get_module(module_name, cur_path)
            && let Some(callee_info) = module.func_map.get(original)
        {
            return infer_call_return_from_info(
                call,
                &func_name.id,
                callee_info,
                &module.source,
                &module.func_map,
                &module.imports,
                &module.class_map,
                vars,
                func_map,
                imports,
                class_map,
                call_stack,
                diagnostics,
                hover_entries,
                record_hovers,
                source,
                module_cache.as_deref_mut(),
                Some(module.path.as_path()),
                0,
                false,
            );
        }
        if let Some(callee_info) = func_map.get(&func_name.id) {
            return infer_call_return_from_info(
                call,
                &func_name.id,
                callee_info,
                source,
                func_map,
                imports,
                class_map,
                vars,
                func_map,
                imports,
                class_map,
                call_stack,
                diagnostics,
                hover_entries,
                record_hovers,
                source,
                module_cache.as_deref_mut(),
                module_path,
                0,
                false,
            );
        }
    }
    if let Expr::Attribute(attr) = call.func.as_ref()
        && let Expr::Name(base_name) = attr.value.as_ref()
        && let Some(class_ref) = vars.get(&base_name.id).and_then(|v| v.class_ref.as_ref())
    {
        return infer_class_method_call_return(
            call,
            class_ref,
            &attr.attr,
            vars,
            func_map,
            imports,
            class_map,
            call_stack,
            diagnostics,
            hover_entries,
            record_hovers,
            source,
            module_cache.as_deref_mut(),
            module_path,
        );
    }
    if let Expr::Attribute(attr) = call.func.as_ref()
        && let Expr::Name(module_ident) = attr.value.as_ref()
        && let Some(module_name) = imports.module_aliases.get(&module_ident.id)
        && module_name != "torch"
        && let (Some(cache), Some(cur_path)) = (module_cache.as_deref_mut(), module_path)
        && let Some(module) = cache.get_module(module_name, cur_path)
        && let Some(callee_info) = module.func_map.get(&attr.attr)
    {
        return infer_call_return_from_info(
            call,
            &attr.attr,
            callee_info,
            &module.source,
            &module.func_map,
            &module.imports,
            &module.class_map,
            vars,
            func_map,
            imports,
            class_map,
            call_stack,
            diagnostics,
            hover_entries,
            record_hovers,
            source,
            module_cache,
            Some(module.path.as_path()),
            0,
            false,
        );
    }
    None
}

fn tensor_or_shape_as_arg(
    is_size: bool,
    vars: &HashMap<Identifier, VarState>,
    func_map: &FuncMap,
    imports: &Imports,
    class_map: &ClassMap,
    call_stack: &mut Vec<Identifier>,
    diagnostics: &mut Vec<Diagnostic>,
    hover_entries: &mut Vec<(Range, HoverInfo)>,
    record_hovers: bool,
    source: &str,
    module_cache: &mut Option<&mut ModuleCache>,
    module_path: Option<&Path>,
    call: &ExprCall,
) -> Option<Shape> {
    match call.args.first() {
        // a torch.Tensor.shape might be the first argument
        Some(arg0 @ Expr::Attribute(_)) if is_size => infer_expr_shape(
            arg0,
            vars,
            func_map,
            imports,
            class_map,
            call_stack,
            diagnostics,
            hover_entries,
            record_hovers,
            source,
            module_cache.as_deref_mut(),
            module_path,
        ),
        // zeros_like, ones_like etc. accept a tensor as first arg
        Some(arg0 @ Expr::Name(_)) if !is_size => infer_expr_shape(
            arg0,
            vars,
            func_map,
            imports,
            class_map,
            call_stack,
            diagnostics,
            hover_entries,
            record_hovers,
            source,
            module_cache.as_deref_mut(),
            module_path,
        ),
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

/// An operation might be a function `torch.FUNCTION` (might be imported and
/// aliased) or a `torch.Tensor.METHOD`.
enum TorchOpKind {
    /// `torch.FUNCTION`
    Function,
    /// `torch.Tensor.METHOD`
    Method,
}

fn function_or_method<R>(expr: &Expr<R>, imports: &Imports) -> TorchOpKind {
    match expr {
        Expr::Name(n)
            if n.id.as_str() == "torch"
                || imports.torch_aliases.contains(&n.id)
                || imports.torch_nn_functional_aliases.contains(&n.id) =>
        {
            TorchOpKind::Function
        }
        _ => TorchOpKind::Method,
    }
}

fn is_alias_of(canonical: &str, ident: &Identifier, imports: &Imports) -> bool {
    imports
        .func_aliases
        .get(canonical)
        .map(|set| set.contains(ident))
        .unwrap_or(false)
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
                    class_ref: None,
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
                class_ref: None,
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
    provided: Option<&mut HashMap<Identifier, VarState>>,
    record_hovers: bool,
    imports: &Imports,
    class_map: &ClassMap,
) {
    for arg in &args.args {
        let ann_shape = arg
            .def
            .annotation
            .as_ref()
            .and_then(|expr| parse_shape_annotation(expr.as_ref()));
        let (union_shape, union_class) = arg
            .def
            .annotation
            .as_ref()
            .map(|ann| shape_or_class_from_union(ann.as_ref(), imports, class_map))
            .unwrap_or((None, None));
        let range = text_range_to_lsp(arg.def.range, source);
        let provided_state = provided.as_ref().and_then(|p| p.get(&arg.def.arg));
        let state = VarState {
            annotated: ann_shape.clone().or(union_shape),
            inferred: provided_state.and_then(|s| s.inferred.clone()),
            class_ref: provided_state
                .and_then(|s| s.class_ref.clone())
                .or_else(|| {
                    arg.def
                        .annotation
                        .as_ref()
                        .and_then(|ann| class_ref_from_annotation(ann.as_ref(), imports, class_map))
                })
                .or(union_class),
        };

        if state.annotated.is_some() || state.inferred.is_some() || state.class_ref.is_some() {
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
        Expr::BoolOp(b) => b.range,
        Expr::NamedExpr(n) => n.range,
        Expr::BinOp(b) => b.range,
        Expr::UnaryOp(u) => u.range,
        Expr::Lambda(l) => l.range,
        Expr::IfExp(i) => i.range,
        Expr::Dict(d) => d.range,
        Expr::Set(s) => s.range,
        Expr::ListComp(l) => l.range,
        Expr::SetComp(s) => s.range,
        Expr::DictComp(d) => d.range,
        Expr::GeneratorExp(g) => g.range,
        Expr::Await(a) => a.range,
        Expr::Yield(y) => y.range,
        Expr::YieldFrom(y) => y.range,
        Expr::Compare(c) => c.range,
        Expr::Call(c) => c.range,
        Expr::FormattedValue(f) => f.range,
        Expr::Subscript(s) => s.range,
        Expr::Attribute(a) => a.range,
        Expr::Starred(s) => s.range,
        Expr::Constant(c) => c.range,
        Expr::JoinedStr(j) => j.range,
        Expr::List(l) => l.range,
        Expr::Tuple(t) => t.range,
        Expr::Slice(s) => s.range,
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
    // first check for positional argument, then named argument
    (call.args.get(as_positional)).or(call
        .keywords
        .iter()
        .find(|kw| kw.arg.as_deref() == Some(name_arg))
        .map(|kw| &kw.value))
}

fn get_dtype<'expr, R>(dtype_expr: &'expr Expr<R>, imports: &Imports) -> Option<&'expr str> {
    match dtype_expr {
        Expr::Constant(constant) => match &constant.value {
            Constant::Str(string_dtype) if TORCH_DTYPES.contains(string_dtype.as_str()) => {
                Some(string_dtype.as_str())
            }
            _ => Some("Float"),
        },
        Expr::Name(name) => Some(name.id.as_str()),
        Expr::Attribute(attr)
            if matches!(
                function_or_method(attr.value.as_ref(), imports),
                TorchOpKind::Function
            ) && TORCH_DTYPES.contains(attr.attr.as_str()) =>
        {
            Some(attr.attr.as_str())
        }
        _ => Some("Float"),
    }
}
