//! Module loading and cross-file class resolution.
//!
//! This file keeps the parsed metadata needed to:
//! - resolve imported user-defined modules,
//! - follow `self.<attr>` references initialized in `__init__`,
//! - and cache that work across LSP requests.

use crate::VarState;
use crate::context::ContextRef;
use crate::expr_text_range;
use crate::infer::infer_creation_size;
use crate::infer_expr_shape;
use crate::normalize_return_annotations;
use crate::op_groups::{Imports, TorchOp, collect_imports};
use crate::text_range_to_lsp;
use crate::torch_nn::{TorchNNModule, module_from_constructor_call};
use lsp_types::Diagnostic;
use lsp_types::DiagnosticSeverity;
use rustpython_parser::Parse;
use rustpython_parser::ast::{Arguments, Expr, ExprCall, Identifier, Stmt, Suite};
use std::cell::OnceCell;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::rc::Rc;

/// Lightweight copy of a function definition used during symbolic dispatch.
#[derive(Clone)]
pub(crate) struct FunctionInfo {
    pub(crate) args: Box<Arguments>,
    pub(crate) body: Vec<Stmt>,
    pub(crate) returns: Option<Box<Expr>>,
}

pub(crate) type FuncMap = HashMap<Identifier, FunctionInfo>;

/// Reference to a user-defined class, either in the current module or an import.
#[derive(Debug, Clone)]
pub(crate) struct ClassRef {
    pub(crate) name: Identifier,
    pub(crate) module: Option<String>,
}

/// Cached metadata for a class plus lazily discovered `self.<attr>` states.
#[derive(Clone)]
pub(crate) struct ClassInfo {
    pub(crate) is_torch_module: bool,
    pub(crate) forward_name: Option<Identifier>,
    pub(crate) methods: HashMap<Identifier, FunctionInfo>,
    self_attr_states: OnceCell<HashMap<Identifier, VarState>>,
    self_attr_modules: OnceCell<HashMap<Identifier, Rc<ResolvedModule>>>,
}

pub(crate) type ClassMap = HashMap<Identifier, ClassInfo>;

#[derive(Debug, Clone)]
pub(crate) enum ResolvedModule {
    User(ClassRef),
    Builtin(TorchNNModule),
}

impl ResolvedModule {
    pub(crate) fn as_user(&self) -> Option<&ClassRef> {
        match self {
            Self::User(class_ref) => Some(class_ref),
            Self::Builtin(_) => None,
        }
    }
}

/// Parsed module data reused by cross-file inference.
pub(crate) struct CachedModule {
    pub(crate) file_path: PathBuf,
    pub(crate) source: String,
    pub(crate) body: Suite,
    pub(crate) func_map: FuncMap,
    pub(crate) imports: Imports,
    pub(crate) class_map: ClassMap,
}

/// Incremental cache for parsed modules, keyed by normalized file path.
pub struct ModuleCache {
    modules: HashMap<PathBuf, Rc<CachedModule>>,
    resolved_paths: HashMap<(PathBuf, String), Option<PathBuf>>,
    dirty_files: HashSet<PathBuf>,
    source_overrides: HashMap<PathBuf, String>,
    project_roots: HashSet<PathBuf>,
    cwd: PathBuf,
}

impl ModuleCache {
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
            resolved_paths: HashMap::new(),
            dirty_files: HashSet::new(),
            source_overrides: HashMap::new(),
            project_roots: HashSet::new(),
            cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        }
    }

    pub fn update_file_source(&mut self, file_path: &Path, source: String) {
        self.remember_path(file_path);
        let file_path = self.normalize_file_path(file_path);
        self.source_overrides.insert(file_path.clone(), source);
        self.dirty_files.insert(file_path);
    }

    pub fn mark_file_changed(&mut self, file_path: &Path) {
        self.remember_path(file_path);
        self.dirty_files.insert(self.normalize_file_path(file_path));
    }

    pub(crate) fn project_root_for(&mut self, current_file: &Path) -> Option<PathBuf> {
        let current_file = self.normalize_file_path(current_file);
        if let Some(root) = self
            .project_roots
            .iter()
            .filter(|root| current_file.starts_with(root))
            .max_by_key(|root| root.components().count())
        {
            return Some(root.clone());
        }
        let root = find_project_root(&current_file).map(|root| self.normalize_file_path(&root));
        if let Some(root) = &root {
            self.project_roots.insert(root.clone());
        }
        root
    }

    fn remember_path(&mut self, path: &Path) {
        let _ = self.project_root_for(path);
    }

    fn normalize_file_path(&self, path: &Path) -> PathBuf {
        normalize_file_path(path, &self.cwd)
    }

    fn load_module(&mut self, file_path: &Path, source: String) -> Option<Rc<CachedModule>> {
        let module = Suite::parse(&source, &file_path.to_string_lossy()).ok()?;
        let project_root = self.project_root_for(file_path);
        let imports = collect_imports(&module, Some(file_path), project_root.as_deref());
        let mut func_map = FuncMap::new();
        collect_function_defs(&module, &mut func_map);
        let class_map = collect_class_defs(&module, &imports);
        Some(Rc::new(CachedModule {
            file_path: file_path.to_path_buf(),
            source,
            body: module,
            func_map,
            imports,
            class_map,
        }))
    }

    fn load_normalized_module(
        &mut self,
        file_path: &Path,
        source: String,
    ) -> Option<Rc<CachedModule>> {
        self.load_module(file_path, source)
    }

    fn normalized_source(source: &str) -> String {
        normalize_return_annotations(source).into_owned()
    }

    pub(crate) fn get_file_module(
        &mut self,
        file_path: &Path,
        source: Option<&str>,
    ) -> Option<Rc<CachedModule>> {
        let file_path = self.normalize_file_path(file_path);
        self.remember_path(&file_path);

        if let Some(source) = source {
            let normalized = Self::normalized_source(source);
            if !self.dirty_files.contains(&file_path)
                && let Some(cached) = self.modules.get(&file_path)
                && cached.source == normalized
            {
                return Some(Rc::clone(cached));
            }
            let cached = self.load_normalized_module(&file_path, normalized)?;
            self.modules.insert(file_path.clone(), Rc::clone(&cached));
            self.dirty_files.remove(&file_path);
            return Some(cached);
        }

        if !self.dirty_files.contains(&file_path)
            && let Some(cached) = self.modules.get(&file_path)
        {
            return Some(Rc::clone(cached));
        }
        let source = self
            .source_overrides
            .get(&file_path)
            .cloned()
            .or_else(|| fs::read_to_string(&file_path).ok())?;
        let cached = self.load_normalized_module(&file_path, Self::normalized_source(&source))?;
        self.modules.insert(file_path.clone(), Rc::clone(&cached));
        self.dirty_files.remove(&file_path);
        Some(cached)
    }

    pub(crate) fn get_module(
        &mut self,
        module_name: &str,
        current_file: &Path,
    ) -> Option<Rc<CachedModule>> {
        let current_file = self.normalize_file_path(current_file);
        self.remember_path(&current_file);
        let resolve_key = (current_file.clone(), module_name.to_string());
        let file_path = if let Some(cached_path) = self.resolved_paths.get(&resolve_key) {
            cached_path.clone()?
        } else {
            let project_root = self.project_root_for(&current_file);
            let path = resolve_module_path(module_name, &current_file, project_root.as_deref());
            let normalized = path.as_ref().map(|path| self.normalize_file_path(path));
            self.resolved_paths.insert(resolve_key, normalized.clone());
            normalized?
        };
        self.get_file_module(&file_path, None)
    }
}

impl Default for ModuleCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns `1` for instance methods whose first argument is `self`.
pub(crate) fn method_param_offset(args: &Arguments) -> usize {
    match args.args.first() {
        Some(param) if param.def.arg.as_str() == "self" => 1,
        _ => 0,
    }
}

/// Collect all top-level function definitions in a module body.
pub(crate) fn collect_function_defs(body: &[Stmt], func_map: &mut FuncMap) {
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
    forward_name: Option<Identifier>,
    methods: HashMap<Identifier, FunctionInfo>,
    base_names: Vec<Identifier>,
    direct_torch: bool,
}

fn is_torch_nn_module_base(expr: &Expr, imports: &Imports) -> bool {
    let Expr::Attribute(attr) = expr else {
        return false;
    };
    attr.attr.as_str() == "Module" && imports.is_torch_nn_namespace_expr(attr.value.as_ref())
}

/// Collect class definitions and mark which ones inherit from `torch.nn.Module`.
pub(crate) fn collect_class_defs(body: &[Stmt], imports: &Imports) -> ClassMap {
    let mut defs = HashMap::new();
    for stmt in body {
        if let Stmt::ClassDef(class_def) = stmt {
            let mut methods = HashMap::new();
            let mut forward_name = None;
            for stmt in &class_def.body {
                if let Stmt::FunctionDef(func) = stmt {
                    let info = FunctionInfo {
                        args: func.args.clone(),
                        body: func.body.clone(),
                        returns: func.returns.clone(),
                    };
                    if func.name.as_str() == "forward" {
                        forward_name = Some(func.name.clone());
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
                    forward_name,
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
        for (name, info) in &defs {
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

    defs.into_iter()
        .map(|(name, info)| {
            (
                name.clone(),
                ClassInfo {
                    is_torch_module: is_torch.get(&name).copied().unwrap_or(false),
                    forward_name: info.forward_name,
                    methods: info.methods,
                    self_attr_states: OnceCell::new(),
                    self_attr_modules: OnceCell::new(),
                },
            )
        })
        .collect()
}

pub(crate) enum ResolvedModuleRef<'a> {
    Borrowed(&'a ResolvedModule),
    Owned(ResolvedModule),
    Shared(Rc<ResolvedModule>),
}

impl<'a> ResolvedModuleRef<'a> {
    pub(crate) fn as_ref(&self) -> &ResolvedModule {
        match self {
            Self::Borrowed(module) => module,
            Self::Owned(module) => module,
            Self::Shared(module) => module.as_ref(),
        }
    }

    pub(crate) fn into_owned(self) -> ResolvedModule {
        match self {
            Self::Borrowed(module) => module.clone(),
            Self::Owned(module) => module,
            Self::Shared(module) => module.as_ref().clone(),
        }
    }
}

/// Look up the metadata behind a [`ClassRef`], loading the defining module if needed.
pub(crate) fn with_class_info<R, F>(
    class_ref: &ClassRef,
    mut context: ContextRef,
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
    let (_, func_map, imports, class_map, _, _, _, source, mut module_cache, module_path) =
        context.module_infer_parts();
    with_class_info_parts(
        class_ref,
        source,
        func_map,
        imports,
        class_map,
        &mut module_cache,
        module_path,
        f,
    )
}

pub(crate) fn with_class_info_parts<R, F>(
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
                Some(module.file_path.as_path()),
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

/// Resolve a class annotation such as `UserLinear` or an imported alias.
pub(crate) fn class_ref_from_annotation(
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

/// Resolve a module-valued expression from either a constructor call or a bound variable.
pub(crate) fn resolved_module_from_expr<'a>(
    expr: &Expr,
    vars: &'a HashMap<Identifier, VarState>,
    source: &str,
    func_map: &FuncMap,
    imports: &Imports,
    class_map: &ClassMap,
    diagnostics: &mut Vec<Diagnostic>,
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> Option<ResolvedModuleRef<'a>> {
    if let Some(resolved_module) = module_from_constructor_call(
        expr,
        class_map,
        imports,
        module_cache.as_deref_mut(),
        module_path,
    ) {
        if matches!(
            resolved_module,
            ResolvedModule::Builtin(TorchNNModule::Unknown)
        ) {
            // TODO(carrascomj): consider if this is too annoying for users
            diagnostics.push(Diagnostic {
                range: text_range_to_lsp(expr_text_range(expr), source),
                severity: Some(DiagnosticSeverity::INFORMATION),
                code: None,
                code_description: None,
                source: Some("shapels".into()),
                message: "Module not yet understood by shapels (treated as Noop)".to_string(),
                related_information: None,
                tags: None,
                data: None,
            });
        }

        return Some(ResolvedModuleRef::Owned(resolved_module));
    }
    resolve_module_expr(
        expr,
        vars,
        source,
        func_map,
        imports,
        class_map,
        module_cache,
        module_path,
    )
}

fn self_attr_name(expr: &Expr) -> Option<Identifier> {
    let Expr::Attribute(attr) = expr else {
        return None;
    };
    let Expr::Name(name) = attr.value.as_ref() else {
        return None;
    };
    (name.id.as_str() == "self").then(|| attr.attr.clone())
}

fn self_attr_storage_key(attr: &Identifier) -> Identifier {
    Identifier::from(format!("self.{}", attr.as_str()))
}

fn shape_state(shape: crate::Shape) -> VarState {
    VarState {
        annotated: None,
        inferred: Some(shape),
        resolved_module: None,
        callable: None,
    }
}

fn module_state(resolved_module: ResolvedModule) -> VarState {
    VarState {
        annotated: None,
        inferred: None,
        resolved_module: Some(resolved_module),
        callable: None,
    }
}

pub(crate) fn imported_class_ref(
    module_name: &str,
    class_name: &Identifier,
    module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> Option<ClassRef> {
    if module_name == "torch" || module_name.starts_with("torch.") {
        return None;
    }
    let (Some(cache), Some(cur_path)) = (module_cache, module_path) else {
        return None;
    };
    let module = cache.get_module(module_name, cur_path)?;
    module.class_map.contains_key(class_name).then(|| ClassRef {
        name: class_name.clone(),
        module: Some(module_name.to_string()),
    })
}

fn torch_call_op(func: &Expr, imports: &Imports) -> TorchOp {
    match func {
        Expr::Name(name) => TorchOp::as_call(&name.id, imports),
        Expr::Attribute(attr) if imports.is_torch_namespace_expr(attr.value.as_ref()) => {
            TorchOp::from_attr(attr.attr.as_str())
        }
        _ => TorchOp::Unknown,
    }
}

fn creation_call_shape(
    call: &ExprCall,
    vars: &HashMap<Identifier, VarState>,
    imports: &Imports,
    source: &str,
) -> Option<crate::Shape> {
    let torch_op = torch_call_op(call.func.as_ref(), imports);
    let TorchOp::Creation { is_size } = torch_op else {
        return None;
    };
    let mut diagnostics = Vec::new();
    infer_creation_size(call, vars, &mut diagnostics, source, None, None, is_size)
}

/// Infer the cached state for a value assigned in `__init__`.
///
/// Only class constructors, `torch.nn.Parameter(...)`, and tensor creation calls are tracked.
fn infer_tracked_state_from_expr(
    expr: &Expr,
    vars: &HashMap<Identifier, VarState>,
    self_attrs: &HashMap<Identifier, VarState>,
    class_map: &ClassMap,
    imports: &Imports,
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
    source: &str,
) -> Option<VarState> {
    if let Some(resolved_module) = module_from_constructor_call(
        expr,
        class_map,
        imports,
        module_cache.as_deref_mut(),
        module_path,
    ) {
        return Some(module_state(resolved_module));
    }

    if let Expr::Name(name) = expr {
        return vars.get(&name.id).cloned();
    }

    if let Some(attr_name) = self_attr_name(expr) {
        return vars
            .get(&self_attr_storage_key(&attr_name))
            .cloned()
            .or_else(|| self_attrs.get(&attr_name).cloned());
    }

    if let Expr::Call(call) = expr {
        if imports.is_torch_nn_storage_constructor(call.func.as_ref()) {
            let first_arg = call.args.first()?;
            return infer_tracked_state_from_expr(
                first_arg,
                vars,
                self_attrs,
                class_map,
                imports,
                module_cache.as_deref_mut(),
                module_path,
                source,
            )
            .and_then(|state| state.annotated.or(state.inferred).map(shape_state));
        }

        if let Some(shape) = creation_call_shape(call, vars, imports, source) {
            return Some(shape_state(shape));
        }
    }

    let mut diagnostics = Vec::new();
    let mut hover_entries = Vec::new();
    let mut call_stack = Vec::new();
    infer_expr_shape(
        expr,
        vars,
        &FuncMap::new(),
        imports,
        class_map,
        &mut call_stack,
        &mut diagnostics,
        &mut hover_entries,
        false,
        source,
        module_cache,
        module_path,
    )
    .map(shape_state)
}

fn update_tracked_binding(
    target: &Expr,
    value: &Expr,
    vars: &mut HashMap<Identifier, VarState>,
    self_attrs: &mut HashMap<Identifier, VarState>,
    class_map: &ClassMap,
    imports: &Imports,
    module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
    source: &str,
) {
    let state = infer_tracked_state_from_expr(
        value,
        vars,
        self_attrs,
        class_map,
        imports,
        module_cache,
        module_path,
        source,
    );
    if let Some(attr_name) = self_attr_name(target) {
        if let Some(state) = state {
            vars.insert(self_attr_storage_key(&attr_name), state.clone());
            self_attrs.insert(attr_name, state);
        } else {
            vars.remove(&self_attr_storage_key(&attr_name));
            self_attrs.remove(&attr_name);
        }
    } else if let Expr::Name(name) = target {
        if let Some(state) = state {
            vars.insert(name.id.clone(), state);
        } else {
            vars.remove(&name.id);
        }
    }
}

/// Scan `__init__` once and cache every tracked `self.<attr>` assignment.
fn collect_self_attr_states_from_init(
    init: Option<&FunctionInfo>,
    source: &str,
    class_map: &ClassMap,
    imports: &Imports,
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> HashMap<Identifier, VarState> {
    let Some(init) = init else {
        return HashMap::new();
    };
    let mut vars = HashMap::from([(Identifier::from("self"), VarState::default())]);
    let mut states = HashMap::new();

    for stmt in &init.body {
        match stmt {
            Stmt::Assign(assign) if assign.targets.len() == 1 => update_tracked_binding(
                &assign.targets[0],
                &assign.value,
                &mut vars,
                &mut states,
                class_map,
                imports,
                module_cache.as_deref_mut(),
                module_path,
                source,
            ),
            Stmt::AnnAssign(assign) if assign.value.is_some() => update_tracked_binding(
                assign.target.as_ref(),
                assign.value.as_deref().unwrap(),
                &mut vars,
                &mut states,
                class_map,
                imports,
                module_cache.as_deref_mut(),
                module_path,
                source,
            ),
            _ => {}
        }
    }

    states
}

fn get_or_collect_self_attr_state(
    class_info: &ClassInfo,
    target_attr: &Identifier,
    source: &str,
    class_map: &ClassMap,
    imports: &Imports,
    module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> Option<VarState> {
    self_attr_states(
        class_info,
        source,
        class_map,
        imports,
        module_cache,
        module_path,
    )
    .get(target_attr)
    .cloned()
}

fn self_attr_states<'a>(
    class_info: &'a ClassInfo,
    source: &str,
    class_map: &ClassMap,
    imports: &Imports,
    module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> &'a HashMap<Identifier, VarState> {
    class_info.self_attr_states.get_or_init(|| {
        collect_self_attr_states_from_init(
            class_info.methods.get(&Identifier::from("__init__")),
            source,
            class_map,
            imports,
            module_cache,
            module_path,
        )
    })
}

fn get_or_collect_self_attr_module(
    class_info: &ClassInfo,
    target_attr: &Identifier,
    source: &str,
    class_map: &ClassMap,
    imports: &Imports,
    module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> Option<Rc<ResolvedModule>> {
    class_info
        .self_attr_modules
        .get_or_init(|| {
            self_attr_states(
                class_info,
                source,
                class_map,
                imports,
                module_cache,
                module_path,
            )
            .iter()
            .filter_map(|(attr, state)| {
                state
                    .resolved_module
                    .as_ref()
                    .cloned()
                    .map(|module| (attr.clone(), Rc::new(module)))
            })
            .collect()
        })
        .get(target_attr)
        .cloned()
}

fn resolve_cached_self_attr_state(
    base_class_ref: &ClassRef,
    target_attr: &Identifier,
    context: ContextRef,
) -> Option<VarState> {
    with_class_info(
        base_class_ref,
        context,
        |class_info,
         callee_source,
         _callee_func_map,
         callee_imports,
         callee_class_map,
         callee_path,
         module_cache| {
            get_or_collect_self_attr_state(
                class_info,
                target_attr,
                callee_source,
                callee_class_map,
                callee_imports,
                module_cache.as_deref_mut(),
                callee_path,
            )
        },
    )
    .flatten()
}

fn resolve_cached_self_attr_module(
    base_class_ref: &ClassRef,
    target_attr: &Identifier,
    source: &str,
    func_map: &FuncMap,
    imports: &Imports,
    class_map: &ClassMap,
    module_cache: &mut Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> Option<Rc<ResolvedModule>> {
    with_class_info_parts(
        base_class_ref,
        source,
        func_map,
        imports,
        class_map,
        module_cache,
        module_path,
        |class_info,
         callee_source,
         _callee_func_map,
         callee_imports,
         callee_class_map,
         callee_path,
         module_cache| {
            get_or_collect_self_attr_module(
                class_info,
                target_attr,
                callee_source,
                callee_class_map,
                callee_imports,
                module_cache.as_deref_mut(),
                callee_path,
            )
        },
    )
    .flatten()
}

pub(crate) fn self_attr_module_from_self(
    target_attr: &Identifier,
    vars: &HashMap<Identifier, VarState>,
    source: &str,
    func_map: &FuncMap,
    imports: &Imports,
    class_map: &ClassMap,
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> Option<Rc<ResolvedModule>> {
    let self_class_ref = vars
        .get(&Identifier::from("self"))
        .and_then(|state| state.resolved_module.as_ref())
        .and_then(ResolvedModule::as_user)?;
    resolve_cached_self_attr_module(
        self_class_ref,
        target_attr,
        source,
        func_map,
        imports,
        class_map,
        &mut module_cache,
        module_path,
    )
}

/// Resolve an attribute access such as `self.weight` to the cached state from `__init__`.
pub(crate) fn attr_state_from_expr(expr: &Expr, mut context: ContextRef) -> Option<VarState> {
    let Expr::Attribute(attr) = expr else {
        return None;
    };
    let base_class_ref = {
        let (vars, func_map, imports, class_map, _, _, _, source, module_cache, module_path) =
            context.module_infer_parts();
        let base_module = resolve_module_expr(
            attr.value.as_ref(),
            vars,
            source,
            func_map,
            imports,
            class_map,
            module_cache,
            module_path,
        )?;
        base_module.as_ref().as_user()?.clone()
    };
    resolve_cached_self_attr_state(&base_class_ref, &attr.attr, context.reborrow())
}

fn resolve_module_expr<'a>(
    expr: &Expr,
    vars: &'a HashMap<Identifier, VarState>,
    source: &str,
    func_map: &FuncMap,
    imports: &Imports,
    class_map: &ClassMap,
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> Option<ResolvedModuleRef<'a>> {
    match expr {
        Expr::Name(name) => vars
            .get(&name.id)
            .and_then(|v| v.resolved_module.as_ref())
            .map(ResolvedModuleRef::Borrowed),
        Expr::Attribute(attr) => {
            let base_module = resolve_module_expr(
                attr.value.as_ref(),
                vars,
                source,
                func_map,
                imports,
                class_map,
                module_cache.as_deref_mut(),
                module_path,
            )?;
            resolve_cached_self_attr_module(
                base_module.as_ref().as_user()?,
                &attr.attr,
                source,
                func_map,
                imports,
                class_map,
                &mut module_cache,
                module_path,
            )
            .map(ResolvedModuleRef::Shared)
        }
        _ => None,
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

fn normalize_file_path(path: &Path, cwd: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            std::path::Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            std::path::Component::RootDir => normalized.push(component.as_os_str()),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            std::path::Component::Normal(part) => normalized.push(part),
        }
    }
    normalized
}

fn should_skip_module_search_dir(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(".git" | "target" | "__pycache__" | "node_modules")
    )
}

fn find_module_path_recursively(root: &Path, relative: &Path) -> Option<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if should_skip_module_search_dir(&dir) {
            continue;
        }
        let candidate = dir.join(relative);
        if candidate.exists() {
            return Some(candidate);
        }
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            }
        }
    }
    None
}

/// Resolve an import like `a.b.c` to the python file that defines it.
fn resolve_module_path(
    module: &str,
    current_file: &Path,
    project_root: Option<&Path>,
) -> Option<PathBuf> {
    let mut search_roots = Vec::new();

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

    if let Ok(path_var) = std::env::var("PATH") {
        for mut path_entry in std::env::split_paths(&path_var) {
            if !path_entry.to_string_lossy().contains(".venv") {
                continue;
            }
            while let Some(parent) = path_entry.parent() {
                if let Some(name) = parent.file_name()
                    && name.to_string_lossy().contains(".venv")
                {
                    let venv_dir = parent.to_path_buf();
                    if let Some(project_root) = venv_dir.parent() {
                        search_roots.push(project_root.to_path_buf());
                    }
                    let lib_dir = venv_dir.join("lib");
                    if lib_dir.exists()
                        && let Ok(entries) = fs::read_dir(&lib_dir)
                    {
                        for entry in entries.flatten() {
                            let fname = entry.file_name();
                            if fname.to_string_lossy().starts_with("python") {
                                let site_packages = entry.path().join("site-packages");
                                if site_packages.exists() {
                                    search_roots.push(site_packages);
                                }
                            }
                        }
                    }
                    break;
                }
                path_entry = parent.to_path_buf();
            }
        }
    }

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
        if let Some(root_name) = root.file_name()
            && root_name
                == parts
                    .first()
                    .map(std::ffi::OsStr::new)
                    .unwrap_or_else(|| std::ffi::OsStr::new(""))
            && parts.len() > 1
        {
            let mut nested_base = root.clone();
            for part in parts.iter().skip(1) {
                nested_base.push(part);
            }
            let file_candidate = nested_base.with_extension("py");
            if file_candidate.exists() {
                return Some(file_candidate);
            }
            let init_candidate = nested_base.join("__init__.py");
            if init_candidate.exists() {
                return Some(init_candidate);
            }
        }
        let mut relative = PathBuf::new();
        for part in &parts {
            relative.push(part);
        }
        if let Some(found) = find_module_path_recursively(&root, &relative.with_extension("py")) {
            return Some(found);
        }
        if let Some(found) = find_module_path_recursively(&root, &relative.join("__init__.py")) {
            return Some(found);
        }
    }
    None
}
