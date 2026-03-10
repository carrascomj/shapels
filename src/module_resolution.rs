use crate::VarState;
use crate::normalize_return_annotations;
use crate::op_groups::{Imports, collect_imports};
use rustpython_parser::Parse;
use rustpython_parser::ast::{Arguments, Expr, Identifier, Stmt, Suite};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub(crate) struct FunctionInfo {
    pub(crate) args: Box<Arguments>,
    pub(crate) body: Vec<Stmt>,
    pub(crate) returns: Option<Box<Expr>>,
}

pub(crate) type FuncMap = HashMap<Identifier, FunctionInfo>;

#[derive(Debug, Clone)]
pub(crate) struct ClassRef {
    pub(crate) name: Identifier,
    pub(crate) module: Option<String>,
}

#[derive(Clone)]
pub(crate) struct ClassInfo {
    pub(crate) is_torch_module: bool,
    pub(crate) init: Option<FunctionInfo>,
    pub(crate) forward: Option<FunctionInfo>,
    pub(crate) methods: HashMap<Identifier, FunctionInfo>,
}

pub(crate) type ClassMap = HashMap<Identifier, ClassInfo>;

#[derive(Clone)]
pub(crate) struct CachedModule {
    pub(crate) path: PathBuf,
    pub(crate) source: String,
    pub(crate) func_map: FuncMap,
    pub(crate) imports: Imports,
    pub(crate) class_map: ClassMap,
}

pub(crate) struct ModuleCache {
    modules: HashMap<String, CachedModule>,
    project_root: Option<PathBuf>,
}

impl ModuleCache {
    pub(crate) fn new(current_file: &Path) -> Self {
        let project_root = find_project_root(current_file);
        Self {
            modules: HashMap::new(),
            project_root,
        }
    }

    pub(crate) fn project_root(&self) -> Option<&Path> {
        self.project_root.as_deref()
    }

    pub(crate) fn get_module(
        &mut self,
        module_name: &str,
        current_file: &Path,
    ) -> Option<CachedModule> {
        if let Some(cached) = self.modules.get(module_name) {
            return Some(cached.clone());
        }
        let path = resolve_module_path(module_name, current_file, self.project_root.as_deref())?;
        let source = fs::read_to_string(&path).ok()?;
        let normalized = normalize_return_annotations(&source);
        let source = normalized.into_owned();
        let module = Suite::parse(&source, module_name).ok()?;
        let imports = collect_imports(&module, Some(&path), self.project_root());
        let mut func_map = FuncMap::new();
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

pub(crate) fn method_param_offset(args: &Arguments) -> usize {
    match args.args.first() {
        Some(param) if param.def.arg.as_str() == "self" => 1,
        _ => 0,
    }
}

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
    init: Option<FunctionInfo>,
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

pub(crate) fn collect_class_defs(body: &[Stmt], imports: &Imports) -> ClassMap {
    let mut defs = HashMap::new();
    for stmt in body {
        if let Stmt::ClassDef(class_def) = stmt {
            let mut methods = HashMap::new();
            let mut init = None;
            let mut forward = None;
            for stmt in &class_def.body {
                if let Stmt::FunctionDef(func) = stmt {
                    let info = FunctionInfo {
                        args: func.args.clone(),
                        body: func.body.clone(),
                        returns: func.returns.clone(),
                    };
                    if func.name.as_str() == "__init__" {
                        init = Some(info.clone());
                    }
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
                    init,
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
                    init: info.init,
                    forward: info.forward,
                    methods: info.methods,
                },
            )
        })
        .collect()
}

pub(crate) fn with_class_info<R, F>(
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

pub(crate) fn class_ref_from_expr(
    expr: &Expr,
    vars: &HashMap<Identifier, VarState>,
    source: &str,
    func_map: &FuncMap,
    imports: &Imports,
    class_map: &ClassMap,
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> Option<ClassRef> {
    if let Some(class_ref) = class_ref_from_constructor_call(
        expr,
        class_map,
        imports,
        module_cache.as_deref_mut(),
        module_path,
    ) {
        return Some(class_ref);
    }
    resolve_class_ref_expr(
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

fn collect_self_class_refs_from_init(
    init: Option<&FunctionInfo>,
    class_map: &ClassMap,
    imports: &Imports,
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> HashMap<Identifier, ClassRef> {
    let mut self_attrs = HashMap::new();
    let Some(init) = init else {
        return self_attrs;
    };

    for stmt in &init.body {
        let maybe_attr_and_value = match stmt {
            Stmt::Assign(assign) if assign.targets.len() == 1 => self_attr_name(&assign.targets[0])
                .map(|attr_name| (attr_name, assign.value.as_ref())),
            Stmt::AnnAssign(assign) => assign.value.as_deref().and_then(|value| {
                self_attr_name(&assign.target).map(|attr_name| (attr_name, value))
            }),
            _ => None,
        };
        if let Some((attr_name, value)) = maybe_attr_and_value
            && let Some(class_ref) = class_ref_from_constructor_call(
                value,
                class_map,
                imports,
                module_cache.as_deref_mut(),
                module_path,
            )
        {
            self_attrs.insert(attr_name, class_ref);
        }
    }

    self_attrs
}

fn resolve_class_ref_expr(
    expr: &Expr,
    vars: &HashMap<Identifier, VarState>,
    source: &str,
    func_map: &FuncMap,
    imports: &Imports,
    class_map: &ClassMap,
    mut module_cache: Option<&mut ModuleCache>,
    module_path: Option<&Path>,
) -> Option<ClassRef> {
    match expr {
        Expr::Name(name) => vars.get(&name.id).and_then(|v| v.class_ref.clone()),
        Expr::Attribute(attr) => {
            let base_class_ref = resolve_class_ref_expr(
                attr.value.as_ref(),
                vars,
                source,
                func_map,
                imports,
                class_map,
                module_cache.as_deref_mut(),
                module_path,
            )?;
            with_class_info(
                &base_class_ref,
                source,
                func_map,
                imports,
                class_map,
                &mut module_cache,
                module_path,
                |class_info,
                 _callee_source,
                 _callee_func_map,
                 callee_imports,
                 callee_class_map,
                 callee_path,
                 module_cache| {
                    collect_self_class_refs_from_init(
                        class_info.init.as_ref(),
                        callee_class_map,
                        callee_imports,
                        module_cache.as_deref_mut(),
                        callee_path,
                    )
                    .get(&attr.attr)
                    .cloned()
                },
            )
            .flatten()
        }
        _ => None,
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
        for entry in path_var.split(':') {
            if !entry.contains(".venv") {
                continue;
            }
            let mut path = PathBuf::from(entry);
            while let Some(parent) = path.parent() {
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
                path = parent.to_path_buf();
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
