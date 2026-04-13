//! [`ContextRef`] implementation and helpers.

use lsp_types::{Diagnostic, DiagnosticSeverity, Range};
use rustpython_parser::{
    ast::{Expr, Identifier},
    text_size::TextRange,
};
use std::collections::HashMap;
use std::path::Path;

use crate::{
    HoverInfo, ModuleCache, Shape, VarState, expr_text_range, expr_var_key, infer_expr_shape,
    module_resolution::{ClassMap, FuncMap},
    op_groups::Imports,
    state_shape, text_range_to_lsp,
};

/// State passed through and mutated by the inference stack.
pub(crate) struct ContextRef<'ctx> {
    /// Variable states visible at the current inference site.
    pub vars: &'ctx HashMap<Identifier, VarState>,
    /// Functions available for local or cross-module symbolic dispatch.
    func_map: &'ctx FuncMap,
    /// Import metadata used to resolve names, aliases, and dtypes.
    pub imports: &'ctx Imports,
    /// Class metadata available to the current inference context.
    class_map: &'ctx ClassMap,
    /// Active symbolic call stack used to detect recursive inference.
    call_stack: &'ctx mut Vec<Identifier>,
    /// Diagnostic sink for inference errors and informational messages.
    pub diagnostics: &'ctx mut Vec<Diagnostic>,
    /// Hover entries populated while recording inferred shapes.
    pub hover_entries: &'ctx mut Vec<(Range, HoverInfo)>,
    /// Source text for the current module being inferred.
    pub source: &'ctx str,
    /// Shared cache for loading and reusing resolved modules across files.
    module_cache: Option<&'ctx mut ModuleCache>,
    /// Filesystem path of the current module, if known.
    module_path: Option<&'ctx Path>,
}

impl<'ctx> ContextRef<'ctx> {
    pub fn new(
        vars: &'ctx HashMap<Identifier, VarState>,
        func_map: &'ctx FuncMap,
        imports: &'ctx Imports,
        class_map: &'ctx ClassMap,
        call_stack: &'ctx mut Vec<Identifier>,
        diagnostics: &'ctx mut Vec<Diagnostic>,
        hover_entries: &'ctx mut Vec<(Range, HoverInfo)>,
        source: &'ctx str,
        module_cache: Option<&'ctx mut ModuleCache>,
        module_path: Option<&'ctx Path>,
    ) -> ContextRef<'ctx> {
        Self {
            vars,
            func_map,
            imports,
            class_map,
            call_stack,
            diagnostics,
            hover_entries,
            source,
            module_cache,
            module_path,
        }
    }

    pub fn lookup_or_infer(&mut self, expr: &Expr, record_hovers: bool) -> Option<Shape> {
        if let Some(shape) = self.lookup_shape(expr, record_hovers) {
            return Some(shape);
        }

        self.infer_shape(expr, record_hovers)
    }

    pub fn reborrow(&mut self) -> ContextRef<'_> {
        ContextRef {
            vars: self.vars,
            func_map: self.func_map,
            imports: self.imports,
            class_map: self.class_map,
            call_stack: self.call_stack,
            diagnostics: self.diagnostics,
            hover_entries: self.hover_entries,
            source: self.source,
            module_cache: self.module_cache.as_deref_mut(),
            module_path: self.module_path,
        }
    }

    pub fn lookup_shape(&mut self, expr: &Expr, record_hovers: bool) -> Option<Shape> {
        let key = expr_var_key(expr)?;
        let shape = self.vars.get(&key).and_then(state_shape).cloned();
        if record_hovers && let Some(s) = shape.clone() {
            let range = text_range_to_lsp(expr_text_range(expr), self.source);
            self.hover_entries
                .push((range, HoverInfo { shape: Some(s) }));
        }
        shape
    }

    pub fn infer_shape(&mut self, expr: &Expr, record_hovers: bool) -> Option<Shape> {
        infer_expr_shape(
            expr,
            self.vars,
            self.func_map,
            self.imports,
            self.class_map,
            self.call_stack,
            self.diagnostics,
            self.hover_entries,
            record_hovers,
            self.source,
            self.module_cache.as_deref_mut(),
            self.module_path,
        )
    }

    pub fn module_infer_parts(
        &mut self,
    ) -> (
        &HashMap<Identifier, VarState>,
        &FuncMap,
        &Imports,
        &ClassMap,
        &mut Vec<Identifier>,
        &mut Vec<Diagnostic>,
        &mut Vec<(Range, HoverInfo)>,
        &str,
        Option<&mut ModuleCache>,
        Option<&Path>,
    ) {
        (
            self.vars,
            self.func_map,
            self.imports,
            self.class_map,
            self.call_stack,
            self.diagnostics,
            self.hover_entries,
            self.source,
            self.module_cache.as_deref_mut(),
            self.module_path,
        )
    }

    pub fn diagnostics_and_source(&mut self) -> (&mut Vec<Diagnostic>, &str) {
        (self.diagnostics, self.source)
    }

    pub fn vars_diagnostics_source(
        &mut self,
    ) -> (&HashMap<Identifier, VarState>, &mut Vec<Diagnostic>, &str) {
        (self.vars, self.diagnostics, self.source)
    }

    pub fn push_diagnostic(&mut self, range: Range, sev: DiagnosticSeverity, msg: String) {
        self.diagnostics.push(Diagnostic {
            range,
            severity: Some(sev),
            code: None,
            code_description: None,
            source: Some("shapels".into()),
            message: msg,
            related_information: None,
            tags: None,
            data: None,
        });
    }

    pub fn push_diagnostic_text(&mut self, range: TextRange, sev: DiagnosticSeverity, msg: String) {
        self.push_diagnostic(text_range_to_lsp(range, self.source), sev, msg);
    }
}
