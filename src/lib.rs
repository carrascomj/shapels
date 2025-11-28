use lsp_types::{Diagnostic, DiagnosticSeverity, Position, Range};
use rustpython_parser::Parse;
use rustpython_parser::ast::{self, Arguments, Expr, ExprBinOp, Identifier, Operator, Stmt, Suite};
use rustpython_parser::text_size::{TextRange, TextSize};
use std::collections::HashMap;

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
    range: Range,
}

pub fn analyze_source(source: &str) -> Analysis {
    let mut analysis = Analysis::default();
    match Suite::parse(source, "<memory>") {
        Ok(module) => {
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
    call_stack: &mut Vec<Identifier>,
) -> Analysis {
    let (diagnostics, hover_entries, _) = simulate_function(
        args,
        body,
        source,
        func_map,
        call_stack,
        HashMap::new(),
        true,
    );

    Analysis {
        diagnostics,
        hover_entries,
    }
}

fn simulate_function(
    args: &Arguments,
    body: &[Stmt],
    source: &str,
    func_map: &HashMap<Identifier, (Box<Arguments>, Vec<Stmt>)>,
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
                            call_stack,
                            &mut diagnostics,
                            &mut hover_entries,
                            record_hovers,
                            source,
                        );
                    }
                    if let (Some(ann), Some(inf)) = (ann_shape.clone(), inferred.clone()) {
                        if ann.dims != inf.dims {
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
                    }
                    let chosen_shape = ann_shape.clone().or(inferred.clone());
                    if let Some(shape) = chosen_shape {
                        vars.insert(
                            name.clone(),
                            VarState {
                                annotated: ann_shape.clone(),
                                inferred,
                                range,
                            },
                        );
                        if record_hovers {
                            hover_entries.push((range, HoverInfo { shape: Some(shape) }));
                        }
                    }
                }
            }
            Stmt::Assign(assign) => {
                if assign.targets.len() == 1 {
                    if let Some(name) = name_from_expr(&assign.targets[0]) {
                        let range = text_range_to_lsp(expr_text_range(&assign.targets[0]), source);
                        if let Some(shape) = infer_expr_shape(
                            &assign.value,
                            &vars,
                            func_map,
                            call_stack,
                            &mut diagnostics,
                            &mut hover_entries,
                            record_hovers,
                            source,
                        ) {
                            vars.insert(
                                name.clone(),
                                VarState {
                                    annotated: None,
                                    inferred: Some(shape.clone()),
                                    range,
                                },
                            );
                            if record_hovers {
                                hover_entries.push((range, HoverInfo { shape: Some(shape) }));
                            }
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
                        call_stack,
                        &mut diagnostics,
                        &mut hover_entries,
                        record_hovers,
                        source,
                    )
                    .or(return_shape);
                    if record_hovers {
                        if let Some(shape) = return_shape.clone() {
                            let range = text_range_to_lsp(expr_text_range(val), source);
                            hover_entries.push((range, HoverInfo { shape: Some(shape) }));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    (diagnostics, hover_entries, return_shape)
}

fn infer_expr_shape(
    expr: &Expr,
    vars: &HashMap<Identifier, VarState>,
    func_map: &HashMap<Identifier, (Box<Arguments>, Vec<Stmt>)>,
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
                let left_shape = lookup_shape(left, vars, hover_entries, record_hovers, source);
                let right_shape = lookup_shape(right, vars, hover_entries, record_hovers, source);
                match (left_shape, right_shape) {
                    (Some(l), Some(r)) => match infer_matmul(&l, &r) {
                        Ok(shape) => Some(shape),
                        Err(msg) => {
                            diagnostics.push(Diagnostic {
                                range: text_range_to_lsp(*expr_range, source),
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
            } else {
                None
            }
        }
        Expr::Call(call) => {
            if let Expr::Name(func_name) = call.func.as_ref() {
                if let Some((callee_args, callee_body)) = func_map.get(&func_name.id) {
                    // avoid infinite recursion
                    if call_stack.iter().any(|id| id == &func_name.id) {
                        return None;
                    }
                    // build argument binding map
                    let mut arg_shapes: HashMap<Identifier, VarState> = HashMap::new();
                    for (idx, param) in callee_args.args.iter().enumerate() {
                        if let Some(arg_expr) = call.args.get(idx) {
                            if let Some(shape) = infer_expr_shape(
                                arg_expr,
                                vars,
                                func_map,
                                call_stack,
                                diagnostics,
                                hover_entries,
                                record_hovers,
                                source,
                            ) {
                                arg_shapes.insert(
                                    param.def.arg.clone(),
                                    VarState {
                                        annotated: None,
                                        inferred: Some(shape),
                                        range: text_range_to_lsp(param.def.range, source),
                                    },
                                );
                            }
                        }
                    }
                    call_stack.push(func_name.id.clone());
                    let (mut diag, mut hovers, ret_shape) = simulate_function(
                        callee_args.as_ref(),
                        callee_body,
                        source,
                        func_map,
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
            None
        }
        Expr::Name(expr_name) => {
            let shape = vars
                .get(&expr_name.id)
                .and_then(|v| v.annotated.clone().or_else(|| v.inferred.clone()));
            if record_hovers {
                if let Some(s) = shape.clone() {
                    let range = text_range_to_lsp(expr_text_range(expr), source);
                    hover_entries.push((range, HoverInfo { shape: Some(s) }));
                }
            }
            shape
        }
        _ => None,
    }
}

fn infer_matmul(left: &Shape, right: &Shape) -> Result<Shape, String> {
    if left.dims.is_empty() || right.dims.is_empty() {
        return Err("Matmul requires both operands to have shapes".into());
    }
    let left_inner = left.dims.last().unwrap();
    let right_inner = right.dims.first().unwrap();
    if left_inner != right_inner {
        return Err(format!(
            "Matmul inner dimensions mismatch: {} vs {}",
            left_inner, right_inner
        ));
    }
    // batch dims: keep all left dims except the last, then all right dims except the first
    let mut dims: Vec<String> = left.dims[..left.dims.len() - 1].to_vec();
    dims.extend_from_slice(&right.dims[1..]);
    Ok(Shape {
        dtype: left.dtype.clone().or(right.dtype.clone()),
        dims,
    })
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
            if record_hovers {
                if let Some(s) = shape.clone() {
                    let range = text_range_to_lsp(expr_text_range(expr), source);
                    hover_entries.push((range, HoverInfo { shape: Some(s) }));
                }
            }
            shape
        }
        _ => None,
    }
}

fn parse_shape_annotation(expr: &Expr) -> Option<Shape> {
    if let Expr::Subscript(sub) = expr {
        let dtype = name_like(&sub.value);
        let components: Vec<&Expr> = match &*sub.slice {
            ast::Expr::Tuple(t) => t.elts.iter().collect(),
            other => vec![other],
        };
        if components.len() >= 2 {
            if let Some(raw) = string_literal_value(components[1]) {
                let dims = normalize_shape_tokens(&raw);
                return Some(Shape { dtype, dims });
            }
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
            base.push_str(&attr.attr.to_string());
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
                if let Expr::Constant(c) = val {
                    if let ast::Constant::Str(s) = &c.value {
                        buf.push_str(s);
                    }
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
        ) {
            if ann.dims != inf_state.dims {
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
        }

        let state = VarState {
            annotated: ann_shape.clone(),
            inferred: provided_state.take().and_then(|s| s.inferred).or(None),
            range,
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
    let line_span = (range.end.line as i64 - range.start.line as i64).abs() as u32;
    let char_span = if range.start.line == range.end.line {
        (range.end.character as i64 - range.start.character as i64).abs() as u32
    } else {
        1000
    };
    line_span * 1000 + char_span
}
