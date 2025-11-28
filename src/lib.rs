use lsp_types::{Diagnostic, DiagnosticSeverity, Position, Range};
use rustpython_parser::ast::{self, Arguments, Expr, ExprBinOp, Identifier, Operator, Stmt};
use rustpython_parser::parse_program;
use rustpython_parser::text_size::TextRange;
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
    match parse_program(source, "<memory>") {
        Ok(module) => {
            for stmt in module {
                if let Stmt::FunctionDef(func) = stmt {
                    let mut func_analysis = analyze_function(&func.args, &func.body, source);
                    analysis.diagnostics.append(&mut func_analysis.diagnostics);
                    analysis.hover_entries.append(&mut func_analysis.hover_entries);
                }
            }
        }
        Err(err) => {
            analysis.diagnostics.push(Diagnostic {
                range: default_range(),
                severity: Some(DiagnosticSeverity::ERROR),
                code: None,
                code_description: None,
                source: Some("shapelsp".into()),
                message: format!("Parse error: {err}"),
                related_information: None,
                tags: None,
                data: None,
            });
        }
    }
    analysis
}

fn analyze_function(args: &Arguments, body: &[Stmt], source: &str) -> Analysis {
    let mut diagnostics = Vec::new();
    let mut hover_entries = Vec::new();
    let mut vars: HashMap<Identifier, VarState> = HashMap::new();

    for arg in &args.args {
        if let Some(shape) = arg
            .def
            .annotation
            .as_ref()
            .and_then(|expr| parse_shape_annotation(expr.as_ref()))
        {
            let range = text_range_to_lsp(arg.def.range, source);
            vars.insert(
                arg.def.arg.clone(),
                VarState {
                    annotated: Some(shape.clone()),
                    inferred: None,
                    range,
                },
            );
            hover_entries.push((range, HoverInfo { shape: Some(shape) }));
        }
    }

    for stmt in body {
        match stmt {
            Stmt::AnnAssign(assign) => {
                if let Some(name) = name_from_expr(&assign.target) {
                    let ann_shape = parse_shape_annotation(&assign.annotation);
                    let range = text_range_to_lsp(assign.range, source);
                    let mut inferred = None;
                    if let Some(val) = &assign.value {
                        inferred = infer_expr_shape(val, &vars, &mut diagnostics, source);
                    }
                    if let (Some(ann), Some(inf)) = (ann_shape.clone(), inferred.clone()) {
                        if ann.dims != inf.dims {
                            diagnostics.push(Diagnostic {
                                range,
                                severity: Some(DiagnosticSeverity::ERROR),
                                code: None,
                                code_description: None,
                                source: Some("shapelsp".into()),
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
                        hover_entries.push((range, HoverInfo { shape: Some(shape) }));
                    }
                }
            }
            Stmt::Assign(assign) => {
                if assign.targets.len() == 1 {
                    if let Some(name) = name_from_expr(&assign.targets[0]) {
                        let range = text_range_to_lsp(assign.range, source);
                        if let Some(shape) = infer_expr_shape(&assign.value, &vars, &mut diagnostics, source) {
                            vars.insert(
                                name.clone(),
                                VarState {
                                    annotated: None,
                                    inferred: Some(shape.clone()),
                                    range,
                                },
                            );
                            hover_entries.push((range, HoverInfo { shape: Some(shape) }));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    Analysis {
        diagnostics,
        hover_entries,
    }
}

fn infer_expr_shape(
    expr: &Expr,
    vars: &HashMap<Identifier, VarState>,
    diagnostics: &mut Vec<Diagnostic>,
    source: &str,
) -> Option<Shape> {
    match expr {
        Expr::BinOp(ExprBinOp { left, op, right, range: expr_range }) => {
            if matches!(op, Operator::MatMult) {
                let left_shape = lookup_shape(left, vars);
                let right_shape = lookup_shape(right, vars);
                match (left_shape, right_shape) {
                    (Some(l), Some(r)) => match infer_matmul(&l, &r) {
                        Ok(shape) => Some(shape),
                        Err(msg) => {
                            diagnostics.push(Diagnostic {
                                range: text_range_to_lsp(*expr_range, source),
                                severity: Some(DiagnosticSeverity::ERROR),
                                code: None,
                                code_description: None,
                                source: Some("shapelsp".into()),
                                message: msg,
                                related_information: None,
                                tags: None,
                                data: None,
                            });
                            None
                        }
                    },
                    _ => {
                        diagnostics.push(Diagnostic {
                            range: text_range_to_lsp(*expr_range, source),
                            severity: Some(DiagnosticSeverity::ERROR),
                            code: None,
                            code_description: None,
                            source: Some("shapelsp".into()),
                            message: "Cannot infer matmul operands".into(),
                            related_information: None,
                            tags: None,
                            data: None,
                        });
                        None
                    }
                }
            } else {
                None
            }
        }
        Expr::Name(expr_name) => vars
            .get(&expr_name.id)
            .and_then(|v| v.annotated.clone().or_else(|| v.inferred.clone())),
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
    let mut dims = Vec::new();
    dims.push(left.dims.first().unwrap().clone());
    dims.push(right.dims.last().unwrap().clone());
    Ok(Shape {
        dtype: left.dtype.clone().or(right.dtype.clone()),
        dims,
    })
}

fn lookup_shape(expr: &Expr, vars: &HashMap<Identifier, VarState>) -> Option<Shape> {
    match expr {
        Expr::Name(expr_name) => vars
            .get(&expr_name.id)
            .and_then(|v| v.annotated.clone().or_else(|| v.inferred.clone())),
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
        .filter(|token| {
            let t = token.trim_matches('"');
            !matches!(t, "x" | "X" | "*")
        })
        .map(|s| s.trim_matches('"').to_string())
        .collect()
}

fn name_from_expr(expr: &Expr) -> Option<Identifier> {
    match expr {
        Expr::Name(n) => Some(n.id.clone()),
        _ => None,
    }
}

fn text_range_to_lsp(range: TextRange, source: &str) -> Range {
    Range {
        start: offset_to_position(source, range.start().to_usize()),
        end: offset_to_position(source, range.end().to_usize()),
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
        self.hover_entries
            .iter()
            .find(|(range, _)| within(range, &position))
            .map(|(_, info)| info)
    }
}

fn within(range: &Range, pos: &Position) -> bool {
    (pos.line > range.start.line
        || (pos.line == range.start.line && pos.character >= range.start.character))
        && (pos.line < range.end.line || (pos.line == range.end.line && pos.character <= range.end.character))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PY_DATA: &str = include_str!("../test_data/multiplication.py");

    fn extract_test_case(n: usize) -> String {
        let marker = format!("# test {n}");
        let mut buf = Vec::new();
        let mut capture = false;
        for line in PY_DATA.lines() {
            if line.trim() == marker {
                capture = true;
                continue;
            }
            if capture && line.starts_with("# test ") {
                break;
            }
            if capture {
                buf.push(line);
            }
        }
        buf.join("\n")
    }

    #[test]
    fn test_proper_multiply_no_diag() {
        let src = extract_test_case(1);
        let analysis = analyze_source(&src);
        assert_eq!(analysis.diagnostics.len(), 0);
    }

    #[test]
    fn test_bad_annotation_has_diag() {
        let src = extract_test_case(2);
        let analysis = analyze_source(&src);
        assert!(!analysis.diagnostics.is_empty());
    }

    #[test]
    fn test_alias_annotation_produces_diag() {
        let src = extract_test_case(3);
        let analysis = analyze_source(&src);
        assert!(!analysis.diagnostics.is_empty());
    }

    #[test]
    fn test_hover_inferred_shape() {
        let src = extract_test_case(4);
        let analysis = analyze_source(&src);
        assert_eq!(analysis.diagnostics.len(), 0);
        let mut line_idx = 0u32;
        let mut col_idx = 0u32;
        for (idx, line) in src.lines().enumerate() {
            if let Some(pos) = line.find("z =") {
                line_idx = idx as u32;
                col_idx = pos as u32;
                break;
            }
        }
        let hover = analysis.hover(Position { line: line_idx, character: col_idx }).expect("hover info");
        let shape = hover.shape.as_ref().unwrap();
        assert_eq!(shape.render(), "B S");
    }
}
