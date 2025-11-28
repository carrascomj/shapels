use crate::analyze_source;
use crate::tests::extract_test_case;
use lsp_types::Position;

const PY_DATA: &str = include_str!("../../test_data/multiplication.py");

#[test]
fn test_proper_multiply_no_diag() {
    let src = extract_test_case(PY_DATA, 1);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
}

#[test]
fn test_bad_annotation_has_diag() {
    let src = extract_test_case(PY_DATA, 2);
    let analysis = analyze_source(&src);
    assert!(!analysis.diagnostics.is_empty());
}

#[test]
fn test_alias_annotation_does_not_produce_diag() {
    let src = extract_test_case(PY_DATA, 3);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
}

#[test]
fn test_hover_inferred_shape() {
    let src = extract_test_case(PY_DATA, 4);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    eprintln!("hover count {}", analysis.hover_entries.len());
    let mut line_idx = 0u32;
    let mut col_idx = 0u32;
    for (idx, line) in src.lines().enumerate() {
        if let Some(pos) = line.find("z =") {
            line_idx = idx as u32;
            col_idx = pos as u32;
            break;
        }
    }
    let hover = analysis
        .hover(Position {
            line: line_idx,
            character: col_idx,
        })
        .expect("hover info");
    let shape = hover.shape.as_ref().unwrap();
    assert_eq!(shape.render(), "B X S");
}

#[test]
fn test_hover_inferred_shape_from_caller_to_callee() {
    let src = extract_test_case(PY_DATA, 5);
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
    let hover = analysis
        .hover(Position {
            line: line_idx,
            character: col_idx,
        })
        .expect("hover info");
    let shape = hover.shape.as_ref().unwrap();
    assert_eq!(shape.render(), "B X O");
}

#[test]
fn test_hover_infer_shape_on_return() {
    let src = extract_test_case(PY_DATA, 1);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    let mut line_idx = 0u32;
    let mut col_idx = 0u32;
    for (idx, line) in src.lines().enumerate() {
        if let Some(pos) = line.find("return z") {
            line_idx = idx as u32;
            col_idx = pos as u32 + 7;
            break;
        }
    }
    let hover = analysis
        .hover(Position {
            line: line_idx,
            character: col_idx,
        })
        .expect("hover info");
    let shape = hover.shape.as_ref().unwrap();
    assert_eq!(shape.render(), "B X S");
}

#[test]
fn test_hover_infer_shape_same_stack_with_other_variables() {
    let src = extract_test_case(PY_DATA, 5);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    let mut line_idx = 0u32;
    let mut col_idx = 0u32;
    for (idx, line) in src.lines().enumerate() {
        // this is the first x, inside the parent function (early break)
        if let Some(pos) = line.find("x,") {
            line_idx = idx as u32;
            col_idx = pos as u32 - 1;
            break;
        }
    }
    let hover = analysis
        .hover(Position {
            line: line_idx,
            character: col_idx,
        })
        .expect("hover info");
    let shape = hover.shape.as_ref().unwrap();
    assert_eq!(shape.render(), "B X R");
}
