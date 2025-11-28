use crate::analyze_source;
use crate::tests::extract_test_case;
use lsp_types::Position;

const VIEW_PY_DATA: &str = include_str!("../../test_data/view.py");

#[test]
fn test_proper_view_on_same_variable() {
    let src = extract_test_case(VIEW_PY_DATA, 1);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    let mut line_idx = 0u32;
    let mut col_idx = 0u32;
    for (idx, line) in src.lines().enumerate() {
        // this is the first x, inside the parent function (early break)
        if let Some(pos) = line.find("x =") {
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
    assert_eq!(shape.render(), "B*X R O");
}

#[test]
fn test_proper_view_on_different_variable() {
    let src = extract_test_case(VIEW_PY_DATA, 1);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    let mut line_idx = 0u32;
    let mut col_idx = 0u32;
    for (idx, line) in src.lines().enumerate() {
        // this is the first x, inside the parent function (early break)
        if let Some(pos) = line.find("y =") {
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
    assert_eq!(shape.render(), "B*X R*O");
}

#[test]
fn test_proper_reshape_on_different_variable() {
    let src = extract_test_case(VIEW_PY_DATA, 2);
    let analysis = analyze_source(&src);
    // 1 diagnostic for wrong annotated y
    assert_eq!(analysis.diagnostics.len(), 1);
    let mut line_idx = 0u32;
    let mut col_idx = 0u32;
    for (idx, line) in src.lines().enumerate() {
        // this is the first x, inside the parent function (early break)
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
    assert_eq!(shape.render(), "B X*Watch");
}
