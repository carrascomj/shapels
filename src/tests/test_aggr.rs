//! Tests for aggregation functions like mean, min, max, sum, std

use crate::analyze_source;
use crate::tests::extract_test_case;
use lsp_types::Position;

const VIEW_PY_DATA: &str = include_str!("../../test_data/aggr.py");

#[test]
fn test_sum_after_unsqueeze_multiply() {
    let src = extract_test_case(VIEW_PY_DATA, 1);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
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
    assert_eq!(shape.render(), "B 1");
}

#[test]
fn test_sum_multiple_dims() {
    let src = extract_test_case(VIEW_PY_DATA, 2);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
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
    assert_eq!(shape.render(), "B");
}

#[test]
fn sum_all() {
    let src = extract_test_case(VIEW_PY_DATA, 3);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    let mut line_idx = 0u32;
    let mut col_idx = 0u32;
    for (idx, line) in src.lines().enumerate() {
        // this is the first x, inside the parent function (early break)
        if let Some(pos) = line.find("aggr =") {
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
    assert_eq!(shape.render(), "");
}
