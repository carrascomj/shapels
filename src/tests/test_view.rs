use crate::analyze_source;
use crate::tests::extract_test_case;
use lsp_types::{Position, Range};

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
    assert_eq!(shape.dim_string(), "B*X R O");
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
    assert_eq!(shape.dim_string(), "B*X R*O");
}

#[test]
fn test_proper_reshape_on_different_variable() {
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
    assert_eq!(shape.dim_string(), "B X*Watch");
}

#[test]
fn test_exp_then_proper_reshape() {
    let src = extract_test_case(VIEW_PY_DATA, 2);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    let mut line_idx = 0u32;
    let mut col_idx = 0u32;
    for (idx, line) in src.lines().enumerate() {
        // this is the first x, inside the parent function (early break)
        if let Some(pos) = line.find("exp_then_reshape =") {
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
    assert_eq!(shape.dim_string(), "B X*Watch");
}

#[test]
fn test_squeeze_after_multiply() {
    let src = extract_test_case(VIEW_PY_DATA, 3);
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
    assert_eq!(shape.dim_string(), "B X");
}

#[test]
fn test_squeeze_multiply_oneliner() {
    let src = extract_test_case(VIEW_PY_DATA, 4);
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
    assert_eq!(shape.dim_string(), "B X");
}

#[test]
fn squeeze_nonexisting_dims_produces_diagnostics() {
    let src = extract_test_case(VIEW_PY_DATA, 5);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 4);
    // make sure that this 3 expected diagnostics are each in a different line
    let (all_ranges_different, _) = analysis.diagnostics.iter().fold(
        (
            true,
            vec![Range::new(
                Position {
                    line: 0,
                    character: 4,
                },
                Position {
                    line: 0,
                    character: 4,
                },
            )],
        ),
        |(acc, mut ranges), x| {
            let acc = acc && ranges.iter().all(|&range| range != x.range);
            ranges.push(x.range);
            (acc, ranges)
        },
    );
    assert!(all_ranges_different);
}

#[test]
fn test_hover_on_squeeze_all() {
    let src = extract_test_case(VIEW_PY_DATA, 6);
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
    assert_eq!(shape.dim_string(), "B R");
}

#[test]
fn test_unsqueeze_hover_dim0_pos() {
    let src = extract_test_case(VIEW_PY_DATA, 7);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    let mut line_idx = 0u32;
    let mut col_idx = 0u32;
    let (pat, expected) = ("z_pos", "1 B R");
    for (idx, line) in src.lines().enumerate() {
        // this is the first x, inside the parent function (early break)
        if let Some(pos) = line.find(pat) {
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
        .expect(format!("No hover found for {pat}").as_str());
    let shape = hover.shape.as_ref().unwrap();
    assert_eq!(shape.dim_string(), expected);
}

#[test]
fn test_unsqueeze_hover_dim0_arg() {
    let src = extract_test_case(VIEW_PY_DATA, 7);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    let mut line_idx = 0u32;
    let mut col_idx = 0u32;
    let (pat, expected) = ("z_arg", "1 R");
    for (idx, line) in src.lines().enumerate() {
        // this is the first x, inside the parent function (early break)
        if let Some(pos) = line.find(pat) {
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
        .expect(format!("No hover found for {pat}").as_str());
    let shape = hover.shape.as_ref().unwrap();
    assert_eq!(shape.dim_string(), expected);
}
