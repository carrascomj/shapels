use crate::analyze_source;
use crate::tests::extract_test_case;
use lsp_types::Position;
const PERMUTE_PY_DATA: &str = include_str!("../../test_data/permute.py");

#[test]
fn permute_hovers_and_diagnostics_are_captured() {
    let src = extract_test_case(PERMUTE_PY_DATA, 1);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 2);
    let mut line_idx = 0u32;
    let mut col_idx = 0u32;
    for (var, expected) in [("z =", "X B R"), ("y =", "B R X")] {
        for (idx, line) in src.lines().enumerate() {
            if let Some(pos) = line.find(var) {
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
        assert_eq!(shape.render(), expected);
    }
}

#[test]
fn transpose_hovers_and_diagnostics_are_captured() {
    let src = extract_test_case(PERMUTE_PY_DATA, 2);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 2);
    let mut line_idx = 0u32;
    let mut col_idx = 0u32;
    for (var, expected) in [("z =", "X B R"), ("y =", "B R X")] {
        for (idx, line) in src.lines().enumerate() {
            if let Some(pos) = line.find(var) {
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
        assert_eq!(shape.render(), expected);
    }
}

#[test]
fn torch_t_hover() {
    let src = extract_test_case(PERMUTE_PY_DATA, 3);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    let mut line_idx = 0u32;
    let mut col_idx = 0u32;
    for (var, expected) in [("y =", "B R X"), ("w =", "B R X"), ("z =", "B R X")] {
        for (idx, line) in src.lines().enumerate() {
            if let Some(pos) = line.find(var) {
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
        let shape = hover.shape.as_ref().expect("Shape can be rendered");
        let shape_render = shape.render();
        println!(" {shape_render} | {expected}");
        assert_eq!(shape.render(), expected);
    }
}

#[test]
fn torch_t_hover_oneliner() {
    let src = extract_test_case(PERMUTE_PY_DATA, 4);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    let mut line_idx = 0u32;
    let mut col_idx = 0u32;
    let var = "z =";
    let expected = "B";
    for (idx, line) in src.lines().enumerate() {
        if let Some(pos) = line.find(var) {
            line_idx = idx as u32;
            col_idx = pos as u32;
            break;
        }
    }
    println!("{var}");
    let hover = analysis
        .hover(Position {
            line: line_idx,
            character: col_idx,
        })
        .expect("hover info");
    println!("{:#?}", hover.shape);
    let shape = hover.shape.as_ref().expect("Shape can be rendered");
    let shape_render = shape.render();
    println!(" {shape_render} | {expected}");
    assert_eq!(shape.render(), expected);
}
