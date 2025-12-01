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
fn test_matmul_top_level_caller_to_callee() {
    let src = extract_test_case(PY_DATA, 10);
    let analysis = analyze_source(&src);
    // one diagnostic for x @ y
    assert_eq!(analysis.diagnostics.len(), 1);
    let mut line_idx = 0u32;
    let mut col_idx = 0u32;
    for (idx, line) in src.lines().enumerate() {
        if let Some(pos) = line.find("z =") {
            line_idx = idx as u32;
            col_idx = pos as u32;
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

#[test]
fn test_hover_inferred_shape_from_caller_to_callee_torchmm() {
    let src = extract_test_case(PY_DATA, 6);
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
fn test_hover_inferred_shape_from_caller_to_callee_mm() {
    let src = extract_test_case(PY_DATA, 7);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    let mut line_idx = 0u32;
    let mut col_idx = 0u32;
    for (pat, offset) in [("z =", 0), ("return z", 7)] {
        for (idx, line) in src.lines().enumerate() {
            if let Some(pos) = line.find(pat) {
                line_idx = idx as u32;
                col_idx = pos as u32 + offset;
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
}

#[test]
fn test_hover_hadamard() {
    let mult_source = extract_test_case(PY_DATA, 8);
    for pat in ["*", "+", "-", "/"] {
        let src = mult_source.replace("*", pat);
        let analysis = analyze_source(&src);
        // one diagnostic for non-compatible shapes `output_wrong`
        assert_eq!(analysis.diagnostics.len(), 1);
        let mut line_idx = 0u32;
        let mut col_idx = 0u32;
        let var = "output_right =";
        let expected = "B X R";
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
fn test_hover_hadamard_with_broadcasting_not_broadcastable() {
    // this is the example from the torch docs but with * instead of +
    let mult_source = extract_test_case(PY_DATA, 9);
    for pat in ["*", "+", "-", "/"] {
        let src = mult_source.replace("*", pat);
        let analysis = analyze_source(&src);
        assert_eq!(analysis.diagnostics.len(), 2);
        let mut line_idx = 0u32;
        let mut col_idx = 0u32;
        let var = "z =";
        let expected = "A B C 1";
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
fn test_torch_mm_as_method_works() {
    let src = extract_test_case(PY_DATA, 11);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    let mut line_idx = 0u32;
    let mut col_idx = 0u32;
    let var = "z:";
    let expected = "B X S";
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
        .expect("No hover info on `z:` for x.mm(y)");
    let shape = hover.shape.as_ref().unwrap();
    assert_eq!(shape.render(), expected);
}
