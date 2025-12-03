use crate::analyze_source;
use crate::tests::{assert_hover_expected, extract_test_case};

const PERMUTE_PY_DATA: &str = include_str!("../../test_data/permute.py");

#[test]
fn permute_hovers_and_diagnostics_are_captured() {
    let src = extract_test_case(PERMUTE_PY_DATA, 1);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 2);
    for (var, expected) in [("z =", "X B R"), ("y =", "B R X")] {
        assert_hover_expected(&src, &analysis, var, expected);
    }
}

#[test]
fn transpose_hovers_and_diagnostics_are_captured() {
    let src = extract_test_case(PERMUTE_PY_DATA, 2);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 2);
    for (var, expected) in [("z =", "X B R"), ("y =", "B R X")] {
        assert_hover_expected(&src, &analysis, var, expected);
    }
}

#[test]
fn torch_t_hover() {
    let src = extract_test_case(PERMUTE_PY_DATA, 3);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    for (var, expected) in [("y =", "B R X"), ("w =", "B R X"), ("z =", "B R X")] {
        assert_hover_expected(&src, &analysis, var, expected);
    }
}

#[test]
fn torch_t_hover_oneliner() {
    let src = extract_test_case(PERMUTE_PY_DATA, 4);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    let var = "z =";
    let expected = "B";
    assert_hover_expected(&src, &analysis, var, expected);
}
