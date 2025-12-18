use crate::analyze_source;
use crate::tests::{extract_test_case, is_hover_expected};

const PY_DATA: &str = include_str!("../../test_data/class.py");

#[test]
fn proper_multiply_no_diag_inside_class_method() {
    let src = extract_test_case(PY_DATA, 1);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z =", "B X S"));
}

#[test]
fn module_is_resolved_to_its_forward_function() {
    let src = extract_test_case(PY_DATA, 2);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 1);
    assert!(is_hover_expected(&src, &analysis, "output =", "B X S"));
}

#[test]
fn module_as_fn_arg_is_resolved_to_its_forward_function() {
    let src = extract_test_case(PY_DATA, 3);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 1);
    assert!(is_hover_expected(&src, &analysis, "output =", "B X O"));
}

#[test]
fn module_as_fn_union_arg_is_resolved_to_its_forward_function() {
    let src = extract_test_case(PY_DATA, 4);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 1);
    assert!(is_hover_expected(&src, &analysis, "output =", "B X O"));
}

#[test]
fn module_as_fn_arg_with_tensor_as_union_arg_is_resolved_to_its_forward_function() {
    let src = extract_test_case(PY_DATA, 5);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 1);
    assert!(is_hover_expected(&src, &analysis, "output =", "B X O"));
    assert!(is_hover_expected(&src, &analysis, "alias_x =", "B X R"));
}
