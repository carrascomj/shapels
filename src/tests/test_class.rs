use crate::analyze_source;
use crate::tests::{extract_test_case, is_hover_expected};

const PY_DATA: &str = include_str!("../../test_data/class.py");

#[test]
fn test_proper_multiply_no_diag() {
    let src = extract_test_case(PY_DATA, 1);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z =", "B X S"));
}

#[test]
fn test_module_is_resolved_to_its_forward_function() {
    let src = extract_test_case(PY_DATA, 2);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 1);
    assert!(is_hover_expected(&src, &analysis, "output =", "B X S"));
}
