use crate::analyze_source;
use crate::tests::{extract_test_case, is_hover_expected};

const CALLABLE_PY_DATA: &str = include_str!("../../test_data/callable.py");

#[test]
fn empty_args_callable_is_inferred() {
    let src = extract_test_case(CALLABLE_PY_DATA, 1);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "y =", "H J K D"));
}

#[test]
fn wrong_dim_arg_should_emit_diagnostics() {
    let src = extract_test_case(CALLABLE_PY_DATA, 2);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 1);
}

#[test]
fn annotated_overwrites_arg_union() {
    let src = extract_test_case(CALLABLE_PY_DATA, 4);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "loss =", "H J K L"));
}

#[test]
fn annotated_var_overwrites() {
    let src = extract_test_case(CALLABLE_PY_DATA, 5);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "loss =", "H J K L"));
}

#[test]
fn multiple_args_callable_is_inferred() {
    let src = extract_test_case(CALLABLE_PY_DATA, 3);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "loss =", "H"));
}

#[test]
fn annotated_ann_asignment_overwrites() {
    let src = extract_test_case(CALLABLE_PY_DATA, 6);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "loss =", "X Y Z"));
}

#[test]
fn tuple_destructuring_from_callable_works() {
    let src = extract_test_case(CALLABLE_PY_DATA, 7);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "a,", "X Y Z"));
    assert!(is_hover_expected(&src, &analysis, "b =", "X Y"));
    assert!(is_hover_expected(&src, &analysis, "out =", "X Z X"));
}
