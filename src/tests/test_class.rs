use shapels::analyze_source_at_path;
use std::path::Path;

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
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "output =", "B X O"));
}

#[test]
fn module_as_fn_union_arg_is_resolved_to_its_forward_function() {
    let src = extract_test_case(PY_DATA, 4);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
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

#[test]
fn module_improper_is_resolved_to_its_forward_type_hints() {
    let src = extract_test_case(PY_DATA, 6);
    let analysis = analyze_source(&src);
    // 1 diagnostic inside the callee class, but not at the caller!
    assert_eq!(analysis.diagnostics.len(), 1);
    assert!(is_hover_expected(&src, &analysis, "output =", "B X O"));
}

#[test]
fn module_improper_is_resolved_to_its_forward_tuple_type_hints() {
    let src = extract_test_case(PY_DATA, 7);
    let analysis = analyze_source(&src);
    // 1 diagnostic inside the callee class, but not at the caller!
    assert_eq!(analysis.diagnostics.len(), 1);
    assert!(is_hover_expected(&src, &analysis, "output,", "B X O"));
    assert!(is_hover_expected(&src, &analysis, "output2 =", "B O O"));
}

#[test]
fn inference_propages_through_method() {
    let src = extract_test_case(PY_DATA, 8);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 1);
    assert!(is_hover_expected(&src, &analysis, "out", "B X R"));
}

#[test]
fn method_improper_is_resolved_to_its_forward_tuple_type_hints() {
    let src = extract_test_case(PY_DATA, 9);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/class.py"));
    assert!(is_hover_expected(&src, &analysis, "output,", "B X R"));
    assert!(is_hover_expected(&src, &analysis, "output2 =", "B O T"));
}

#[test]
fn inference_propagates_to_self() {
    let src = extract_test_case(PY_DATA, 10);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/class.py"));
    assert!(is_hover_expected(&src, &analysis, "z =", "B X Z"));
}

#[test]
fn inference_propagates_to_nested_self() {
    let src = extract_test_case(PY_DATA, 11);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/class.py"));
    assert!(is_hover_expected(&src, &analysis, "z =", "B X L"));
}

#[test]
fn inference_propagates_to_annotated_self() {
    let src = extract_test_case(PY_DATA, 12);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/class.py"));
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z =", "B A A"));
}

#[test]
fn inference_emits_diagnostics_on_unknown_self() {
    let src = extract_test_case(PY_DATA, 13);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.len() > 0);
}

#[test]
fn concrete_parameter_is_properly_multiplied() {
    let src = extract_test_case(PY_DATA, 14);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/class.py"));
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "out =", "B X 57"));
}

#[test]
fn abstract_parameter_is_properly_multiplied() {
    let src = extract_test_case(PY_DATA, 15);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/class.py"));
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "out =", "B X 1"));
}

#[test]
fn abstract_parameter_emits_diag_with_wrong_shape() {
    let src = extract_test_case(PY_DATA, 16);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/class.py"));
    assert!(analysis.diagnostics.len() > 0);
}

#[test]
fn self_tensor_is_properly_multiplied() {
    let src = extract_test_case(PY_DATA, 17);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/class.py"));
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "out =", "B X U"));
}

#[test]
fn self_tensor_is_properly_registered_after_ops() {
    let src = extract_test_case(PY_DATA, 18);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/class.py"));
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "out =", "B X 1"));
}

#[test]
fn self_param_is_properly_registered_after_ops() {
    let src = extract_test_case(PY_DATA, 19);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/class.py"));
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "self.my_param =", "S W"));
    assert!(is_hover_expected(&src, &analysis, "out =", "B X W"));
}

#[test]
fn self_param_is_properly_reassigned() {
    let src = extract_test_case(PY_DATA, 20);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/class.py"));
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "my_tensor =", "S T"));
    assert!(is_hover_expected(&src, &analysis, "out =", "B X T"));
}

#[test]
fn annotated_ellipsis_preserves_caller_prefix() {
    let src = extract_test_case(PY_DATA, 21);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/class.py"));
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "y =", "Batch L OutDim"));
}

#[test]
fn annotated_ellipsis_instantiates_tuple_destructuring() {
    let src = extract_test_case(PY_DATA, 22);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/class.py"));
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(
        &src,
        &analysis,
        "y, residual =",
        "Batch Heads Tokens OutDim"
    ));
    assert!(is_hover_expected(
        &src,
        &analysis,
        "residual =",
        "Batch Heads Tokens Embed"
    ));
}
