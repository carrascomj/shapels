use crate::analyze_source;
use crate::tests::{extract_test_case, is_hover_dtype};

const VIEW_PY_DATA: &str = include_str!("../../test_data/view.py");
const PERMUTE_PY_DATA: &str = include_str!("../../test_data/permute.py");
const SHAPE_ASSIGN_PY_DATA: &str = include_str!("../../test_data/shape_assign.py");

#[test]
fn bool_method_returns_bool_dtype() {
    let src = extract_test_case(VIEW_PY_DATA, 6);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_dtype(&src, &analysis, "z =", "bool"));
}

#[test]
fn to_returns_constant_dtype_arg() {
    let src = extract_test_case(VIEW_PY_DATA, 7);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_dtype(&src, &analysis, "z_pos", "int32"));
}

#[test]
fn to_does_not_modify_dtype_inplace() {
    let src = extract_test_case(VIEW_PY_DATA, 7);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_dtype(&src, &analysis, "z_as_x", "F"));
}

#[test]
fn to_returns_torch_dtype_arg() {
    let src = extract_test_case(PERMUTE_PY_DATA, 3);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_dtype(&src, &analysis, "y =", "bfloat16"));
}

#[test]
fn shape_assign_from_tensor_dtype_attr() {
    let src = extract_test_case(SHAPE_ASSIGN_PY_DATA, 7);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_dtype(&src, &analysis, "eps =", "bfloat16"));
}
