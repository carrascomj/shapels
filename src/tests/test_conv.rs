use crate::analyze_source;
use crate::tests::{extract_test_case, is_hover_expected};

const PY_DATA: &str = include_str!("../../test_data/conv.py");

#[test]
fn conv1d_with_concrete_dims() {
    let src = extract_test_case(PY_DATA, 1);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "out =", "32 128 510"));
}

#[test]
fn conv2d_with_mixed_dims() {
    let src = extract_test_case(PY_DATA, 2);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(
        &src,
        &analysis,
        "y =",
        "32 128 (H+5)/2+1 (W+5)/2+1"
    ));
}

#[test]
fn conv3d_with_concrete_dims() {
    let src = extract_test_case(PY_DATA, 3);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "y =", "2 8 10 9 10"));
}

#[test]
fn conv3d_with_mixed_dims() {
    let src = extract_test_case(PY_DATA, 4);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(
        &src,
        &analysis,
        "y =",
        "2 8 19/S+1 (29+2*P-D*4)/3+1 10"
    ));
}

#[test]
fn wrong_input_for_conv1d() {
    let src = extract_test_case(PY_DATA, 5);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 1);
}

#[test]
fn conv1d_module_with_concrete_dims() {
    let src = extract_test_case(PY_DATA, 6);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "out =", "32 128 510"));
}

#[test]
fn conv2d_module_with_mixed_dims() {
    let src = extract_test_case(PY_DATA, 7);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(
        &src,
        &analysis,
        "y =",
        "32 128 (H+5)/2+1 (W+5)/2+1"
    ));
}

#[test]
fn conv3d_module_with_concrete_dims() {
    let src = extract_test_case(PY_DATA, 8);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "y =", "2 8 10 9 128"));
}

#[test]
fn conv3d_module_with_mixed_dims() {
    let src = extract_test_case(PY_DATA, 9);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(
        &src,
        &analysis,
        "y =",
        "2 8 19/S+1 (29+2*P-D*4)/3+1 10"
    ));
}

#[test]
fn wrong_input_for_conv1d_module() {
    let src = extract_test_case(PY_DATA, 10);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 1);
}
