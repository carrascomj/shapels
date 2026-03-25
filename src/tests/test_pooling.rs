use crate::analyze_source;
use crate::tests::{extract_test_case, is_hover_expected};

const PY_DATA: &str = include_str!("../../test_data/pooling.py");

#[test]
fn maxpool1d_with_concrete_dims() {
    let src = extract_test_case(PY_DATA, 1);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "y =", "4 8 10"));
}

#[test]
fn maxpool2d_with_concrete_dims() {
    let src = extract_test_case(PY_DATA, 2);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "y =", "32 16 10 9"));
}

#[test]
fn avgpool3d_with_concrete_dims() {
    let src = extract_test_case(PY_DATA, 3);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "y =", "2 8 9 14 9"));
}

#[test]
fn adaptiveavgpool2d_scalar_output() {
    let src = extract_test_case(PY_DATA, 4);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "y =", "B C 1 1"));
}

#[test]
fn adaptivemaxpool2d_partial_output() {
    let src = extract_test_case(PY_DATA, 5);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "y =", "B C H 7"));
}

#[test]
fn fractionalmaxpool2d_with_output_size() {
    let src = extract_test_case(PY_DATA, 6);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "y =", "2 3 5 7"));
}

#[test]
fn lppool1d_with_concrete_dims() {
    let src = extract_test_case(PY_DATA, 7);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "y =", "4 8 10"));
}

#[test]
fn wrong_input_for_maxpool2d_module() {
    let src = extract_test_case(PY_DATA, 8);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 1);
}

#[test]
fn maxpool2d_mixed_symbolic_and_concrete_dims() {
    let src = extract_test_case(PY_DATA, 9);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(
        &src,
        &analysis,
        "y =",
        "2 8 26/S+1 (30+2*P-D*4)/2+1"
    ));
}

#[test]
fn avgpool2d_mixed_symbolic_and_concrete_dims() {
    let src = extract_test_case(PY_DATA, 10);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(
        &src,
        &analysis,
        "y =",
        "2 8 (H+-3)/S+1 15"
    ));
}

#[test]
fn functional_max_pool1d_with_concrete_dims() {
    let src = extract_test_case(PY_DATA, 11);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "y =", "4 8 10"));
}

#[test]
fn functional_max_pool2d_with_concrete_dims() {
    let src = extract_test_case(PY_DATA, 12);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "y =", "32 16 10 9"));
}

#[test]
fn functional_avg_pool3d_with_concrete_dims() {
    let src = extract_test_case(PY_DATA, 13);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "y =", "2 8 9 14 9"));
}

#[test]
fn functional_adaptive_avg_pool2d_scalar_output() {
    let src = extract_test_case(PY_DATA, 14);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "y =", "B C 1 1"));
}

#[test]
fn functional_adaptive_max_pool2d_partial_output() {
    let src = extract_test_case(PY_DATA, 15);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "y =", "B C H 7"));
}

#[test]
fn functional_fractional_max_pool2d_with_output_size() {
    let src = extract_test_case(PY_DATA, 16);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "y =", "2 3 5 7"));
}

#[test]
fn functional_lp_pool1d_with_concrete_dims() {
    let src = extract_test_case(PY_DATA, 17);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "y =", "4 8 10"));
}

#[test]
fn wrong_input_for_functional_max_pool2d() {
    let src = extract_test_case(PY_DATA, 18);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 1);
}

#[test]
fn functional_max_pool2d_mixed_symbolic_and_concrete_dims() {
    let src = extract_test_case(PY_DATA, 19);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(
        &src,
        &analysis,
        "y =",
        "2 8 26/S+1 (30+2*P-D*4)/2+1"
    ));
}

#[test]
fn functional_avg_pool2d_mixed_symbolic_and_concrete_dims() {
    let src = extract_test_case(PY_DATA, 20);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(
        &src,
        &analysis,
        "y =",
        "2 8 (H+-3)/S+1 15"
    ));
}
