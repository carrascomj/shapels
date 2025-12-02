use shapels::analyze_source;

use crate::tests::{assert_hover_expected, extract_test_case};

///! Tests for assignments of shapes by hints or special methods.

const PY_DATA: &str = include_str!("../../test_data/shape_assign.py");

#[test]
fn test_inferred_from_shape_unrolling() {
    let src = extract_test_case(PY_DATA, 1);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert_hover_expected(&src, &analysis, "output =", "Batch Width Height NumClasses");
}

#[test]
fn test_wrong_shape_assign_has_diag() {
    let src = extract_test_case(PY_DATA, 2);
    let analysis = analyze_source(&src);
    assert!(!analysis.diagnostics.is_empty());
}

#[test]
fn test_shape_unrolling_renames_shapes_on_samedims() {
    let src = extract_test_case(PY_DATA, 3);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert_hover_expected(
        &src,
        &analysis,
        "after_renamed",
        "Batch Channels Height Width",
    );
}

#[test]
fn test_ann_assign_renames_shapes_on_samedims() {
    let src = extract_test_case(PY_DATA, 4);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert_hover_expected(
        &src,
        &analysis,
        "after_renamed",
        "Batch Channels Height Width",
    );
}

#[test]
fn test_ann_assign_missalignment_emits_diagnostics() {
    let src = extract_test_case(PY_DATA, 5);
    let analysis = analyze_source(&src);
    assert!(!analysis.diagnostics.is_empty());
}
