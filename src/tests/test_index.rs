//! Tests for indexing.

use crate::analyze_source;
use crate::tests::{extract_test_case, is_hover_expected};

const INDEX_PY_DATA: &str = include_str!("../../test_data/index.py");

#[test]
fn none_as_index_is_a_squeeze() {
    let src = extract_test_case(INDEX_PY_DATA, 1);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z =", "1 A B C D"));
}

#[test]
fn ellipsis_type() {
    let src = extract_test_case(INDEX_PY_DATA, 2);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z =", "A B C D"));
}

#[test]
fn integer_indexing() {
    let src = extract_test_case(INDEX_PY_DATA, 3);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z =", "C D"));
}

#[test]
fn boolean_indexing() {
    let src = extract_test_case(INDEX_PY_DATA, 4);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z =", "0 A B C D"));
}

#[test]
fn slice_indexing() {
    let src = extract_test_case(INDEX_PY_DATA, 5);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z =", "A/2 B C D"));
}

#[test]
fn tensor_indexing() {
    let src = extract_test_case(INDEX_PY_DATA, 6);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z =", "C B 2 D"));
}

#[test]
fn multi_indices() {
    let src = extract_test_case(INDEX_PY_DATA, 7);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z =", "2 A C/2"));
}

#[test]
fn infer_half_slice_symbolic() {
    let src = extract_test_case(INDEX_PY_DATA, 8);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z =", "A D C D"));
}

#[test]
fn infer_slice_half_symbolic() {
    let src = extract_test_case(INDEX_PY_DATA, 9);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z1 =", "A B-D+1 C D"));
    assert!(is_hover_expected(&src, &analysis, "z2 =", "A B C-B+3 D"));
}

#[test]
fn infer_half_slice_concrete() {
    let src = extract_test_case(INDEX_PY_DATA, 10);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z =", "128 32 224 16"));
}

#[test]
fn infer_slice_half_concrete() {
    let src = extract_test_case(INDEX_PY_DATA, 11);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z =", "128 32 224 16"));
}

#[test]
fn infer_slice_half_concrete_wrong() {
    let src = extract_test_case(INDEX_PY_DATA, 12);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 1);
}

#[test]
fn infer_slice_half_concrete_neg_wrong() {
    let src = extract_test_case(INDEX_PY_DATA, 13);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 1);
}

#[test]
fn infer_slice_half_concrete_neg_right() {
    let src = extract_test_case(INDEX_PY_DATA, 14);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z =", "128 219 16"));
}

#[test]
fn step_should_be_greater_than_0() {
    let src = extract_test_case(INDEX_PY_DATA, 15);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 1);
}
