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
