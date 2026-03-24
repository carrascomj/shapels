//! Tests for native torch modules from `torch.nn`.
use crate::tests::extract_test_case;
use crate::{analyze_source, tests::is_hover_expected};

const LOSS_PY_DATA: &str = include_str!("../../test_data/loss.py");

#[test]
fn loss_none_is_a_noop() {
    let src = extract_test_case(LOSS_PY_DATA, 1);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "loss =", "B X Y"));
}

#[test]
fn loss_mean_reduces_to_singleton() {
    let src = extract_test_case(LOSS_PY_DATA, 2);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "loss =", ""));
}

#[test]
fn loss_default() {
    let src = extract_test_case(LOSS_PY_DATA, 3);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "loss =", ""));
}

#[test]
fn loss_none_by_position() {
    let src = extract_test_case(LOSS_PY_DATA, 4);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "loss =", "B X Y"));
}

#[test]
fn functional_loss_none_is_a_noop() {
    let src = extract_test_case(LOSS_PY_DATA, 5);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "loss =", "B X Y"));
}

#[test]
fn functional_loss_mean_reduces_to_singleton() {
    let src = extract_test_case(LOSS_PY_DATA, 6);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "loss =", ""));
}

#[test]
fn functional_loss_default() {
    let src = extract_test_case(LOSS_PY_DATA, 7);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "loss =", ""));
}

#[test]
fn functional_loss_none_by_position() {
    let src = extract_test_case(LOSS_PY_DATA, 8);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "loss =", "B X Y"));
}
