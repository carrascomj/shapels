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

#[test]
fn cross_entropy_none_returns_target_shape() {
    let src = extract_test_case(LOSS_PY_DATA, 9);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "loss =", "B H W"));
}

#[test]
fn cross_entropy_wrong_target_shape_emits_diagnostic() {
    let src = extract_test_case(LOSS_PY_DATA, 10);
    let analysis = analyze_source(&src);
    assert!(!analysis.diagnostics.is_empty());
}

#[test]
fn ctc_loss_none_returns_batch_shape() {
    let src = extract_test_case(LOSS_PY_DATA, 11);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "loss =", "N"));
}

#[test]
fn ctc_loss_wrong_input_lengths_emits_diagnostic() {
    let src = extract_test_case(LOSS_PY_DATA, 12);
    let analysis = analyze_source(&src);
    assert!(!analysis.diagnostics.is_empty());
}

#[test]
fn cosine_embedding_loss_none_returns_batch_shape() {
    let src = extract_test_case(LOSS_PY_DATA, 13);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "loss =", "N"));
}

#[test]
fn cosine_embedding_loss_wrong_target_emits_diagnostic() {
    let src = extract_test_case(LOSS_PY_DATA, 14);
    let analysis = analyze_source(&src);
    assert!(!analysis.diagnostics.is_empty());
}

#[test]
fn triplet_margin_loss_none_returns_batch_shape() {
    let src = extract_test_case(LOSS_PY_DATA, 15);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "loss =", "N"));
}

#[test]
fn triplet_margin_loss_wrong_positive_emits_diagnostic() {
    let src = extract_test_case(LOSS_PY_DATA, 16);
    let analysis = analyze_source(&src);
    assert!(!analysis.diagnostics.is_empty());
}

#[test]
fn functional_triplet_margin_with_distance_loss_none_returns_batch_shape() {
    let src = extract_test_case(LOSS_PY_DATA, 17);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "loss =", "N"));
}

#[test]
fn functional_triplet_margin_with_distance_loss_wrong_negative_emits_diagnostic() {
    let src = extract_test_case(LOSS_PY_DATA, 18);
    let analysis = analyze_source(&src);
    assert!(!analysis.diagnostics.is_empty());
}
