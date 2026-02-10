use crate::tests::extract_test_case;
use crate::{analyze_source, tests::is_hover_expected};

const BRANCH_DATA: &str = include_str!("../../test_data/branching.py");

#[test]
fn forloop_inference_on_proper_ops() {
    let src = extract_test_case(BRANCH_DATA, 1);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z =", "B X S"));
}

#[test]
fn forloop_inference_on_wrong_ops() {
    let src = extract_test_case(BRANCH_DATA, 3);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 1);
}

#[test]
fn while_inference_on_proper_ops() {
    let src = extract_test_case(BRANCH_DATA, 2);
    let analysis = analyze_source(&src);
    assert!(is_hover_expected(&src, &analysis, "z =", "B X S"));
}

#[test]
fn while_inference_on_wrong_ops() {
    let src = extract_test_case(BRANCH_DATA, 4);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 1);
}

#[test]
fn forloop_inference_does_not_emit_on_continue() {
    let src = extract_test_case(BRANCH_DATA, 5);
    for keyword in ["continue", "break", "return"] {
        let new_src = src.replace("continue", keyword);
        let analysis = analyze_source(&new_src);
        assert!(analysis.diagnostics.is_empty());
        assert!(is_hover_expected(
            &new_src,
            &analysis,
            "inferred =",
            "B X S"
        ));
    }
}

#[test]
fn if_inference_on_proper_ops() {
    let src = extract_test_case(BRANCH_DATA, 6);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z =", "B X S"));
}

#[test]
fn if_inference_on_wrong_ops() {
    let src = extract_test_case(BRANCH_DATA, 7);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 1);
}

#[test]
fn with_inference_on_proper_ops() {
    let src = extract_test_case(BRANCH_DATA, 8);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z =", "B X U"));
}

#[test]
fn with_inference_on_wrong_ops() {
    let src = extract_test_case(BRANCH_DATA, 9);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 1);
}
