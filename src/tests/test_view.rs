use crate::tests::extract_test_case;
use crate::{analyze_source, tests::is_hover_expected};
use lsp_types::{Position, Range};

const VIEW_PY_DATA: &str = include_str!("../../test_data/view.py");

#[test]
fn test_proper_view_on_same_variable() {
    let src = extract_test_case(VIEW_PY_DATA, 1);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "x =", "B*X R O"));
}

#[test]
fn test_proper_view_on_different_variable() {
    let src = extract_test_case(VIEW_PY_DATA, 1);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "y =", "B*X R*O"));
}

#[test]
fn test_proper_reshape_on_different_variable() {
    let src = extract_test_case(VIEW_PY_DATA, 2);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z =", "B X*Watch"));
}

#[test]
fn test_exp_then_proper_reshape() {
    let src = extract_test_case(VIEW_PY_DATA, 2);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(
        &src,
        &analysis,
        "exp_then_reshape =",
        "B X*Watch"
    ));
}

#[test]
fn test_squeeze_after_multiply() {
    let src = extract_test_case(VIEW_PY_DATA, 3);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "z =", "B X"));
}

#[test]
fn test_squeeze_multiply_oneliner() {
    let src = extract_test_case(VIEW_PY_DATA, 4);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z =", "B X"));
}

#[test]
fn squeeze_nonexisting_dims_produces_diagnostics() {
    let src = extract_test_case(VIEW_PY_DATA, 5);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 4);
    // make sure that this 3 expected diagnostics are each in a different line
    let (all_ranges_different, _) = analysis.diagnostics.iter().fold(
        (
            true,
            vec![Range::new(
                Position {
                    line: 0,
                    character: 4,
                },
                Position {
                    line: 0,
                    character: 4,
                },
            )],
        ),
        |(acc, mut ranges), x| {
            let acc = acc && ranges.iter().all(|&range| range != x.range);
            ranges.push(x.range);
            (acc, ranges)
        },
    );
    assert!(all_ranges_different);
}

#[test]
fn test_hover_on_squeeze_all() {
    let src = extract_test_case(VIEW_PY_DATA, 6);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "z =", "B R"));
}

#[test]
fn test_unsqueeze_hover_dim0_pos() {
    let src = extract_test_case(VIEW_PY_DATA, 7);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    let (pat, expected) = ("z_pos", "1 B R");
    assert!(is_hover_expected(&src, &analysis, pat, expected));
}

#[test]
fn test_unsqueeze_hover_dim0_arg() {
    let src = extract_test_case(VIEW_PY_DATA, 7);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    let (pat, expected) = ("z_arg", "1 R");
    assert!(is_hover_expected(&src, &analysis, pat, expected));
}

#[test]
fn expand_with_proper_args() {
    let src = extract_test_case(VIEW_PY_DATA, 8);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "y =", "A Ex B C"));
}

#[test]
fn expand_with_improper_args() {
    let src = extract_test_case(VIEW_PY_DATA, 9);
    let analysis = analyze_source(&src);
    // only one since expand inference returns early at the first wrong dimension
    assert_eq!(analysis.diagnostics.len(), 1);
}
