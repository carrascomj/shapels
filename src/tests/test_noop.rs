//! Tests for operations like softmax that, shape-wise, are NoOps, but
//! require bound checks for the dimension if present.

use crate::analyze_source;
use crate::tests::extract_test_case;
use lsp_types::Position;

const NOOP_PY_DATA: &str = include_str!("../../test_data/noop.py");

#[test]
fn softmax_does_not_produce_diagnostics_for_valid_tensor_method() {
    let src = extract_test_case(NOOP_PY_DATA, 1);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 0);
    for (idx, line) in src.lines().enumerate() {
        if let Some(pos) = line.find("=") {
            let line_idx = idx as u32;
            let col_idx = pos as u32 - 2;
            let hover = analysis
                .hover(Position {
                    line: line_idx,
                    character: col_idx,
                })
                .expect("hover info");
            assert!(hover.shape.is_some());
        }
    }
}

#[test]
fn softmax_produces_diagnostics_for_invalid_cases() {
    let src = extract_test_case(NOOP_PY_DATA, 2);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 5);
}
