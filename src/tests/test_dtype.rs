use crate::analyze_source;
use crate::tests::extract_test_case;
use lsp_types::Position;
use shapels::Analysis;

const VIEW_PY_DATA: &str = include_str!("../../test_data/view.py");
const PERMUTE_PY_DATA: &str = include_str!("../../test_data/permute.py");

fn assert_hover_dtype(src: &str, analysis: &Analysis, pat: &str, expected_dtype: &str) {
    let mut line_idx = 0;
    let mut col_idx = 0;
    for (idx, line) in src.lines().enumerate() {
        if let Some(pos) = line.find(pat) {
            line_idx = idx as u32;
            col_idx = pos as u32;
            break;
        }
    }
    let hover = analysis
        .hover(Position {
            line: line_idx,
            character: col_idx,
        })
        .expect(
            format!("Hover info failed for pat {pat} with expected shape {expected_dtype}")
                .as_str(),
        );
    let dtype = hover
        .shape
        .as_ref()
        .unwrap()
        .dtype
        .as_ref()
        .expect("Hover on {pat} does not contain a dtype.");
    assert_eq!(dtype, expected_dtype);
}

#[test]
fn bool_method_returns_bool_dtype() {
    let src = extract_test_case(VIEW_PY_DATA, 6);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert_hover_dtype(&src, &analysis, "z =", "bool");
}

#[test]
fn to_returns_constant_dtype_arg() {
    let src = extract_test_case(VIEW_PY_DATA, 7);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert_hover_dtype(&src, &analysis, "z_pos", "int32");
}

#[test]
fn to_does_not_modify_dtype_inplace() {
    let src = extract_test_case(VIEW_PY_DATA, 7);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert_hover_dtype(&src, &analysis, "z_as_x", "F");
}

#[test]
fn to_returns_torch_dtype_arg() {
    let src = extract_test_case(PERMUTE_PY_DATA, 3);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert_hover_dtype(&src, &analysis, "y =", "bfloat16");
}
