//! Tests for assignments of shapes by hints or special methods.
use lsp_types::Position;
use shapels::analyze_source;

use crate::tests::{extract_test_case, is_hover_expected};

const PY_DATA: &str = include_str!("../../test_data/shape_assign.py");

#[test]
fn test_inferred_from_shape_unrolling() {
    let src = extract_test_case(PY_DATA, 1);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(
        &src,
        &analysis,
        "output =",
        "Batch Width Height NumClasses"
    ));
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
    assert!(is_hover_expected(
        &src,
        &analysis,
        "after_renamed",
        "Batch Channels Height Width",
    ));
}

#[test]
fn test_ann_assign_renames_shapes_on_samedims() {
    let src = extract_test_case(PY_DATA, 4);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(
        &src,
        &analysis,
        "after_renamed",
        "Batch Channels Height Width",
    ));
}

#[test]
fn test_ann_assign_missalignment_emits_diagnostics() {
    let src = extract_test_case(PY_DATA, 5);
    let analysis = analyze_source(&src);
    assert!(!analysis.diagnostics.is_empty());
}

#[test]
fn test_zeros_creation_size() {
    let src = extract_test_case(PY_DATA, 6);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    let (pat, expected) = ("x =", "Batch Channels Height Width");
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
        .unwrap_or_else(|| {
            panic!("Hover info failed for pat {pat} with expected shape {expected}")
        });
    let shape = hover.shape.as_ref().unwrap();
    assert_eq!(shape.dim_string(), expected);
    assert_eq!(shape.dtype, Some(String::from("bool")));
}

#[test]
fn test_infer_from_shape_attr() {
    let src = extract_test_case(PY_DATA, 7);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(
        &src,
        &analysis,
        "eps =",
        "Batch Features"
    ));
}

#[test]
fn randperm_from_namevar() {
    let src = extract_test_case(PY_DATA, 8);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(
        &src,
        &analysis,
        "out =",
        "n_instances Features"
    ));
}

#[test]
fn range_from_all_numbers() {
    let src = extract_test_case(PY_DATA, 9);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "some_range =", "101"));
    assert!(is_hover_expected(&src, &analysis, "some_arange =", "100"));
}

#[test]
fn range_from_denom_numbers() {
    let src = extract_test_case(PY_DATA, 9);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(
        &src,
        &analysis,
        "some_range_n =",
        "10/n_instances+1"
    ));
    assert!(is_hover_expected(
        &src,
        &analysis,
        "some_arange_n =",
        "10/n_instances"
    ));
}

#[test]
fn linspace_from_name_expr() {
    let src = extract_test_case(PY_DATA, 9);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "lin_n =", "n_instances"));
}

#[test]
fn init_full_from_tuple() {
    let src = extract_test_case(PY_DATA, 10);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "bag_max =", "num_bags"));
    assert!(is_hover_expected(&src, &analysis, "bag_list =", "num_bags"));
}

#[test]
fn creation_size_op_from_index_on_shape() {
    let src = extract_test_case(PY_DATA, 11);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(
        &src,
        &analysis,
        "bag_max_shape =",
        "num_bags Feat"
    ));
    assert!(is_hover_expected(
        &src,
        &analysis,
        "bag_max_size =",
        "num_bags Batch"
    ));
}

#[test]
fn arange_single_arg_symbolic() {
    let src = extract_test_case(PY_DATA, 12);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
}

#[test]
fn range_single_arg_concrete() {
    let src = extract_test_case(PY_DATA, 13);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "bag_indices =", "32"));
}

#[test]
fn only_one_arg_is_wrong_for_range() {
    let src = extract_test_case(PY_DATA, 14);
    let analysis = analyze_source(&src);
    assert_eq!(analysis.diagnostics.len(), 1);
}

#[test]
fn range_arange_concrete_shapes() {
    let src = extract_test_case(PY_DATA, 15);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(&src, &analysis, "arange_single =", "10"));
    assert!(is_hover_expected(
        &src,
        &analysis,
        "arange_step_one =",
        "10"
    ));
    assert!(is_hover_expected(
        &src,
        &analysis,
        "arange_step_point =",
        "34"
    ));
    assert!(is_hover_expected(&src, &analysis, "range_two_args =", "11"));
    assert!(is_hover_expected(
        &src,
        &analysis,
        "range_point_one =",
        "101"
    ));
    assert!(is_hover_expected(
        &src,
        &analysis,
        "range_step_one_val =",
        "11"
    ));
}
