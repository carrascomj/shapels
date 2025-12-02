//! Tests for aggregation/reduction functions like "mean", "amin", "amax", "sum", "std".
//!
//! For each python snippet, we evaluate each agg fn by replacing "sum"
//! with each of the functions. The results, shape-wise, should be the same for
//! any supported aggregation.

use crate::analyze_source;
use crate::tests::extract_test_case;
use lsp_types::Position;

use shapels::AGGR_ALIASES;

const VIEW_PY_DATA: &str = include_str!("../../test_data/aggr.py");

#[test]
fn test_sum_after_unsqueeze_multiply() {
    for agg_fn in AGGR_ALIASES {
        let src = extract_test_case(VIEW_PY_DATA, 1).replace("sum", agg_fn);
        let analysis = analyze_source(&src);
        assert_eq!(analysis.diagnostics.len(), 0);
        let mut line_idx = 0u32;
        let mut col_idx = 0u32;
        for (idx, line) in src.lines().enumerate() {
            // this is the first x, inside the parent function (early break)
            if let Some(pos) = line.find("z =") {
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
            .expect("hover info");
        let shape = hover.shape.as_ref().unwrap();
        assert_eq!(shape.dim_string(), "B 1");
    }
}

#[test]
fn test_sum_multiple_dims() {
    for agg_fn in AGGR_ALIASES {
        let src = extract_test_case(VIEW_PY_DATA, 2).replace("sum", agg_fn);
        let analysis = analyze_source(&src);
        assert_eq!(analysis.diagnostics.len(), 0);
        let mut line_idx = 0u32;
        let mut col_idx = 0u32;
        for (idx, line) in src.lines().enumerate() {
            // this is the first x, inside the parent function (early break)
            if let Some(pos) = line.find("z =") {
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
            .expect("hover info");
        let shape = hover.shape.as_ref().unwrap();
        assert_eq!(shape.dim_string(), "B");
    }
}

#[test]
fn sum_all() {
    for agg_fn in AGGR_ALIASES {
        let src = extract_test_case(VIEW_PY_DATA, 3).replace("sum", agg_fn);
        let analysis = analyze_source(&src);
        assert_eq!(analysis.diagnostics.len(), 0);
        let mut line_idx = 0u32;
        let mut col_idx = 0u32;
        for (idx, line) in src.lines().enumerate() {
            // this is the first x, inside the parent function (early break)
            if let Some(pos) = line.find("aggr =") {
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
            .expect("hover info");
        let shape = hover.shape.as_ref().unwrap();
        assert_eq!(shape.dim_string(), "");
    }
}
