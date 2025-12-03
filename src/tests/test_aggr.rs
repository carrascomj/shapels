//! Tests for aggregation/reduction functions like "mean", "amin", "amax", "sum", "std".
//!
//! For each python snippet, we evaluate each agg fn by replacing "sum"
//! with each of the functions. The results, shape-wise, should be the same for
//! any supported aggregation.

use crate::analyze_source;
use crate::tests::{assert_hover_expected, extract_test_case};

use shapels::AGGR_ALIASES;

const VIEW_PY_DATA: &str = include_str!("../../test_data/aggr.py");

#[test]
fn test_sum_after_unsqueeze_multiply() {
    for agg_fn in AGGR_ALIASES {
        let src = extract_test_case(VIEW_PY_DATA, 1).replace("sum", agg_fn);
        let analysis = analyze_source(&src);
        assert_eq!(analysis.diagnostics.len(), 0);
        let (pat, expected) = ("z =", "B 1");
        assert_hover_expected(&src, &analysis, pat, expected);
    }
}

#[test]
fn test_sum_multiple_dims() {
    for agg_fn in AGGR_ALIASES {
        let src = extract_test_case(VIEW_PY_DATA, 2).replace("sum", agg_fn);
        let analysis = analyze_source(&src);
        assert_eq!(analysis.diagnostics.len(), 0);
        assert_hover_expected(&src, &analysis, "z =", "B");
    }
}

#[test]
fn sum_all() {
    for agg_fn in AGGR_ALIASES {
        let src = extract_test_case(VIEW_PY_DATA, 3).replace("sum", agg_fn);
        let analysis = analyze_source(&src);
        assert_eq!(analysis.diagnostics.len(), 0);
        assert_hover_expected(&src, &analysis, "aggr =", "");
    }
}
