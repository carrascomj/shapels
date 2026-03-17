use crate::analyze_source;
use crate::tests::{extract_test_case, is_hover_dtype, is_hover_expected};

const VIEW_PY_DATA: &str = include_str!("../../test_data/view.py");
const PERMUTE_PY_DATA: &str = include_str!("../../test_data/permute.py");
const SHAPE_ASSIGN_PY_DATA: &str = include_str!("../../test_data/shape_assign.py");

#[test]
fn bool_method_returns_bool_dtype() {
    let src = extract_test_case(VIEW_PY_DATA, 6);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_dtype(&src, &analysis, "z =", "bool"));
}

#[test]
fn to_returns_constant_dtype_arg() {
    let src = extract_test_case(VIEW_PY_DATA, 7);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_dtype(&src, &analysis, "z_pos", "int32"));
}

#[test]
fn to_does_not_modify_dtype_inplace() {
    let src = extract_test_case(VIEW_PY_DATA, 7);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_dtype(&src, &analysis, "z_as_x", "F"));
}

#[test]
fn to_returns_torch_dtype_arg() {
    let src = extract_test_case(PERMUTE_PY_DATA, 3);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_dtype(&src, &analysis, "y =", "bfloat16"));
}

#[test]
fn shape_assign_from_tensor_dtype_attr() {
    let src = extract_test_case(SHAPE_ASSIGN_PY_DATA, 7);
    let analysis = analyze_source(&src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_dtype(&src, &analysis, "eps =", "bfloat16"));
}

#[test]
fn unary_invert_preserves_shape_and_dtype() {
    let src = r#"
import torch

def unary_invert_preserves_shape_and_dtype():
    x = torch.ones(2, 3, dtype=torch.int32)
    y = ~x
"#;
    let analysis = analyze_source(src);
    assert!(analysis.diagnostics.is_empty());
    assert!(is_hover_expected(src, &analysis, "y =", "2 3"));
    assert!(is_hover_dtype(src, &analysis, "y =", "int32"));
}

#[test]
fn unary_invert_rejects_float_dtype() {
    let src = r#"
import torch
from jaxtyping import Float
from torch import Tensor as T

def unary_invert_rejects_float_dtype():
    x: Float[T, "B X"] = torch.ones(2, 3, dtype=torch.float32)
    y = ~x
"#;
    let analysis = analyze_source(src);
    assert_eq!(analysis.diagnostics.len(), 1);
    assert!(
        analysis.diagnostics[0]
            .message
            .contains("Bitwise invert only supports integer or bool dtypes")
    );
    assert!(is_hover_expected(src, &analysis, "y =", "B X"));
}

#[test]
fn unary_plus_and_minus_reject_bool_dtype() {
    let src = r#"
import torch
from jaxtyping import Bool
from torch import Tensor as T

def unary_plus_and_minus_reject_bool_dtype():
    x: Bool[T, "B X"] = torch.ones(2, 3, dtype=torch.bool)
    y = -x
    z = +x
"#;
    let analysis = analyze_source(src);
    assert_eq!(analysis.diagnostics.len(), 2);
    assert!(analysis.diagnostics.iter().all(|diag| {
        diag.message
            .contains("Unary +/- operations do not support bool dtype")
    }));
    assert!(is_hover_expected(src, &analysis, "y =", "B X"));
    assert!(is_hover_expected(src, &analysis, "z =", "B X"));
}
