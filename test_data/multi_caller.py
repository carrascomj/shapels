# test 1
import torch
from jaxtyping import Float as F
from torch import Tensor
from multi_callee import multiply_child_unnanotated_torch_mm_aliased as mc


def hovering_with_inference_on_arg_with_mm():
    B, X, R, O = 4, 2, 16, 8
    x: F[Tensor, "B X R"] = torch.Tensor(B, X, R)
    y: F[Tensor, "R O"] = torch.Tensor(R, O)
    # hovering z should return [F, "B X O"]
    z = mc(x, y)
    return z


# test 2
from multi_callee import from_zeros_fully_annotated

def arg_hint_returns_diagnostic_on_mismatch():
    J, K, L = 3, 27, 81
    arg = torch.zeros(J, K, L)
    # arg should be [B T O]
    arg_shape_is_wrong = from_zeros_fully_annotated(arg)
    # this is added here so that there are enough lines
    # when the test extracted to show the diagnostic if wrongly
    # placed as in the callee
    return (
        arg,
        J + K + L,
        arg_shape_is_wrong
    )


# test 3
from multi_callee import from_zeros_no_annotation

def shape_inference_is_propagated_through_non_annotated_callee():
    x = [1, 2, 3, 4]
    # z is [B A X]: float64 because of callee code
    z = from_zeros_no_annotation(x)


# test 4
from multi_callee import from_zeros_with_float_return_dtype

def float_return_type_hint_is_returned_as_hover():
    z = from_zeros_with_float_return_dtype(None)


# test 5
from multi_callee import from_zeros_with_int_return_dtype

def int_return_type_hint_is_returned_as_hover():
    z = from_zeros_with_int_return_dtype([1, 2])


# test 6
from multi_callee import from_zeros_to_tuple

def simple_tuple_destructuring():
    z, a_list = from_zeros_to_tuple([1, 2])


# test 7
from multi_callee import from_zeros_to_tuple

def int_return_tint_return_type_hint_is_returned_as_hoverype_hint_is_returned_as_hover():
    z, (val1, val2) = from_zeros_to_tuple((1, 2))


# test 8
from multi_callee import from_zeros_to_tuple
from multi_callee import UserLinear

def int_return_tint_return_type_hint_is_returned_as_hoverype_hint_is_returned_as_hover():
    z, (val1, val2) = from_zeros_to_tuple((1, 2))
    linear = UserLinear()
    b = linear(z, z[0].T)
