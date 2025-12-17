from torch import mm as whatever
from jaxtyping import Float as F
from torch import Tensor as T


def multiply_child_unnanotated_torch_mm_aliased(x, y):
    mat_mul_result = whatever(x, y)
    return mat_mul_result


def from_zeros_fully_annotated(x: F[T, "B T O"]) -> F[T, "B A X"]:
    """Calling this function with clashing arg should raise a diagnostic
    on the caller."""
    B, A, X = 2, 8, 16
    return torch.zeros(B, A, X, dtype=torch.float64)


def from_zeros_no_annotation(x):
    """Calling this function should return an annotated shape on the caller."""
    B, A, X = 2, 8, 16
    return torch.zeros(B, A, X, dtype=torch.float64)


def from_zeros_with_float_return_dtype(x) -> F[T, "B A X"]:
    """Calling this function should return an annotated shape on the caller
    with known type F because of the return type hint."""
    B, A, X = 2, 8, 16
    return torch.zeros(B, A, X)


def from_zeros_with_int_return_dtype(x) -> Int[T, "X Y Z"]:
    """Calling this function should return an annotated shape on the caller
    with known type Int because of the return type hint."""
    X, Y, Z = 3, 27, 81
    return torch.zeros(X, Y, Z)


def from_zeros_to_tuple(x):
    """Calling this function should return an annotated shape on the caller."""
    B, A, X = 2, 8, 16
    out = torch.zeros(B, A, X, dtype=torch.float64)
    return out, x
