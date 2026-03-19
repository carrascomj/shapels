"""Tests for operations like softmax that, shape-wise, are NoOps, but
require bound checks for the dimension if present."""


# test 1
from jaxtyping import Floas as F
from torch import tensor as T
from torch import softmax

def softmax_valid_cases_tensor_method(x: F[T, "B X Y"], empty: F[T, ""], one: F[T, " Features"]):
    a = x.softmax(dim=-1)
    a = x.softmax(-2)
    a = x.softmax(0)
    a = x.softmax(dim=1)
    a = x.softmax(dim=2)
    b = empty.softmax(0)
    b = empty.softmax(dim=-1)
    c = one.softmax(dim=0)
    c = torch.softmax(one, -1)
    d = softmax(x, 2)


# test 2
from jaxtyping import Floas as F
from torch import tensor as T

def softmax_invalid(x: F[T, "B X Y"], empty: F[T, ""], one: F[T, " Features"]):
    a = x.softmax(3)
    a = x.softmax(dim=-4)
    b = empty.softmax(dim=1)
    c = one.softmax(dim=1)
    c = one.softmax()


# test 3
from jaxtyping import Floas as F, Bool
from torch import tensor as T

def where_valid_cases(x: Bool[T, "B X Y"], input: F[T, "B X Y"], other: F[T, "B X Y"]):
    out = torch.where(x, input, other)
    input_value = 32
    out_2 = torch.where(x, input_value, other)
    out_3 = torch.where(x, input_value, 38)
    out_4 = torch.where(x, input, 38)
    out_5 = torch.where(input > 32, input, 38)
    out_6 = (input > 32).where(input, 38)


# test 4
from jaxtyping import Floas as F, Bool
from torch import tensor as T

def where_invalid_cases(x: Bool[T, "B X Y"], input: F[T, "B X Y"], other: F[T, "B X Y"]):
    B, X, Y, J = 2, 5, 32, 97
    out_2 = torch.where(x, input, torch.zeros(J, J, B))
    input_wrong = torch.zeros(B, X)
    other_value = 32
    out_3 = torch.where(x.int(), input_wrong, 39)
    out_3 = x.int().where(input_wrong, 39)
