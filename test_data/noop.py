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
