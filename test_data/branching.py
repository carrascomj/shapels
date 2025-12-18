# test 1
from jaxtyping import Float as F
from torch import Tensor as T

def proper_multiply_for_not_produce_diagnostics(x: F[T, "B X R"], y: F[T, "R S"]) -> F[T, "B X S"]:
    for _ in range(10):
        z = x @ y
    return z


# test 2
def proper_multiply_while_not_produce_diagnostics(x: F[T, "B X R"], y: F[T, "R S"]) -> F[T, "B X S"]:
    for _ in range(10):
        z = x @ y
    return z 


# test 3
def wrong_multiply_for_produces_diagnostics(x, y):
    B, X, R = x.shape
    R, S = y.shape
    for _ in range(10):
        z = x.T @ y


# test 4
def wrong_multiply_while_produces_diagnostics():
    x = torch.zeros(B, X, R)
    y = torch.zeros(R, S)
    for _ in range(10):
        z = x.T @ y


# test 5
def wrong_multiply_after_continue(x, y):
    B, X, R = x.shape
    R, S = y.shape
    for _ in range(10):
        inferred = x @ y
        return
        z = x.T @ y



# test 6
def proper_multiply_if_not_produce_diagnostics(x: F[T, "B X R"], y: F[T, "R S"]) -> F[T, "B X S"]:
    if x:
        z = x @ y
    return z


# test 7
def wrong_multiply_if_produces_diagnostics(x: F[T, "B X R"], y: F[T, "R S"]) -> F[T, "B X S"]:
    a = 2
    if (y == x) and 2 == a:
        z = x.T @ y
    return z


