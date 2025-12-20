# test 1
import torch

class Some(torch.nn.Module):
    def forward(x: F[T, "B X R"], y: F[T, "R S"]) -> F[T, "B X S"]:
        # hovering z should return [F, "B X S"]
        z = x @ y
        return z 


# test 2
import torch

class UserLinear(torch.nn.Module):
    def forward(x: F[T, "B X R"], y: F[T, "R S"]) -> F[T, "B X S"]:
        z = x @ y
        return z


def function():
    linear = UserLinear()
    B, X, R, S, U, W = 3, 9, 27, 81, 243, 729
    x = torch.zeros(B, X, R)
    y = torch.ones(R, S)
    bad_y = torch.ones(U, W, R)
    output = linear(x, y)
    output2 = linear(x, bad_y)


# test 3
import torch

class UserLinearAsArg(torch.nn.Module):
    def forward(x: F[T, "B X R"], y: F[T, "R O"]) -> F[T, "B X O"]:
        z = x @ y
        return z


def function_with_arg(linear: UserLinearAsArg):
    B, X, R, O, U, W = 3, 9, 27, 81, 243, 729
    x = torch.zeros(B, X, R)
    y = torch.ones(R, O)
    ok_alpha_y = torch.ones(U, W)
    output = linear(x, y)
    output2 = linear(x, ok_alpha_y)


# test 4
import torch

class UserLinearAsArg(torch.nn.Module):
    def forward(x: F[T, "B X R"], y: F[T, "R O"]) -> F[T, "B X O"]:
        z = x @ y
        return z


def function_with_union_arg(linear: UserLinearAsArg | DataParallel[UserLinearAsArg]):
    B, X, R, O, U, W = 3, 9, 27, 81, 243, 729
    x = torch.zeros(B, X, R)
    y = torch.ones(R, O)
    alpha_y = torch.ones(U, W)
    output = linear(x, y)
    output2 = linear(x, alpha_y)


# test 5
import torch
from jaxtyping import Float as F
from torch import Tensor as T

class UserLinearAsArg(torch.nn.Module):
    def forward(x: F[T, "B X R"], y: F[T, "R O"]) -> F[T, "B X O"]:
        z = x @ y
        return z


def function_with_union_tensor_arg(linear: UserLinearAsArg, x: None | F[T, "B X R"]):
    alias_x = x
    R, O, U = 27, 81, 243
    y = torch.ones(R, O)
    bad_y = torch.ones(U)
    output = linear(alias_x, y)
    output2 = linear(alias_x, bad_y)


# test 6
import torch
from jaxtyping import Float as F
from torch import Tensor as T

class UserLinearWrongButAnn(torch.nn.Module):
    """The forward function is wrong but shapels will resolve to `B X O` at the caller site."""
    def forward(x: F[T, "B X R"], y: F[T, "R O"]) -> F[T, "B X O"] | None:
        z = x @ y.T
        return z


def call_type_hinted_but_wrong_forward(linear: UserLinearWrongButAnn, x: F[T, "B X R"]):
    R, O, U, W = 27, 81, 243, 729
    y = torch.ones(R, O)
    # output should have shape [B, X, O]: F nonetheless
    output = linear(x, y)



# test 7
import torch
from jaxtyping import Float as F
from torch import Tensor as T

class UserLinearWrongButAnnTuple(torch.nn.Module):
    """The forward function is wrong but shapels will resolve to (`B X O`, `B O O`) at the caller site."""
    def forward(x: F[T, "B X R"], y: F[T, "R O"]) -> tuple[F[T, "B X O"] | None, F[T, "B O O"]]:
        z = x @ y.T
        return z, z


def call_type_hinted_but_wrong_forward(linear: UserLinearWrongButAnnTuple, x: F[T, "B X R"]):
    R, O, U, W = 27, 81, 243, 729
    y = torch.ones(R, O)
    # output should have shape [B, X, O]: F nonetheless
    # output2 should have shape [B, O, O]: F nonetheless
    output, output2 = linear(x, y)
