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
    bad_y = torch.ones(U, W)
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
    bad_y = torch.ones(U, W)
    output = linear(x, y)
    output2 = linear(x, bad_y)


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
    bad_y = torch.ones(U, W)
    output = linear(x, y)
    output2 = linear(x, bad_y)


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
    R, O, U, W = 27, 81, 243, 729
    y = torch.ones(R, O)
    bad_y = torch.ones(U, W)
    output = linear(alias_x, y)
    output2 = linear(alias_x, bad_y)
