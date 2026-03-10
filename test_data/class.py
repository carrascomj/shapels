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



# test 8
import torch
from jaxtyping import Float as F
from torch import Tensor as T

class UserLinearWithAnotherMethod(torch.nn.Module):
    """The forward function is wrong but we only care about the other method."""
    def ones_from_input(self, x):
        return torch.ones_like(x)

    def forward(self, x: F[T, "B X R"], y: F[T, "R O"]) -> tuple[F[T, "B X O"] | None, F[T, "B O O"]]:
        z = x @ y.T
        return z, z


def inference_runs_through_method_callee(linear: UserLinearWithAnotherMethod, x: F[T, "B X R"]):
    out = linear.ones_from_input(x)

# test 9
import torch
from jaxtyping import Float as F
from torch import Tensor as T

class UserLinearWithWrongMethod(torch.nn.Module):

    """The forward function is wrong but shapels will resolve to the type hint at the caller site."""
    def ones_from_input(self, x: F[T, "B X R"]) -> F[T, "B X R"], F[T, "B O T"]:
        return torch.ones(25, 72)


def inference_shorcircuits_through_method_callee_type_hints():
    B, O = 3, 9
    x: F[T, "B X R"] = torch.ones(B, X, R)
    linear = UserLinearWithWrongMethod()
    output, output2 = linear.ones_from_input(x)


# test 10
import torch
from multi_callee import UserLinear


class MyModel(torch.nn.Module):
    def __init__(self):
        self.proj = UserLinear()

    def forward(self, x, y):
        B, X, Y = x.shape
        Y, Z = y.shape
        z = self.proj(x, y)
        return z


# test 11
from multi_callee import UserLinear


class MyModelNested(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = UserLinear()

    def forward(self, x, y):
        z = self.proj(x, y)
        return z.T


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = MyModelNested()

    def forward(self, x, y):
        B, X, U = x.shape
        U, L = y.shape
        z = self.proj(x, y)
        return z


# test 12
from torch import Tensor as T
from jaxtyping import Float as F
from multi_callee import UserLinear


class MyModelNested(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = UserLinear()

    def forward(self, x: F[T, "B A X"], y: F[T, "A X"]) -> tuple[F[T, "B A A"], F[T, "B X A"]]:
        z = self.proj(x, y.T)
        return z, y


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = MyModelNested()

    def forward(self, x, y):
        z, u = self.proj(x, y)
        return z.T

    def another_method(self, x: F[T, "B A X"], y: F[T, "A X"]):
        z, u = self.proj(x, y)
        x = z.view(1, 0, 2)
        return x


# test 13
from torch import Tensor as T
from jaxtyping import Float as F
from multi_callee import UserLinear


class MyModelNestedNoSelf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = UserLinear()

    def forward(x: F[T, "B A X"], y: F[T, "A X"]) -> tuple[F[T, "B A A"], F[T, "B X A"]]:
        # should emit a diagnostic, self unknown
        w = self.proj(x, y.T)
        return z, y

