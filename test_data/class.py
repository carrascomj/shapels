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
        return w, y

# test 14
from torch import Tensor as T
from jaxtyping import Float as F
from multi_callee import UserLinear
import torch.nn as nn


class MyModelNestedParamConcrete(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = UserLinear()
        self.p = nn.Parameter(torch.ones(32, 57))

    def forward(self, x: F[T, "B X A"], y: F[T, "S A"]) -> tuple[F[T, "B X S"], F[T, "S A"]]:
        w = self.proj(x, y.T)
        # annotating to rename from concrete to abstract dimensions
        self.p: F[T, "S 57"]
        out = w @ self.p
        return w, y


# test 15
from torch import Tensor as T
from jaxtyping import Float as F
from multi_callee import UserLinear


class MyModelNestedParamAbstract(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = UserLinear()
        S, U = 32, 2
        self.pos_emb = torch.nn.Parameter(torch.Tensor(S, U))

    def forward(self, x: F[T, "B X A"], y: F[T, "S A"]) -> tuple[F[T, "B X S"], F[T, "S A"]]:
        w = self.proj(x, y.T)
        out = w @ self.pos_emb.sum(dim=1).unsqueeze(1)
        return w, y

# test 16
from torch import Tensor as T
from jaxtyping import Float as F
from multi_callee import UserLinear
import torch.nn as nn


class MyModelNestedParamAbstractWrong(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = UserLinear()
        L, U = 32, 2
        self.pos_emb = nn.Parameter(torch.Tensor(L, U))

    def forward(self, x: F[T, "B X A"], y: F[T, "S A"]) -> tuple[F[T, "B X S"], F[T, "S A"]]:
        w = self.proj(x, y.T)
        # should emit diagnostic, S != L
        out = w @ self.pos_emb
        return w, y


# test 17
from torch import Tensor as T
from jaxtyping import Float as F
from multi_callee import UserLinear


class MyModelNestedTensorAbstract(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = UserLinear()
        S, U = 32, 2
        self.some_tensor = torch.zeros(S, U)

    def forward(self, x: F[T, "B X A"], y: F[T, "S A"]) -> tuple[F[T, "B X S"], F[T, "S A"]]:
        w = self.proj(x, y.T)
        out = w @ self.some_tensor
        return w, y


# test 18
from torch import Tensor as T
from jaxtyping import Float as F
from multi_callee import UserLinear


class MyModelNestedTensorAbstractOps(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = UserLinear()
        S, U = 32, 2
        self.some_tensor = torch.Tensor(S, U).sum(dim=-1, keepdim=True)

    def forward(self, x: F[T, "B X A"], y: F[T, "S A"]) -> tuple[F[T, "B X S"], F[T, "S A"]]:
        w = self.proj(x, y.T)
        out = w @ self.some_tensor
        return w, y


# test 19
from torch import Tensor as T
from jaxtyping import Float as F
from multi_callee import UserLinear


class MyModelNestedParameterAbstractOps(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = UserLinear()
        S, W = 32, 2
        self.my_param = torch.nn.Parameter(torch.ones(W, S).permute(1, 0))

    def forward(self, x: F[T, "B X A"], y: F[T, "S A"]) -> tuple[F[T, "B X S"], F[T, "S A"]]:
        w = self.proj(x, y.T)
        out = w @ self.my_param
        return w, y


# test 20
from torch import Tensor as T
from jaxtyping import Float as F
from multi_callee import UserLinear
import torch.nn as nn


class MyModelNestedParameterAbstractReassigned(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = UserLinear()
        S, T = 32, 2
        self.some_param = nn.Parameter(torch.Tensor(T, S).permute(1, 0))

    def forward(self, x: F[T, "B X A"], y: F[T, "S A"]) -> tuple[F[T, "B X S"], F[T, "S A"]]:
        w = self.proj(x, y.T)
        my_tensor = self.some_param
        out = w @ my_tensor
        return w, y


# test 21
from jaxtyping import Float as F
from torch import Tensor
import torch.nn as nn


class EllipsisMlp(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: F[Tensor, "... InDim"]) -> F[Tensor, "... OutDim"]:
        return self.proj(x)


class EllipsisModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: F[Tensor, "Batch L InEmb"], module_with_annotated_ellipsis: EllipsisMlp):
        y = module_with_annotated_ellipsis(x)
        return y


# test 22
from jaxtyping import Float as F
from torch import Tensor
import torch.nn as nn


class EllipsisTuple(nn.Module):
    def forward(self, x: F[Tensor, "... InDim"]) -> tuple[F[Tensor, "... OutDim"], F[Tensor, "... InDim"]]:
        return x, x


def ellipsis_tuple_destructuring(module: EllipsisTuple, x: F[Tensor, "Batch Heads Tokens Embed"]):
    y, residual = module(x)
    return y, residual


# test 23
from jaxtyping import Float as F
from torch import Tensor as T
import torch.nn as nn


class BuiltinLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(7, 11)

    def forward(self, x: F[T, "B X 7"]):
        z = self.proj(x)
        return z


# test 24
from jaxtyping import Float as F
from torch import Tensor as T
import torch.nn as nn


class BuiltinLinearModelAbstract(nn.Module):
    def __init__(self):
        super().__init__()
        Y, Z = 7, 14
        self.proj = nn.Linear(Y, Z)

    def forward(self, x: F[T, "B X Y"]):
        z = self.proj(x)
        return z

# test 25
from jaxtyping import Float as F
from torch import Tensor as T
import torch.nn as nn


class BuiltinLinearModelAbstractWrong(nn.Module):
    def __init__(self):
        super().__init__()
        Out = 14
        self.proj = nn.Linear(7, Out)

    def forward(self, x: F[T, "B X 5"]):
        # wrong, doesn't type check
        z = self.proj(x)
        return z


# test 26
from jaxtyping import Float as F
from torch import Tensor as T
import torch.nn as nn


class BuiltinLinearModelConcreteAlphaEquiv(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(7, 12),
            nn.ReLU(),
        )   
    
    def forward(self, x: F[T, "B X Y"]):
        # B X 12
        z = self.proj(x)
        return z


# test 27
from jaxtyping import Float as F
from torch import Tensor as T
import torch.nn as nn


class BuiltinSequentialCovModel(nn.Module):
    def __init__(self):
        super().__init__()
        OutCh = 16
        self.net = nn.Sequential(
            nn.Conv2d(3, OutCh, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

    def forward(self, x: F[T, "B 3 H W"]):
        y = self.net(x)
        return y


# test 28
from jaxtyping import Float as F
from torch import Tensor as T
import torch.nn as nn


class BuiltinSequentialCovModelWrong(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

    def forward(self, x: F[T, "B 5 H W"]):
        y = self.net(x)
        return y


# test 29
from jaxtyping import Float as F
from torch import Tensor as T
from torch.nn import Buffer as Buf, Parameter as Param
import torch
import torch.nn as nn


class ImportedParameterAndBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = Param(torch.ones(32, 57))
        self.bias = Buf(torch.ones(57))

    def forward(self, x: F[T, "B X 32"]):
        y = x @ self.weight + self.bias
        return y
