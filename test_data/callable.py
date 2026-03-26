"""Tests related to Callable."""

# test 1
from typing import Callable
import torch
from jaxtyping import Float as F
from torch import Tensor as T
from typing import Callable

def empty_args_callable_is_inferred(initializer: Callable[[], F[T, "H J K L"]]):
    x = initializer()
    L, D = 32, 9
    mat = torch.zeros(L, D)
    y = x @ mat
    return y


# test 2
from typing import Callable
import torch
from jaxtyping import Float

def wrong_dim_arg_should_emit_diagnostics(op: Callable[[Float[torch.Tensor, "H J K L"]], Float[torch.Tensor, "H J K L"]]):
    L, D = 32, 9
    mat = torch.zeros(L, D)
    # should raise a diagnostics, different number of dimensions
    x = op(mat)
    return x


# test 3
from typing import Callable
import torch
from jaxtyping import Float

def multiple_args_callable_is_inferred(loss_fn: Callable[[Float[torch.Tensor, "H J K L"], Float[torch.Tensor, "H J K L"]], Float[torch.Tensor, "H"]]):
    H, J, K, L = 3, 9, 27, 81
    x = torch.Tensor(H, J, K, L)
    y = torch.ones(H, J, K, L)
    loss = loss_fn(x, y)
    return loss


# test 4
from typing import Callable
import torch
from jaxtyping import Float

def annotated_overwrites_arg_union(loss_fn: Callable[[Float[torch.Tensor, "H J K L"], Float[torch.Tensor, "H J K L"]], Float[torch.Tensor, "H J K L"]] | torch.nn.MSELoss):
    H, J, K, L = 3, 9, 27, 81
    x = torch.Tensor(H, J, K, L)
    y = torch.ones(H, J, K, L)
    loss = loss_fn(x, y)
    return loss


# test 5
from typing import Callable
import torch
from jaxtyping import Float


def annotated_var_overwrites(loss_fn: torch.nn.MSELoss):
    loss_fn: Callable[[Float[torch.Tensor, "H J K L"], Float[torch.Tensor, "H J K L"]], Float[torch.Tensor, "H J K L"]]
    H, J, K, L = 3, 9, 27, 81
    x = torch.Tensor(H, J, K, L)
    y = torch.ones(H, J, K, L)
    loss = loss_fn(x, y)
    return loss

# test 6
from typing import Callable
import torch
from jaxtyping import Float


def wrong_fn(x, y):
    return x, y

def annotated_ann_asignment_overwrites():
    loss_fn: Callable[[Float[torch.Tensor, "A B"], Float[torch.Tensor, "A B"]], Float[torch.Tensor, "X Y Z"]] = wrong_fn
    A, B = 3, 9
    x = torch.Tensor(A, B)
    y = torch.ones(A, B)
    loss = loss_fn(x, y)
    return loss


# test 7
from typing import Callable
import torch
from jaxtyping import Float


def tuple_destructuring_works(some_fn: Callable[[Float[torch.Tensor, "A B"], Float[torch.Tensor, "A B"]], tuple[Float[torch.Tensor, "X Y Z"], Float[torch.Tensor, "X Y"]]] ):
    A, B = 3, 9
    x = torch.Tensor(A, B)
    y = torch.ones(A, B)
    a, b = some_fn(x, y)
    out = a.T @ b.T
    return out
