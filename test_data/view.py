# test 1
from jaxtyping import Float as F
from torch import Tensor as T

def view_returns_right_dimensions():
    B, X, R, O = 4, 2, 16, 8
    x: F[T, "B X R"] = torch.Tensor(B, X, R)
    x = x.view(B*X, R, O)
    y = x.view(B*X, -1)


# test 2
from jaxtyping import Float as F
from torch import Tensor as T

def reshape_returns_right_dimensions():
    B, X, Watch = 4, 2, 32
    x: F[T, "B X Watch"] = torch.Tensor(B, X, Watch)
    y: F[T, "B X*Watch"] = x.reshape(B, X*Watch)
    z = torch.exp(x.reshape(B, X*Watch))
    exp_then_reshape = torch.exp(x).reshape(B, X*Watch)


# test 3
from jaxtyping import Float as F
from torch import Tensor as T

def squeeze_unsqueeze_after_multiply_works():
    B, X, R = 4, 2, 32
    x: F[T, "B X R"] = torch.Tensor(B, X, R)
    y: F[T, " R"] = torch.Tensor(R,)
    y = y.unsqueeze(1)
    matmul_out = x @ y
    z = matmul_out.squeeze(-1)
    return z


# test 4
from jaxtyping import Float as F
from torch import Tensor as T

def squeeze_unsqueeze_after_multiply_works():
    B, X, Watch = 4, 2, 32
    x: F[T, "B X R"] = torch.Tensor(B, X, R)
    y: F[T, " R"] = torch.Tensor(R,)
    z = (x.exp() @ y.unsqueeze(1)).squeeze(-1)
    return z

# test 5
from jaxtyping import Float as F
from torch import Tensor as T

def squeeze_nonexisting_dims_produces_diagnostics():
    B, X, R = 4, 2, 32
    x: F[T, "B X R"] = torch.Tensor(B, X, R)
    z = x.squeeze(1)
    z = x.squeeze(dim=(2, 4))
    z = torch.squeeze(x, dim=1)
    z = torch.squeeze(x, dim=5)

# test 6
from jaxtyping import Float as F
from torch import Tensor as T

def squeeze_all_is_correct():
    x: F[T, "B 1 R 1"] = torch.Tensor(B, 1, R, 1)
    z = x.squeeze().bool()

# test 7
from jaxtyping import Float as F
from torch import Tensor as T

def unsqueeze_first_is_correct():
    x: F[T, "B R"] = torch.Tensor(B, R)
    y: F[T, "R"] = torch.Tensor(R, )
    z_pos = torch.unsqueeze(x.to(torch.int32), 0)
    z_arg = y.unsqueeze(dim=0)
    z_as_x = x


# test 8
import torch

def expand_is_proper():
    A, B, C, Ex = 2, 8, 16, 32
    x = torch.Tensor(A, B, C)
    y = x.unsqueeze(1).expand(-1, Ex, B, C)


# test 9
import torch

def expand_improper():
    A, B, C, Ex = 2, 8, 16, 32
    x = torch.Tensor(A, 1, B, C)
    y = x.expand(-1, Ex, C, B)


# test 10
import torch

def repeat_is_proper():
    """Example adapter from torch docs: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.repeat.html#torch.Tensor.repeat"""
    x = torch.Tensor(3)
    # [4 6]
    repeated_2d = x.repeat(4, 2)
    # [4 2 3]
    repeated_3d = x.repeat(4, 2, 1)


# test 11
import torch

def repeat_improper(x: F[T, "A B"]):
    C = 32
    # 0 dimensions are allowed
    repeated_2d = x.repeat(0, C)
    # negative dimensions are not allowed
    # "Trying to create tensor with negative dimension -1: [C, -1, 3]"
    repeated_3d = x.repeat(C, -1, 3)


# test 12
import torch

def flatten_proper_symbolic(x):
    A, B, C, D, E, F = x.shape
    # flat is [A B*C*D E F]
    flat = torch.flatten(x, start_dim=1, end_dim=3)  


# test 13
import torch

def flatten_proper_concrete():
    x = torch.zeros(32, 64, 128)
    # flat is [32, 8192]
    flat = x.flatten(start_dim=1)


# test 14
from jaxtyping import Float as F
from torch import Tensor as T
import torch

def flatten_improper(x: F[T, "A B"]):
    wrong = x.flatten(start_dim=3)
    

# test 15
from jaxtyping import Float as F
from torch import Tensor as T, ravel

def flatten_improper(x: F[T, "A B C D"]):
    flat = ravel(x)

# test 16
import torch

class FlattenDiversion(torch.nn.Module):
    def __init__(self):
        self.op = torch.nn.Flatten(start_dim=1, end_dim=3)
    def forward(self, x):
        return self.op(x)


def flatten_proper_symbolic(x):
    A, B, C, D, E, F = x.shape
    my_flatten_diversion = FlattenDiversion()
    # flat is [A B*C*D E F]
    flat = my_flatten_diversion(x, start_dim=1, end_dim=3)


# test 17
import torch
from torch import nn

def flatten_proper_concrete():
    x = torch.zeros(32, 64, 128)
    # flat is [32, 8192]
    flattener = nn.Flatten(start_dim=1)
    flat = flattener(x)


# test 18
from jaxtyping import Float as F
from torch import Tensor as T
import torch
import torch.nn as nn

def flatten_improper(x: F[T, "A B"]):
    flattener = nn.Flatten(start_dim=3)
    wrong = flattener(x)

# test 19
import torch
def repeat_interleave_concrete():
    """Adapted rom torch docs."""
    x = torch.zeros(3)
    # [6]
    rep = x.repeat_interleave(2)
    y = torch.ones(2, 2)
    # [8]
    rep2 = torch.repeat_interleave(y, 2)
    # tensor([1, 1, 2, 2, 3, 3, 4, 4])

    # [2, 6]
    rep3 = torch.repeat_interleave(y, 3, dim=1)
    # tensor([[1, 1, 1, 2, 2, 2],
    #        [3, 3, 3, 4, 4, 4]])
    
    # [3, 2]
    rep_tensor_0 = torch.repeat_interleave(y, torch.tensor([1, 2]), dim=0)
    # tensor([[1, 2],
    #         [3, 4],
    #         [3, 4]])


# test 20
import torch

def repeat_interleave_symbolic():
    A, B, C = 2, 4, 8
    y = torch.ones(A, B)
    # [A, B*C]
    rep1 = torch.repeat_interleave(y, C, dim=1)


# test 21
import torch

def repeat_interleave_improper():
    A, B, C = 2, 4, 8
    y = torch.ones(A, B)
    # Dimension out of range (expected to be in range of [-2, 1], but got 4)
    wrong = torch.repeat_interleave(y, C, dim=4)
