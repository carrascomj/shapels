# test 1
import torch
from jaxtyping import Float
from torch import Tensor


def permute_legal_and_illegal():
    B, X, R = 4, 2, 32
    x: Float[Tensor, "B X R"] = torch.Tensor(B, X, R)
    y = x.permute(0, 2, 1)
    z = x.permute(1, 0, 2)
    invalid = x.permute(1, 2, 2)
    invalid2 = x.permute(8, 0, 2)


# test 2
def transpose_legal_and_illegal():
    B, X, R = 4, 2, 32
    x: Float[Tensor, "B X R"] = torch.Tensor(B, X, R)
    y = torch.transpose(x, 1, 2)
    z = x.transpose(0, 1)
    invalid = torch.transpose(x, 0, 0)
    invalid2 = x.transpose(0, 5)


# test 3
import torch
from jaxtyping import F
from torch import Tensor

def torch_t():
    B, X, R = 4, 2, 32
    x: F[Tensor, "B X R"] = torch.Tensor(B, X, R)
    y = torch.t(x).to(torch.bfloat16)
    # torch.t for 0-dim and 1-dim returns the tensors as is
    z = x.T
    w = x.t()


# test 4
import torch
from jaxtyping import F
from torch import Tensor

def torch_t_oneliner():
    B, X, R = 4, 2, 32
    x: F[Tensor, "B X R"] = torch.Tensor(B, X, R)
    # torch.t for 0-dim and 1-dim returns the tensors as is
    z = x.sum(dim=(1, 2)).t()


# test 5
import torch

def transpose_negative(images: torch.Tensor):
    n_crops, B, rgb, H, W = images.shape
    transposed = images.transpose(-2, 1)
    permuted = transposed.permute(-4, 2, 3, 0, -1)
    # wrong; -6 is out of range
    transposed_wrong = images.transpose(-6, 1)
    # wrong; -1 and 4 are the same dimension
    permuted_wrong = transposed.permute(4, 2, 3, 0, -1)
