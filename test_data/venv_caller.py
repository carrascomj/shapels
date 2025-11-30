# test 1
import torch
from jaxtyping import Float as F
from torch import Tensor
import mypkg


def hovering_with_inference_on_arg_with_mm():
    B, X, R, O = 4, 2, 16, 8
    x: F[Tensor, "B X R"] = torch.Tensor(B, X, R)
    y: F[Tensor, "R O"] = torch.Tensor(R, O)
    # hovering z should return [F, "B X O"]
    z = mypkg.foo(x, y)
    return z
