# test 1
import torch
from jaxtyping import Float as F
from torch import Tensor
from multi_callee import multiply_child_unnanotated_torch_mm_aliased as mc


def hovering_with_inference_on_arg_with_mm():
    B, X, R, O = 4, 2, 16, 8
    x: F[Tensor, "B X R"] = torch.Tensor(B, X, R)
    y: F[Tensor, "R O"] = torch.Tensor(R, O)
    # hovering z should return [F, "B X O"]
    z = mc(x, y)
    return z
