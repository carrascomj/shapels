# test 1
from jaxtyping import Float as F
from torch import Tensor as T

def sum_dimension():
    B, X, Watch = 4, 2, 32
    x: F[T, "B X R"] = torch.Tensor(B, X, R)
    y: F[T, " R"] = torch.Tensor(R,)
    y = y.unsqueeze(1)
    matmul_out = x @ y
    z = matmul_out.sum(1)
    return z


# test 2
from jaxtyping import Float as F
from torch import Tensor as T
from torch import gelu

def sum_multiple_dimensions():
    B, X, Watch = 4, 2, 32
    x: F[T, "B X R"] = torch.Tensor(B, X, R)
    y: F[T, " R"] = torch.Tensor(R,)
    y = gelu(y.unsqueeze(1))
    matmul_out = x @ y
    z = matmul_out.sum(dim=(1, 2))
    return z


# test 3
from jaxtyping import Float as F
from torch import Tensor as T
from torch import sum as aggr_op

def sum_all():
    B, X, Watch = 4, 2, 32
    x: F[T, "B X R"] = torch.Tensor(B, X, R)
    aggr = aggr_op(x)
    return aggr
