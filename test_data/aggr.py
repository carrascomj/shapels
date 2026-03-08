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


# test 4
def keepdim_is_understood(img):
    a, b, cropped_h, cropped_w = img.shape
    # expand wrongly receives a base of 2 dims instead of 4
    img = img.amin(dim=(2, 3), keepdim=True).expand(-1, -1, cropped_h, cropped_w)


# test 5
from jaxtyping import Float as F
from torch import Tensor as T

def quantile_single_q():
    B, X, Watch = 4, 2, 32
    x: F[T, "B X R"] = torch.Tensor(B, X, R)
    y: F[T, " R"] = torch.Tensor(R,)
    y = y.unsqueeze(1)
    matmul_out = x @ y
    z = matmul_out.quantile(0.9, 1)
    return z


# test 6
from jaxtyping import Float as F
from torch import Tensor as T

def quantile_multi_q():
    """Example from https://docs.pytorch.org/docs/stable/generated/torch.quantile.html."""
    B, X, Watch = 4, 2, 32
    a = torch.randn(2, 3)
    q = torch.tensor([0.25, 0.5, 0.75])
    z = torch.quantile(a, q, dim=1, keepdim=1)
    return z
