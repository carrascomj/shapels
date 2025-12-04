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
    B, X, Watch = 4, 2, 32
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
    z = x.squeeze()

# test 7
from jaxtyping import Float as F
from torch import Tensor as T

def unsqueeze_first_is_correct():
    x: F[T, "B R"] = torch.Tensor(B, R)
    y: F[T, "R"] = torch.Tensor(R, )
    z_pos = torch.unsqueeze(x, 0)
    z_arg = y.unsqueeze(dim=0)
