# test 1
import torch
from torch.nn.functional import conv1d

def infer_conv1d():
    x = torch.ones(32, 256, 512)
    kernel = torch.ones(128, 256, 3)
    out = conv1d(x, kernel)
    return out

# test 2
import torch
import torch.nn.functional as F

x = torch.ones(32, 256, H, W, dtype=torch.float)   # H, W arbitrary spatial sizes
kernel = torch.ones(128, 256, 3, 3, dtype=torch.float)  # (C_out=128, C_in=256, K_h=3, K_w=3)

y = F.conv2d(
    x,
    kernel,
    stride=2,
    padding=6,
    dilation=3
)

# test 3
import torch
from jaxtyping import Float
from torch import Tensor as T
import torch.nn.functional as F

def infer_conv3d_allconcrete(x: Float[T, "2 4 20 30 40"]):
    w = torch.ones(8, 4, 3, 5, 7)
    y = F.conv3d(x, w, stride=(2, 3, 4), padding=(1, 2, 3), dilation=(1, 2, 1))


# test 4
import torch
from jaxtyping import Float
from torch import Tensor as T
import torch.nn.functional as F

def infer_conv3d_allconcrete(x: Float[T, "2 4 20 30 40"]):
    w = torch.ones(8, 4, 3, 5, 7)
    y = F.conv3d(x, w, stride=(S, 3, 4), padding=(1, P, 3), dilation=(1, D, 1))



# test 5
import torch
from torch.nn.functional import conv1d

def wrong_conv1d():
    x = torch.ones(512)
    kernel = torch.ones(18, 256, 9)
    out = conv1d(x, kernel)
    return out
