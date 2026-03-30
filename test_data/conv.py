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

# test 6
import torch
from torch import nn

def infer_conv1d_module():
    x = torch.ones(32, 256, 512)
    conv = nn.Conv1d(256, 128, 3)
    out = conv(x)
    return out

# test 7
import torch
import torch.nn as nn

x = torch.ones(32, 256, H, W, dtype=torch.float)   # H, W arbitrary spatial sizes
conv = nn.Conv2d(256, 128, 3, stride=2, padding=6, dilation=3)

y = conv(x)

# test 8
import torch
from jaxtyping import Float
from torch import Tensor as T
import torch.nn as nn


class Conv3dModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv3d(4, 8, (3, 5, 7), stride=(2, 3, 4), padding=(1, 2, 3), dilation=(1, 2, 1))
        self.out_layer = nn.Sequential(
            nn.SiLU()
        )
        self.param = nn.Parameter(torch.ones(10, 128))

    def forward(self, x):
        return self.out_layer(self.filter(x)) @ self.param

        
def infer_conv3d_module_allconcrete(x: Float[T, "2 4 20 30 40"]):
    conv = Conv3dModule()
    y = conv(x)


# test 9
import torch
from jaxtyping import Float
from torch import Tensor as T
import torch.nn as nn

def infer_conv3d_module_mixed(x: Float[T, "2 4 20 30 40"]):
    conv = nn.Conv3d(4, 8, (3, 5, 7), stride=(S, 3, 4), padding=(1, P, 3), dilation=(1, D, 1))
    y = conv(x)


# test 10
import torch
import torch.nn as nn

def wrong_conv1d_module():
    x = torch.ones(512)
    conv = nn.Conv1d(256, 18, 9)
    out = conv(x)
    return out

# test 11
import torch
from torch import nn

def infer_conv_transpose1d_module():
    x = torch.ones(32, 256, 20)
    conv = nn.ConvTranspose1d(256, 128, 3, 2, 1, 1, 1, False, 2)
    out = conv(x)
    return out

# test 12
import torch
import torch.nn as nn

x = torch.ones(32, 256, H, W, dtype=torch.float)
conv = nn.ConvTranspose2d(256, 128, 2, stride=2)

y = conv(x)

# test 13
import torch
from jaxtyping import Float
from torch import Tensor as T
import torch.nn as nn


class ConvTranspose3dModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.ConvTranspose3d(
            4,
            8,
            (3, 5, 7),
            stride=(2, 3, 4),
            padding=(1, 2, 3),
            output_padding=(0, 1, 0),
            dilation=(1, 2, 1),
        )
        self.out_layer = nn.Sequential(
            nn.SiLU()
        )
        self.param = nn.Parameter(torch.ones(157, 9))

    def forward(self, x):
        return self.out_layer(self.filter(x)) @ self.param


def infer_conv_transpose3d_module_allconcrete(x: Float[T, "2 4 20 30 40"]):
    conv = ConvTranspose3dModule()
    y = conv(x)


# test 14
import torch
from jaxtyping import Float
from torch import Tensor as T
import torch.nn as nn

def infer_conv_transpose3d_module_mixed(x: Float[T, "2 4 D H W"]):
    conv = nn.ConvTranspose3d(
        4,
        8,
        (2, 5, 3),
        stride=(2, 3, 1),
        padding=(0, 1, 2),
        output_padding=(0, 1, 0),
        dilation=(1, 2, 1),
    )
    y = conv(x)


# test 15
import torch
import torch.nn as nn

def wrong_conv_transpose1d_module():
    x = torch.ones(512)
    conv = nn.ConvTranspose1d(256, 18, 9)
    out = conv(x)
    return out
