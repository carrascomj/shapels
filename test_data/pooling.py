"""Tests related to pooling torch.nn.Modules."""

# test 1
from jaxtyping import Float as F
from torch import Tensor as T
from torch import nn


def maxpool1d_concrete(x: F[T, "4 8 21"]):
    pool = nn.MaxPool1d(3, stride=2)
    y = pool(x)
    return y


# test 2
from jaxtyping import Float as F
from torch import Tensor as T
import torch


def maxpool2d_concrete(x: F[T, "32 16 20 18"]):
    pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    y = pool(x)
    return y


# test 3
from jaxtyping import Float as F
from torch import Tensor as T
from torch.nn import AvgPool3d


def avgpool3d_concrete(x: F[T, "2 8 10 29 31"]):
    pool = AvgPool3d(kernel_size=(2, 3, 5), stride=(1, 2, 3))
    y = pool(x)
    return y


# test 4
from jaxtyping import Float as F
from torch import Tensor as T
import torch


def adaptiveavgpool2d_scalar(x: F[T, "B C H W"]):
    pool = torch.nn.AdaptiveAvgPool2d(1)
    y = pool(x)
    return y


# test 5
from jaxtyping import Float as F
from torch import Tensor as T
from torch import nn


def adaptivemaxpool2d_partial(x: F[T, "B C H W"]):
    pool = nn.AdaptiveMaxPool2d((None, 7))
    y = pool(x)
    return y


# test 6
from jaxtyping import Float as F
from torch import Tensor as T
from torch import nn


def fractionalmaxpool2d_output_size(x: F[T, "2 3 20 18"]):
    pool = nn.FractionalMaxPool2d(3, output_size=(5, 7))
    y = pool(x)
    return y


# test 7
from jaxtyping import Float as F
from torch import Tensor as T
from torch.nn import LPPool1d


def lppool1d_concrete(x: F[T, "4 8 21"]):
    pool = LPPool1d(2, kernel_size=3, stride=2)
    y = pool(x)
    return y


# test 8
from jaxtyping import Float as F
from torch import Tensor as T
from torch import nn


def wrong_input_for_maxpool2d(x: F[T, "B C"]):
    pool = nn.MaxPool2d(2)
    y = pool(x)
    return y


# test 9
from jaxtyping import Float as F
from torch import Tensor as T
from torch import nn


def maxpool2d_mixed_symbolic_and_concrete(x: F[T, "2 8 29 31"]):
    pool = nn.MaxPool2d(
        kernel_size=(3, 5),
        stride=(S, 2),
        padding=(0, P),
        dilation=(1, D),
    )
    y = pool(x)
    return y


# test 10
from jaxtyping import Float as F
from torch import Tensor as T
import torch


def avgpool2d_mixed_symbolic_and_concrete(x: F[T, "2 8 H 31"]):
    pool = torch.nn.AvgPool2d(kernel_size=(3, 5), stride=(S, 2), padding=(0, 1))
    y = pool(x)
    return y


# test 11
from jaxtyping import Float as F
from torch import Tensor as T
import torch.nn.functional as fn


def max_pool1d_concrete(x: F[T, "4 8 21"]):
    y = fn.max_pool1d(x, 3, stride=2)
    return y


# test 12
from jaxtyping import Float as F
from torch import Tensor as T
from torch.nn.functional import max_pool2d


def max_pool2d_concrete(x: F[T, "32 16 20 18"]):
    y = max_pool2d(x, kernel_size=3, stride=2, padding=1)
    return y


# test 13
from jaxtyping import Float as F
from torch import Tensor as T
import torch


def avg_pool3d_concrete(x: F[T, "2 8 10 29 31"]):
    y = torch.nn.functional.avg_pool3d(x, kernel_size=(2, 3, 5), stride=(1, 2, 3))
    return y


# test 14
from jaxtyping import Float as F
from torch import Tensor as T
from torch.nn import functional as fn


def adaptive_avg_pool2d_scalar(x: F[T, "B C H W"]):
    y = fn.adaptive_avg_pool2d(x, 1)
    return y


# test 15
from jaxtyping import Float as F
from torch import Tensor as T
from torch.nn.functional import adaptive_max_pool2d


def adaptive_max_pool2d_partial(x: F[T, "B C H W"]):
    y = adaptive_max_pool2d(x, (None, 7))
    return y


# test 16
from jaxtyping import Float as F
from torch import Tensor as T
import torch.nn.functional as fn


def fractional_max_pool2d_output_size_functional(x: F[T, "2 3 20 18"]):
    y = fn.fractional_max_pool2d(x, 3, output_size=(5, 7))
    return y


# test 17
from jaxtyping import Float as F
from torch import Tensor as T
from torch.nn.functional import lp_pool1d


def lp_pool1d_concrete(x: F[T, "4 8 21"]):
    y = lp_pool1d(x, 2, kernel_size=3, stride=2)
    return y


# test 18
from jaxtyping import Float as F
from torch import Tensor as T
from torch.nn import functional as fn


def wrong_input_for_max_pool2d(x: F[T, "B C"]):
    y = fn.max_pool2d(x, 2)
    return y


# test 19
from jaxtyping import Float as F
from torch import Tensor as T
import torch.nn.functional as fn


def max_pool2d_mixed_symbolic_and_concrete(x: F[T, "2 8 29 31"]):
    y = fn.max_pool2d(
        x,
        kernel_size=(3, 5),
        stride=(S, 2),
        padding=(0, P),
        dilation=(1, D),
    )
    return y


# test 20
from jaxtyping import Float as F
from torch import Tensor as T
import torch


def avg_pool2d_mixed_symbolic_and_concrete_functional(x: F[T, "2 8 H 31"]):
    y = torch.nn.functional.avg_pool2d(x, kernel_size=(3, 5), stride=(S, 2), padding=(0, 1))
    return y
