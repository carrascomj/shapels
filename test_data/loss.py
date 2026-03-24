"""Tests related to losses."""

# test 1
from jaxtyping import Float as F
import torch

def loss_none(x: F[T, "B X Y"]):
    mse = torch.nn.MSELoss(reduction="none")
    loss = mse(x)
    

# test 2
from jaxtyping import Float as F
import torch

def loss_mean(x: F[T, "B X Y"]):
    l1 = torch.nn.L1Loss(reduction="mean")
    loss = l1(x)



# test 3
from jaxtyping import Float as F
import torch

def loss_default(x: F[T, "B X Y"]):
    bce = torch.nn.BCELoss()
    loss = bce(x)


# test 4
from jaxtyping import Float as F
import torch

def loss_none_by_position(x: F[T, "B X Y"]):
    bce = torch.nn.BCELoss(None, None, None, "none")
    loss = bce(x)


# test 5
from jaxtyping import Float as F
import torch.nn.functional as fn

def functional_loss_none(x: F[T, "B X Y"]):
    loss = fn.mse_loss(x, x, reduction="none")


# test 6
from jaxtyping import Float as F
from torch.nn import functional as fn

def functional_loss_mean(x: F[T, "B X Y"]):
    loss = fn.l1_loss(x, x, reduction="mean")


# test 7
from jaxtyping import Float as F
import torch

def functional_loss_default(x: F[T, "B X Y"]):
    loss = torch.nn.functional.binary_cross_entropy(x, x)


# test 8
from jaxtyping import Float as F
from torch.nn.functional import binary_cross_entropy

def functional_loss_none_by_position(x: F[T, "B X Y"]):
    loss = binary_cross_entropy(x, x, None, None, None, "none")
