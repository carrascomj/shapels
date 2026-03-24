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
