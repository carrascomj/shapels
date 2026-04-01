"""Tests related to losses."""

# test 1
from jaxtyping import Float as F
import torch

def loss_none(x: F[T, "B X Y"]):
    mse = torch.nn.MSELoss(reduction="none")
    y = torch.ones_like(x)
    loss = mse(x, y)
    

# test 2
from jaxtyping import Float as F
import torch

def loss_mean(x: F[T, "B X Y"]):
    B, X, Y = x.shape
    l1 = torch.nn.L1Loss(reduction="mean")
    loss = l1(x, x.new_zeros(B, X, Y))



# test 3
from jaxtyping import Float as F
import torch

def loss_default(x: F[T, "B X Y"]):
    bce = torch.nn.BCELoss()
    loss = bce(x, torch.randn_like(x))


# test 4
from jaxtyping import Float as F
import torch

def loss_none_by_position(x: F[T, "B X Y"]):
    bce = torch.nn.BCELoss(None, None, None, "none")
    loss = bce(x, torch.randn_like(x))


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


# test 9
from jaxtyping import Float as F, Int as I
from torch import Tensor as T
import torch

def cross_entropy_none(logits: F[T, "B C H W"], target: I[T, "B H W"]):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fn(logits, target)


# test 10
from jaxtyping import Float as F, Int as I
from torch import Tensor as T
import torch

def cross_entropy_wrong_target_shape(logits: F[T, "B C H W"], target: I[T, "B H"]):
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits, target)


# test 11
from jaxtyping import Float as F, Int as I
from torch import Tensor as T
import torch

def ctc_loss_none(log_probs: F[T, "T N C"], targets: I[T, "N S"], input_lengths: I[T, "N"], target_lengths: I[T, "N"]):
    loss_fn = torch.nn.CTCLoss(reduction="none")
    loss = loss_fn(log_probs, targets, input_lengths, target_lengths)


# test 12
from jaxtyping import Float as F, Int as I
from torch import Tensor as T
import torch

def ctc_loss_wrong_input_lengths(log_probs: F[T, "T N C"], targets: I[T, "N S"], input_lengths: I[T, "T"], target_lengths: I[T, "N"]):
    loss_fn = torch.nn.CTCLoss(reduction="none")
    loss = loss_fn(log_probs, targets, input_lengths, target_lengths)


# test 13
from jaxtyping import Float as F, Int as I
from torch import Tensor as T
import torch

def cosine_embedding_loss_none(x1: F[T, "N D"], x2: F[T, "N D"], target: I[T, "N"]):
    loss_fn = torch.nn.CosineEmbeddingLoss(reduction="none")
    loss = loss_fn(x1, x2, target)


# test 14
from jaxtyping import Float as F, Int as I
from torch import Tensor as T
import torch

def cosine_embedding_loss_wrong_target(x1: F[T, "N D"], x2: F[T, "N D"], target: I[T, "N D"]):
    loss_fn = torch.nn.CosineEmbeddingLoss(reduction="none")
    loss = loss_fn(x1, x2, target)


# test 15
from jaxtyping import Float as F
from torch import Tensor as T
import torch

def triplet_margin_loss_none(anchor: F[T, "N D"], positive: F[T, "N D"], negative: F[T, "N D"]):
    loss_fn = torch.nn.TripletMarginLoss(reduction="none")
    loss = loss_fn(anchor, positive, negative)


# test 16
from jaxtyping import Float as F
from torch import Tensor as T
import torch

def triplet_margin_loss_wrong_positive(anchor: F[T, "N D"], positive: F[T, "M D"], negative: F[T, "N D"]):
    loss_fn = torch.nn.TripletMarginLoss(reduction="none")
    loss = loss_fn(anchor, positive, negative)


# test 17
from jaxtyping import Float as F
from torch import Tensor as T
import torch.nn.functional as fn

def functional_triplet_margin_with_distance_loss_none(anchor: F[T, "N A B"], positive: F[T, "N A B"], negative: F[T, "N A B"]):
    loss = fn.triplet_margin_with_distance_loss(anchor, positive, negative, reduction="none")


# test 18
from jaxtyping import Float as F
from torch import Tensor as T
import torch.nn.functional as fn

def functional_triplet_margin_with_distance_loss_wrong_negative(anchor: F[T, "N A B"], positive: F[T, "N A B"], negative: F[T, "M A B"]):
    loss = fn.triplet_margin_with_distance_loss(anchor, positive, negative, reduction="none")
