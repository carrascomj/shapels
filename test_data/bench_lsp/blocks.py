import torch.nn as nn
from jaxtyping import Float as F
from torch import Tensor as T

from helpers import gated_update, mlp_update, normalize_like, residual_add


class ResidualBlock(nn.Module):
    """Unnannotated, will run inference from the caller to the forward body."""
    def forward(self, x, up, down, gate):
        mixed = mlp_update(x, up, down)
        gated = gated_update(mixed, gate)
        out = residual_add(normalize_like(mixed), gated)
        return out


class EncoderStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = ResidualBlock()

    def forward(
        self,
        x: F[T, "B T H"],
        up: F[T, "H M"],
        down: F[T, "M H"],
        gate: F[T, "H H"],
    ) -> F[T, "B T H"]:
        h1 = self.block(x, up, down, gate)
        h2 = self.block(h1, up, down, gate)
        return h2


class ProjectionHead(nn.Module):
    def forward(self, x: F[T, "B T H"], proj: F[T, "H O"]) -> F[T, "B T O"]:
        logits = x @ proj
        return logits
