import torch.nn as nn
from jaxtyping import Float as F
from torch import Tensor as T

from helpers import gated_update, mlp_update, normalize_like, residual_add


class ResidualBlock(nn.Module):
    def forward(
        self,
        x: F[T, "B T H"],
        up: F[T, "H M"],
        down: F[T, "M H"],
        gate: F[T, "H H"],
    ) -> F[T, "B T H"]:
        base = normalize_like(x)
        updated = mlp_update(base, up, down)
        mixed = gated_update(updated, gate)
        out = residual_add(updated, mixed)
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
        h2 = normalize_like(self.block(h1, up, down, gate))
        return h2


class ProjectionHead(nn.Module):
    def forward(self, x: F[T, "B T H"], proj: F[T, "H O"]) -> F[T, "B T O"]:
        activated = normalize_like(x)
        logits = activated @ proj
        return logits
