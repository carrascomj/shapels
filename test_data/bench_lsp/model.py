import torch
import torch.nn as nn
from jaxtyping import Float as F
from torch import Tensor as T

from blocks import EncoderStage, ProjectionHead
from helpers import mlp_update, normalize_like, project_features, residual_add


def encode_once(
    x: F[T, "B T H"],
    stage: EncoderStage,
    up: F[T, "H M"],
    down: F[T, "M H"],
    gate: F[T, "H H"],
) -> F[T, "B T H"]:
    out = stage(x, up, down, gate)
    return out


def encode_twice(
    x: F[T, "B T H"],
    stage: EncoderStage,
    up: F[T, "H M"],
    down: F[T, "M H"],
    gate: F[T, "H H"],
) -> F[T, "B T H"]:
    h1 = encode_once(x, stage, up, down, gate)
    h2 = encode_once(h1, stage, up, down, gate)
    return h2


def feedforward_pass(
    x: F[T, "B T H"],
    up: F[T, "H M"],
    down: F[T, "M H"],
) -> F[T, "B T H"]:
    updated = mlp_update(x, up, down)
    return updated


def fused_pass(
    x: F[T, "B T H"],
    stage: EncoderStage,
    up: F[T, "H M"],
    down: F[T, "M H"],
    gate: F[T, "H H"],
) -> F[T, "B T H"]:
    h1 = feedforward_pass(x, up, down)
    h2 = stage(h1, up, down, gate)
    h3 = residual_add(h1, h2)
    return normalize_like(h3)


class MacroModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage_a = EncoderStage()
        self.stage_b = EncoderStage()
        self.head = ProjectionHead()

    def forward(
        self,
        x: F[T, "B T H"],
        up: F[T, "H M"],
        down: F[T, "M H"],
        gate: F[T, "H H"],
        proj: F[T, "H O"],
    ) -> F[T, "B T O"]:
        h0 = normalize_like(x)
        h1 = self.stage_a(h0, up, down, gate)
        h2 = residual_add(h0, h1)
        h3 = self.stage_b(h2, up, down, gate)
        logits = self.head(h3, proj)
        return logits


def run_pipeline():
    B, T, H, M, O = 8, 128, 256, 512, 64
    x: F[T, "B T H"] = torch.zeros(B, T, H)
    up: F[T, "H M"] = torch.zeros(H, M)
    down: F[T, "M H"] = torch.zeros(M, H)
    gate: F[T, "H H"] = torch.zeros(H, H)
    proj: F[T, "H O"] = torch.zeros(H, O)
    model = MacroModel()
    logits = model(x, up, down, gate, proj)
    return logits


def run_pipeline_variant():
    B, T, H, M, O = 4, 64, 128, 256, 32
    x: F[T, "B T H"] = torch.zeros(B, T, H)
    up: F[T, "H M"] = torch.zeros(H, M)
    down: F[T, "M H"] = torch.zeros(M, H)
    gate: F[T, "H H"] = torch.zeros(H, H)
    proj: F[T, "H O"] = torch.zeros(H, O)
    stage = EncoderStage()
    hidden = encode_twice(x, stage, up, down, gate)
    fused = fused_pass(hidden, stage, up, down, gate)
    logits = project_features(fused, proj)
    return logits


def run_pipeline_small():
    B, T, H, M, O = 2, 32, 64, 128, 16
    x: F[T, "B T H"] = torch.zeros(B, T, H)
    up: F[T, "H M"] = torch.zeros(H, M)
    down: F[T, "M H"] = torch.zeros(M, H)
    gate: F[T, "H H"] = torch.zeros(H, H)
    proj: F[T, "H O"] = torch.zeros(H, O)
    stage = EncoderStage()
    hidden = encode_once(x, stage, up, down, gate)
    hidden = residual_add(hidden, stage(hidden, up, down, gate))
    logits = project_features(hidden, proj)
    return logits
