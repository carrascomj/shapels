import torch
from jaxtyping import Float as F
from torch import Tensor as T


def residual_add(x: F[T, "B T H"], y: F[T, "B T H"]) -> F[T, "B T H"]:
    out = x + y
    return out


def project_features(x: F[T, "B T H"], w: F[T, "H O"]) -> F[T, "B T O"]:
    out = x @ w
    return out


def gated_update(x: F[T, "B T H"], gate: F[T, "H H"]) -> F[T, "B T H"]:
    gated = x @ gate
    return gated


def mlp_update(x, up, down):
    """Unannotated, calling site will resolve to inference through here."""
    hidden = project_features(x, up)
    hidden = torch.relu(hidden)
    update = project_features(hidden, down)
    return residual_add(x, update)


def normalize_like(x: F[T, "B T H"]) -> F[T, "B T H"]:
    return torch.relu(x)
