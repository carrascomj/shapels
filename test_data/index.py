"""Tests from https://docs.pytorch.org/cppdocs/notes/tensor_indexing.html."""

# test 1
import torch

def none_is_an_squeeze():
    A, B, C, D = 256, 32, 224, 16
    tensor = torch.Tensor(A, B, C, D)
    z = tensor[None]

# test 2
import torch

def index_with_ellipsis():
    A, B, C, D = 256, 32, 224, 16
    tensor = torch.Tensor(A, B, C, D)
    z = tensor[Ellipsis, ...]

# test 3
import torch

def test_index_with_integers():
    A, B, C, D = 256, 32, 224, 16
    tensor = torch.Tensor(A, B, C, D)
    z = tensor[1, 2]

# test 4
import torch

def boolean_indexing():
    A, B, C, D = 256, 32, 224, 16
    tensor = torch.Tensor(A, B, C, D)
    z = tensor[True, False]

# test 5
import torch

def indexing_with_slices():
    A, B, C, D = 256, 32, 224, 16
    tensor = torch.Tensor(A, B, C, D)
    z = tensor[1::2]

# test 6
import torch

def indexing_by_tensor():
    A, B, C, D = 256, 32, 224, 16
    tensor = torch.Tensor(A, B, C, D)
    z = tensor[torch.tensor([1, 2])].transpose(0, 2)

# test 7
import torch

def multiple_indexing():
    A, B, C, D = 256, 32, 224, 16
    tensor = torch.Tensor(A, B, C, D)
    z = tensor[..., 0, True, 1::2, torch.tensor([1, 2])]


# test 8
import torch

def half_slice_symbolic():
    A, B, C, D = 256, 32, 224, 16
    tensor = torch.Tensor(A, B, C, D)
    z = tensor[:, :D]


# test 9
import torch

def slice_half_symbolic():
    A, B, C, D = 256, 32, 224, 16
    tensor = torch.Tensor(A, B, C, D)
    z1 = tensor[:, (D + 1):]
    z2 = tensor[:, :, (B + 3):]

# test 10
import torch

def half_slice_concrete():
    tensor = torch.Tensor(256, 32, 224, 16)
    z = tensor[:128]


# test 11
import torch

def slice_half_concrete():
    tensor = torch.Tensor(256, 32, 224, 16)
    z = tensor[128:]


# test 12
import torch

def slice_half_concrete_and_wrong():
    tensor = torch.Tensor(256, 32, 224, 16)
    # this should emit a diagnostic
    z = tensor[128:, 32]


# test 13
import torch

def slice_half_concrete_and_wrong_negative():
    tensor = torch.Tensor(256, 32, 224, 16)
    # this should emit a diagnostic
    z = tensor[128:, -33]

# test 14
import torch

def slice_half_concrete_and_negative():
    tensor = torch.Tensor(256, 32, 224, 16)
    z = tensor[128:, -2, 1:-4]

# test 15
import torch

def slice_half_concrete_and_negative():
    tensor = torch.Tensor(256, 32, 224, 16)
    # this should emit, step cannot be negative
    z = tensor[128:, -2, 1:-2:-1]


# test 16
import torch

def slice_half_concrete_and_negative():
    B, L, E = 256, 32, 1536
    padding = torch.randn((B, L, E))
    token_emb = torch.Tensor(B, L, E)
    attn = torch.Tensor(B, L)
    token_valid = ~padding.eq(0).all(dim=-1)
    # this indexing operation yields a shape unknown at runtime
    # at the first dim but is bound by [0, B*L]
    flat_token = token_emb[token_valid]
    flat_attn = attn[token_valid]


# test 17
import torch

def slice_half_concrete_and_negative():
    B, L, E = 256, 32, 1536
    padding = torch.rand((K, L, E))
    token_emb = torch.Tensor(B, L, E)
    attn = torch.Tensor(B, L)
    token_valid = ~padding.eq(0).all(dim=-1)
    # this indexing operation is wrong because K != B
    flat_token = token_emb[token_valid]


# test 18
import torch

def boolean_runtime_index_collapses_arbitrary_prefix_rank():
    B, H, L, E = 8, 12, 32, 64
    x = torch.Tensor(B, H, L, E)
    mask = torch.Tensor(B, H, L).bool()
    z = x[mask]


# test 19
import torch

def boolean_runtime_index_with_single_mask_axis():
    B, E = 128, 1536
    x = torch.Tensor(B, E)
    mask = torch.Tensor(B).bool()
    z = x[mask]
