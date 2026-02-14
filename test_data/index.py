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
