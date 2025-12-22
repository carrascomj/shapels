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
    z = tensor[:, (D + 1):]

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

