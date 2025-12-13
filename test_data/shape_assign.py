"""Tested by src/tests/test_shape_assign.rs."""

# test 1
import torch

def infer_from_shape_parts(x: torch.Tensor, y: torch.Tensor):
    Batch, Channels, Height, Width = x.shape
    Channels, NumClasses = y.shape
    # ouptut is Batch, Width, Height, NumClasses
    output = x.transpose(1, 3) @ y

# test 2
import torch
from jaxtyping import Float

def bad_inference_dims_dont_match(x: Float[torch.Tensor, "Batch Alpha Epsilon"], y: torch.Tensor):
    Batch, Channels, Height, Width = x.shape


# test 3
import torch
from jaxtyping import Float

def inference_shape_checks(x: torch.Tensor, y: Float[torch.Tensor, "Batch C H W"],):
    Batch, Channels, Height, Width = y.shape
    after_renamed = y

# test 4
import torch
from jaxtyping import Float

def ann_assignment_shape_checks(x: torch.Tensor, y: Float[torch.Tensor, "Batch C H W"],):
    b: Float[torch.Tensor, "Batch Channels Height Width"] = y
    after_renamed = b


# test 5
import torch
from jaxtyping import Float

def ann_assignment_missalignment_diagnoses(x: torch.Tensor, y: Float[torch.Tensor, "Batch C H W"],):
    # this is missing the channels dimension
    b: Float[torch.Tensor, "B H W"] = y
    after_renamed = b


# test 6
from torch import ones

def ann_creation_size_list_is_the_right_shape():
    Batch, Channels, Height, Width = 32, 3, 224, 224
    x = ones([Batch, Channels, Height, Width], dtype="bool")

# test 7
import torch

def ann_creation_size_is_the_right_shape():
    Batch, Features = 32, 3
    z = torch.zeros(Batch, Features, dtype=torch.bfloat16)
    eps = torch.randn(z.shape, device=z.device, dtype=z.dtype)


# test 8
from jaxtyping import Float
import torch

def randperm_from_namevar():
    x: Float[torch.Tensor, "B Features"]
    n_instances = int(x.shape[0] * 0.5)
    # [n_instances]: Long
    idx = torch.randperm(n_instances)[: int(n_instances)]
    # [n_instances, Features]: Float
    out = x[idx] 


# test 9
from jaxtyping import Float
import torch

def randperm_from_namevar():
    x: Float[torch.Tensor, "B Features"]
    n_instances = int(x.shape[0] * 0.5)
    # [n_instances]: Long
    some_range = torch.range(0, 10, 0.1)
    some_arange = torch.arange(0, 10, 0.1)
    some_range_n = torch.range(0, 10, n_instances)
    some_arange_n = torch.arange(0, 10, n_instances)
    lin_n = torch.linspace(steps=n_instances)

