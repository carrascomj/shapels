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



