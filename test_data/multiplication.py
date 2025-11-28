# test 1
import jaxtyping
import torch

def proper_multiply_not_produce_diagnostics(x: jaxtyping.Float[torch.Tensor, "B X R"], y: jaxtyping.Float[torch.Tensor, "R S"]) -> jaxtyping.Float[torch.Tensor, "B X S"]:
    z = x @ y 
    return z 

# test 2
import jaxtyping
import torch

def proper_multiply_bad_annotation_should_produce_diagnostics(x: jaxtyping.Float[torch.Tensor, "B X R"], y: jaxtyping.Float[torch.Tensor, "R S"]) -> jaxtyping.Float[torch.Tensor, "B S"]:
    z: jaxtyping.Float[torch.Tensor, "B X R"] = x @ y
    return z 



# test 3
from jaxtyping import Float as F
from torch import Tensor as T

def proper_multiply_should_not_produce_diagnostics_with_import_aliases(x: F[T, "B X R"], y: F[T, "R S"]) -> F[T, "B X S"]:
    z: F[T, "B X S"] = x @ y
    return z



# test 4
from jaxtyping import Float as F
from torch import Tensor as T

def hover_on_inferred_should_work(x: F[T, "B X R"], y: F[T, "R S"]) -> F[T, "B X S"]:
    # hovering z should return [F, "B X S"]
    z = x @ y
    return z 


# test 5
from jaxtyping import Float as F
from torch import Tensor as T


def hovering_with_inference_on_arg_should_return_inferred_shape():
    B, X, R, O = 4, 8, 16, 32
    x: F[T, "B X R"] = torch.Tensor(B, X, R)
    y: F[T, "R O"] = torch.Tensor(R, O)
    # hovering z should return [F, "B X O"]
    z = multiply_child_unnanotated(x, y)
    return z


def multiply_child_unnanotated(x, y):
    mat_mul_result = x @ y
    return mat_mul_result

