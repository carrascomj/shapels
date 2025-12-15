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


# test 6
from jaxtyping import Float as F
from torch import Tensor as T


def hovering_with_inference_on_arg_with_torch_mm():
    B, X, R, O = 4, 8, 16, 32
    x: F[T, "B X R"] = torch.Tensor(B, X, R)
    y: F[T, "R O"] = torch.Tensor(R, O)
    # hovering z should return [F, "B X O"]
    z = multiply_child_unnanotated_torch_mm(x, y)
    return z


def multiply_child_unnanotated_torch_mm(x, y):
    mat_mul_result = torch.mm(x, y)
    return mat_mul_result



# test 7
from torch import mm as whatever
from jaxtyping import Float as F
from torch import Tensor as T


def hovering_with_inference_on_arg_with_mm():
    B, X, R, O = 4, 2, 16, 8
    x: F[T, "B X R"] = torch.Tensor(B, X, R)
    y: F[T, "R O"] = torch.Tensor(R, O)
    # hovering z should return [F, "B X O"]
    z = multiply_child_unnanotated_torch_mm_aliased(torch.relu(x), y)
    return z


def multiply_child_unnanotated_torch_mm_aliased(x, y):
    mat_mul_result = whatever(x, y)
    return mat_mul_result


# test 8
def hovering_with_inference_on_arg_with_mm():
    B, X, R, U, O = 4, 2, 16, 5, 8
    x: F[T, "B X R"] = torch.Tensor(B, X, R)
    y: F[T, "U R O"] = torch.Tensor(U, R, O)
    z: F[T, "B X R"] = torch.Tensor(B, X, R)
    output_wrong = x * y
    output_right = x * z
    return z


# test 9
from jaxtyping import Float as F
from torch import Tensor as T

def not_broadcastable_tensors_should_produce_diagnostics():
    """Example adapted from https://docs.pytorch.org/docs/stable/notes/broadcasting.html."""
    # same shapes are always broadcastable (i.e. the above rules always hold)
    x: F[T, "0"]=torch.empty((0,))
    y: F[T, "A A"]=torch.empty(2,2)
    # x and y are not broadcastable, because x does not have at least 1 dimension
    bad = x * y.relu()  # diagnostic
    # can line up trailing dimensions
    x: F[T, "A B C 1"]=torch.empty(5,3,4,1)
    y: F[T, "B 1 1"]=torch.empty(  3,1,1)
    # x and y are broadcastable.
    # 1st trailing dimension: both have size 1
    # 2nd trailing dimension: y has size 1
    # 3rd trailing dimension: x size == y size
    # 4th trailing dimension: y dimension doesn't exist
    z = x * y.abs() # good

    # but this does not work
    x: F[T, "A B C 1"]=torch.empty(5,2,4,1)
    y: F[T, "Y 1 1"]=torch.empty(  3,1,1)
    # since 2 is not 3
    bad2 = x * y  # diagnostic


# test 10
import torch
from jaxtyping import Float
from torch import Tensor


def transpose_and_multiply(x, y):
    mat_mul_result = x @ y.T
    return mat_mul_result


B, X, R, O = 5, 8, 16, 32
x: Float[Tensor, "B X R"] = torch.Tensor(B, X, R)
y: Float[Tensor, "O R"] = torch.Tensor(O, R)
bad = x @ y
z = transpose_and_multiply(x, y)


# test 11
from jaxtyping import Float as F
from torch import Tensor as T

def proper_multiply_mm_as_method(x: F[T, "B X R"], y: F[T, "R S"]) -> F[T, "B X S"]:
    z: F[T, "B X S"] = x.mm(y)
    return z


# test 12
from jaxtyping import Float as F
from torch import Tensor as T

def single_line_reassignment(x: F[T, "B X R"], y: F[T, "R S"]) -> F[T, "B X S"]:
    x = x @ y
    z = x @ y.T
    return x


# test 13
from jaxtyping import Float as F
from torch import Tensor as T

def not_broadcastable_func_tensors_should_produce_diagnostics():
    """Example adapted from https://docs.pytorch.org/docs/stable/notes/broadcasting.html."""
    # same shapes are always broadcastable (i.e. the above rules always hold)
    x: F[T, "0"]=torch.empty((0,))
    y: F[T, "A A"]=torch.empty(2,2)
    # x and y are not broadcastable, because x does not have at least 1 dimension
    bad = torch.mul(x, y.relu())  # diagnostic
    # can line up trailing dimensions
    x: F[T, "A B C 1"]=torch.empty(5,3,4,1)
    y: F[T, "B 1 1"]=torch.empty(  3,1,1)
    # x and y are broadcastable.
    # 1st trailing dimension: both have size 1
    # 2nd trailing dimension: y has size 1
    # 3rd trailing dimension: x size == y size
    # 4th trailing dimension: y dimension doesn't exist
    z = torch.mul(x, y.abs()) # good

    # but this does not work
    x: F[T, "A B C 1"]=torch.empty(5,2,4,1)
    y: F[T, "Y 1 1"]=torch.empty(  3,1,1)
    # since 2 is not 3
    bad2 = x.mul(y)  # diagnostic


# test 14
from jaxtyping import Bool, Int, Float as F
from torch import Tensor as T

def not_broadcastable_bitwise_should_produce_diagnostics():
    """Example adapted from https://docs.pytorch.org/docs/stable/notes/broadcasting.html."""
    # same shapes are always broadcastable (i.e. the above rules always hold)
    x: F[T, "0"]=torch.empty((0,))
    y: F[T, "A A"]=torch.empty(2,2)
    # x and y are not broadcastable, because x does not have at least 1 dimension
    bad = x & y.relu()  # diagnostic
    # can line up trailing dimensions
    x: Bool[T, "A B C 1"]=torch.empty(5,3,4,1)
    y: Bool[T, "B 1 1"]=torch.empty(  3,1,1)
    # x and y are broadcastable.
    # 1st trailing dimension: both have size 1
    # 2nd trailing dimension: y has size 1
    # 3rd trailing dimension: x size == y size
    # 4th trailing dimension: y dimension doesn't exist
    z = x & y # good

    # but this does not work
    x: F[Int, "A B C 1"]=torch.empty(5,2,4,1)
    y: F[Int, "Y 1 1"]=torch.empty(  3,1,1)
    # since 2 is not 3
    bad2 = x & y  # diagnostic

 
# test 15
import torch
from jaxtyping import Float as F
from jaxtyping import Bool
from torch import Tensor as T

def wrong_dtype_raises_diagnostics_for_bitwise():
    """Example adapted from https://docs.pytorch.org/docs/stable/notes/broadcasting.html."""
    A, B, C = 5, 3, 4
    # same shapes are always broadcastable (i.e. the above rules always hold)
    x = torch.ones(A, B, C, 1, dtype=torch.int32)
    y = torch.ones(B, 1, 1, dtype=torch.int32)
    good = x & y # good: broadcastable shapes, bitwise with bool
    z = torch.empty(B, 1, 1, dtype=torch.float)
    bad = x & z  # diagnostic, bitwise with float


# test 16
from torch import Tensor


def equality_operators_return_bool():
    """Example adapted from https://docs.pytorch.org/docs/stable/notes/broadcasting.html."""
    A, B, C = 5, 3, 4
    # same shapes are always broadcastable (i.e. the above rules always hold)
    x = torch.ones(A, B, C, 1, dtype=torch.float16)
    y = torch.ones(B, 1, 1, dtype=torch.float16)
    good = x.ge(y) # good: broadcastable shapes, bitwise with bool
    return good
