# test 1
import torch

class Some(torch.nn.Module):
    def forward(x: F[T, "B X R"], y: F[T, "R S"]) -> F[T, "B X S"]:
        # hovering z should return [F, "B X S"]
        z = x @ y
        return z 


# test 2
import torch

class UserLinear(torch.nn.Module):
    def forward(x: F[T, "B X R"], y: F[T, "R S"]) -> F[T, "B X S"]:
        z = x @ y
        return z


def function():
    linear = UserLinear()
    B, X, R, S, U, W = 3, 9, 27, 81, 243, 729
    x = torch.zeros(B, X, R)
    y = torch.ones(R, S)
    bad_y = torch.ones(U, W)
    output = linear(x, y)
    output2 = linear(x, bad_y)
