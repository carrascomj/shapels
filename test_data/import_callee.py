# test 1
import torch
from callee_function import user_rand_like
                                                
def test_import_is_correct(mu: torch.Tensor, log_sigma: torch.Tensor):
    eps = user_rand_like(mu, log_sigma)
    
