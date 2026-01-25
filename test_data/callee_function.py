import torch

def user_rand_like(mu: torch.Tensor, log_sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # repametrized sampling: z_sample = z + sigma * eps
    eps = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype)
    return eps
