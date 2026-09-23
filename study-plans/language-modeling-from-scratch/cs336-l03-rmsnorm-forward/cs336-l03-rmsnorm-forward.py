import torch

def rmsnorm(x: torch.Tensor, g: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Returns the RMS-normalized tensor with the same shape and dtype as x.
    """
    mean_squared =  x.square().mean(keepdim = True, dim = -1)
    normalized = x/torch.sqrt(mean_squared+epsilon)
    return normalized * g
