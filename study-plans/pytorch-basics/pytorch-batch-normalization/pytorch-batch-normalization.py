import torch

def batch_norm(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as X.
    """
    mu = X.mean(dim = 0)
    var = X.var(dim = 0, unbiased = False)
    normalized = (X - mu) / torch.sqrt(var + eps)
    return gamma * normalized + beta
