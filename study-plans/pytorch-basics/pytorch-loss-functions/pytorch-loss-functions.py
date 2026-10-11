import torch

def compute_loss(pred: torch.Tensor, target: torch.Tensor, method: str, delta: float = 1.0) -> float:
    """
    Returns the mean loss as a Python float.
    """
    if method == "mse":
        return  ((pred - target)**2).mean().item()

    if method == "cross_entropy":
        shifted = pred - pred.max(dim = -1, keepdim = True).values
        log_sum_exp = shifted.exp().sum(dim = -1, keepdim = True).log()
        log_probs = shifted - log_sum_exp
        return - log_probs[torch.arange(shifted.shape[0]) , target.long()].mean().item()

    difference = (pred - target).abs()
    return  torch.where(difference <= delta, 0.5 * difference ** 2,
                         delta * (difference - 0.5 * delta)).mean().item()
        
        
        