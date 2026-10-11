import torch

def softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 probability tensor with the same shape as logits.
    """
    max_logit = logits.max(dim = -1, keepdim =True).values
    shifted = logits - max_logit
    exps = shifted.exp()
    return exps  / exps.sum(dim = -1 , keepdim = True)
