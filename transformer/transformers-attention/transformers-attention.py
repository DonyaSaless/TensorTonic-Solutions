import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    return torch.softmax((Q @K.transpose(-1,-2))/ math.sqrt(K.shape[-1]), dim = -1)@V