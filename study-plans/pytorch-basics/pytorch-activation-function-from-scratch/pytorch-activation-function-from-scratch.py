import torch

def activate(x: torch.Tensor, method: str = "relu") -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as x.
    """
    if  method == "relu":
        return torch.relu(x)
    if  method == "sigmoid":
        return torch.sigmoid(x)
    if  method == "tanh":
        return torch.tanh(x)
    if  method == "leaky_relu":
        return torch.nn.functional.leaky_relu(x, negative_slope=0.01, inplace=False)