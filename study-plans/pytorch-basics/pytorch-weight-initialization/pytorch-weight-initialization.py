import torch

def initialize_weights(fan_in: int, fan_out: int, method: str) -> torch.Tensor:
    """
    Returns a float32 weight tensor with shape (fan_out, fan_in).
    """
    if method == "xavier_uniform":
        lim =  math.sqrt(6/(fan_in + fan_out))
        return torch.empty(fan_out, fan_in, dtype = torch.float32).uniform_(-lim,lim)
        
    if method == "xavier_normal":
        return torch.empty(fan_out, fan_in, dtype = torch.float32).normal_(0,math.sqrt(2/(fan_in+fan_out)))

    if method == "he_uniform":
        return torch.empty(fan_out, fan_in, dtype = torch.float32).uniform_(-math.sqrt(6/fan_in), math.sqrt(6/fan_in))

    if method == "he_normal":
        return torch.empty(fan_out, fan_in, dtype = torch.float32).normal_(0, math.sqrt(2/fan_in))