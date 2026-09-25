import torch

def attention_scores(q: torch.Tensor, k: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Returns scores of shape (batch, heads, query_length, key_length).
    """
    B, S_q, D = q.shape
    B_h, S_k, D_h =k.shape
    assert D % num_heads == 0, "D must be divisible by num heads"
    d_head = D // num_heads

    q = q.reshape(B, S_q, num_heads, d_head).transpose(1,2)
    k = k.reshape(B, S_k, num_heads, d_head).transpose(1,2)

    score = q@k.transpose(-1,-2)

    normalized_scores = score / math.sqrt(d_head)

    return normalized_scores

    
    
