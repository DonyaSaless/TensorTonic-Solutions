import torch

def attention_scores(q, k, num_heads):
    """
    Returns: tensor of shape (batch, heads, query_length, key_length)
    """
    batch_q, query_length, embed_dim = q.shape
    batch_k, key_length, key_dim = k.shape
    head_dim = embed_dim // num_heads
    
    q = q.reshape(batch_q, query_length, num_heads, head_dim).transpose(1,2)
    k = k.reshape(batch_k, key_length, num_heads, head_dim).transpose(1,2)
    
    return q @k.transpose(-1,-2) / math.sqrt(head_dim)
    
