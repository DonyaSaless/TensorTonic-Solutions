import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.W_q = nn.Parameter(torch.randn(d_model,d_model))
        self.W_k = nn.Parameter(torch.randn(d_model,d_model))
        self.W_v = nn.Parameter(torch.randn(d_model,d_model))
        self.W_o = nn.Parameter(torch.randn(d_model,d_model))

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor of shape (batch, seq_len, d_model).
        """

        Q_prime = (Q @self.W_q).view(Q.shape[0], -1, self.num_heads, self.d_k).transpose(-2,-3)
        
        K_prime = (K @self.W_k).view(Q.shape[0], -1, self.num_heads, self.d_k).transpose(-2,-3)

        score = torch.softmax(Q_prime@K_prime.transpose(-1,-2)/math.sqrt(self.d_k) , dim =-1)
        
        V_prime = (V @self.W_v).view(Q.shape[0], -1, self.num_heads, self.d_k).transpose(-2,-3)
        attention = score@V_prime
        concat = attention.transpose(1,2).reshape(Q.shape[0], -1, self.d_model )
        return concat@self.W_o
        
        
