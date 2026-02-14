import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)

        # Linear projections
        Q = self.query(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads and pass through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out(attn_output)

        return output
    
class CausalSelfAttention(SelfAttention):
    def __init__(self, d_model, n_heads):
        super(CausalSelfAttention, self).__init__(d_model, n_heads)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)  # Causal mask
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads and pass through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out(attn_output)

        return output
    
class MultiHeadAttention(SelfAttention):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__(d_model, n_heads)