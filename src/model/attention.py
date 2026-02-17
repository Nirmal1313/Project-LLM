import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert self.head_dim * n_heads == d_model

        self.query = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.key = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.value = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool(),
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        Q = self.query(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:seq_len, :seq_len], float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out(attn_output)