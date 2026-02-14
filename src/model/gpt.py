import torch
import torch.nn as nn
from src.model.attention import CausalSelfAttention


class GPTModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, max_length: int):
        super().__init__()
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.attention = CausalSelfAttention(d_model=d_model, n_heads=n_heads)
        # no bias — weight tying with embedding later
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        token_emb = self.token_embedding(input_ids)
        pos_indices = torch.arange(seq_len, device=input_ids.device)
        pos_emb = self.position_embedding(pos_indices)
        x = token_emb + pos_emb

        x = self.attention(x)
        logits = self.output_head(x)

        return logits
