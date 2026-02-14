import torch
import torch.nn as nn
from src.model.attention import CausalSelfAttention


class GPTModel(nn.Module):
    """
    A minimal GPT model: embeddings → causal self-attention → output logits.
    
    This is a simplified single-layer GPT for learning purposes.
    Later you will add: TransformerBlock, FeedForward, LayerNorm, residual connections,
    and stack multiple layers.
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, max_length: int):
        super().__init__()
        self.d_model = d_model

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)

        # Attention layer (single layer for now)
        self.attention = CausalSelfAttention(d_model=d_model, n_heads=n_heads)

        # Output projection: maps from d_model back to vocab_size for next-token prediction
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) tensor of token IDs
        Returns:
            logits: (B, T, vocab_size) — raw scores for each token in vocabulary
        """
        batch_size, seq_len = input_ids.shape

        # Token + positional embeddings
        token_emb = self.token_embedding(input_ids)                          # (B, T, D)
        pos_indices = torch.arange(seq_len, device=input_ids.device)         # (T,)
        pos_emb = self.position_embedding(pos_indices)                       # (T, D)
        x = token_emb + pos_emb                                             # (B, T, D)

        # Causal self-attention
        x = self.attention(x)                                                # (B, T, D)

        # Project to vocabulary size
        logits = self.output_head(x)                                         # (B, T, vocab_size)

        return logits
