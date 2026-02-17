import torch
import torch.nn as nn

from src.model.attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["d_model"], 4 * cfg["d_model"]),
            nn.GELU(),
            nn.Linear(4 * cfg["d_model"], cfg["d_model"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            context_length=cfg["context_length"],
            dropout=cfg["dropout"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["d_model"])
        self.norm2 = nn.LayerNorm(cfg["d_model"])
        self.drop_shortcut = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x