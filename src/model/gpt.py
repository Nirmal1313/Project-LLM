import math

import torch
import torch.nn as nn

from src.model.transformerBlock import TransformerBlock


GPT_CONFIGS = {
    "gpt3-small":  dict(d_model=768,  n_heads=12, n_layers=12, context_length=2048, dropout=0.1, qkv_bias=False),
    "gpt3-medium": dict(d_model=1024, n_heads=16, n_layers=24, context_length=2048, dropout=0.1, qkv_bias=False),
    "gpt3-large":  dict(d_model=1536, n_heads=16, n_layers=24, context_length=2048, dropout=0.1, qkv_bias=False),
    "gpt3-xl":     dict(d_model=2048, n_heads=16, n_layers=24, context_length=2048, dropout=0.1, qkv_bias=False),
    "tiny":        dict(d_model=256,  n_heads=4,  n_layers=4,  context_length=256,  dropout=0.1, qkv_bias=False),
}


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["d_model"])
        self.drop_emb = nn.Dropout(cfg["dropout"])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = nn.LayerNorm(cfg["d_model"])
        self.output_head = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)

        self._init_weights()

    def _init_weights(self):
        init_std = 0.02
        residual_std = init_std / math.sqrt(2.0 * self.cfg["n_layers"])

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)

        for block in self.transformer_blocks:
            nn.init.normal_(block.att.out.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ff.layers[2].weight, mean=0.0, std=residual_std)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape

        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(torch.arange(seq_len, device=input_ids.device))
        x = self.drop_emb(tok_emb + pos_emb)

        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.output_head(x)

        return logits

