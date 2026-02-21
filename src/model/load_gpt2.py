"""
Download GPT-2 (124M) pretrained weights and load into GPTModel.
"""

import os
import torch
import tiktoken

try:
    from transformers import GPT2LMHeadModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from src.model.gpt import GPTModel


# GPT-2 124M config mapped to our model's format
GPT2_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "d_model": 768,
    "n_heads": 12,
    "n_layers": 12,
    "dropout": 0.1,
    "qkv_bias": True,  # GPT-2 uses bias in Q/K/V projections
}


def download_gpt2_weights() -> dict:
    """Download GPT-2 124M state dict via HuggingFace transformers."""
    if not HAS_TRANSFORMERS:
        raise ImportError(
            "transformers package required.\n"
            "Install: pip install transformers"
        )
    print("Downloading GPT-2 124M from HuggingFace (~500MB first time)...")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    state = hf_model.state_dict()
    print(f"Downloaded {len(state)} weight tensors.")
    return state


def map_gpt2_weights(hf_state: dict, n_layers: int = 12, d_model: int = 768) -> dict:
    """
    Map HuggingFace GPT-2 state dict to our GPTModel state dict.

    Key challenge: GPT-2 uses Conv1D which stores weights as (in_features, out_features).
    Our nn.Linear stores weights as (out_features, in_features).
    So we transpose all linear weight matrices.

    Additionally, GPT-2 combines Q/K/V into a single c_attn matrix of shape
    (768, 2304). We split it into three (768, 768) matrices for query, key, value.
    """
    our_state = {}

    # ── Embeddings (no transpose needed for embeddings) ──
    our_state["token_embedding.weight"] = hf_state["transformer.wte.weight"]
    our_state["position_embedding.weight"] = hf_state["transformer.wpe.weight"]

    # ── Final LayerNorm ──
    our_state["final_norm.weight"] = hf_state["transformer.ln_f.weight"]
    our_state["final_norm.bias"] = hf_state["transformer.ln_f.bias"]

    # ── Output head (weight-tied with token embedding in GPT-2) ──
    our_state["output_head.weight"] = hf_state["transformer.wte.weight"].clone()

    # ── Transformer blocks ──
    for i in range(n_layers):
        hf = f"transformer.h.{i}"
        our = f"transformer_blocks.{i}"

        # LayerNorms (same shape, no transpose)
        our_state[f"{our}.norm1.weight"] = hf_state[f"{hf}.ln_1.weight"]
        our_state[f"{our}.norm1.bias"] = hf_state[f"{hf}.ln_1.bias"]
        our_state[f"{our}.norm2.weight"] = hf_state[f"{hf}.ln_2.weight"]
        our_state[f"{our}.norm2.bias"] = hf_state[f"{hf}.ln_2.bias"]

        # ── Attention: split combined QKV ──
        # c_attn is Conv1D: weight shape (768, 2304) -> transpose to (2304, 768)
        # Then split into Q, K, V each (768, 768)
        c_attn_w = hf_state[f"{hf}.attn.c_attn.weight"]    # (768, 2304) Conv1D
        c_attn_b = hf_state[f"{hf}.attn.c_attn.bias"]      # (2304,)

        c_attn_w_t = c_attn_w.t()                           # (2304, 768)
        q_w, k_w, v_w = c_attn_w_t.split(d_model, dim=0)   # Each (768, 768)
        q_b, k_b, v_b = c_attn_b.split(d_model, dim=0)     # Each (768,)

        our_state[f"{our}.att.query.weight"] = q_w
        our_state[f"{our}.att.query.bias"] = q_b
        our_state[f"{our}.att.key.weight"] = k_w
        our_state[f"{our}.att.key.bias"] = k_b
        our_state[f"{our}.att.value.weight"] = v_w
        our_state[f"{our}.att.value.bias"] = v_b

        # ── Attention output projection ──
        # Conv1D -> transpose for nn.Linear
        our_state[f"{our}.att.out.weight"] = hf_state[f"{hf}.attn.c_proj.weight"].t()
        our_state[f"{our}.att.out.bias"] = hf_state[f"{hf}.attn.c_proj.bias"]

        # ── Feed-forward network ──
        # c_fc (expand: 768 -> 3072) and c_proj (contract: 3072 -> 768)
        # Both Conv1D -> transpose
        our_state[f"{our}.ff.layers.0.weight"] = hf_state[f"{hf}.mlp.c_fc.weight"].t()
        our_state[f"{our}.ff.layers.0.bias"] = hf_state[f"{hf}.mlp.c_fc.bias"]
        our_state[f"{our}.ff.layers.2.weight"] = hf_state[f"{hf}.mlp.c_proj.weight"].t()
        our_state[f"{our}.ff.layers.2.bias"] = hf_state[f"{hf}.mlp.c_proj.bias"]

    return our_state


def load_gpt2_into_model(device: torch.device = None) -> tuple[GPTModel, any]:
    """
    Download GPT-2 124M and map weights into our GPTModel.

    Returns:
        (model, tokenizer) — ready for generation or fine-tuning
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Download from HuggingFace
    hf_state = download_gpt2_weights()

    # Create our model with GPT-2 architecture
    print("Creating GPTModel with GPT-2 config...")
    model = GPTModel(GPT2_CONFIG).to(device)

    # Map weights
    print("Mapping GPT-2 weights to our architecture...")
    our_state = map_gpt2_weights(hf_state)

    # Load into model
    missing, unexpected = model.load_state_dict(our_state, strict=False)

    if missing:
        print(f"  WARNING — Missing keys ({len(missing)}):")
        for k in missing[:10]:
            print(f"    {k}")
    if unexpected:
        print(f"  WARNING — Unexpected keys ({len(unexpected)}):")
        for k in unexpected[:10]:
            print(f"    {k}")
    if not missing and not unexpected:
        print("  All weights loaded successfully!")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # GPT-2 tokenizer (50257 vocab)
    encoding = tiktoken.encoding_for_model("gpt-2")
    return model, encoding


def save_pretrained(model, config, path):
    """Save the loaded model for future use (skip download next time)."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, path)
    print(f"Saved to {path}")


# ============================================================
# Run directly to test
# ============================================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load GPT-2 weights into our architecture
    model, encoding = load_gpt2_into_model(device)

    # Save for future use
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "..", "..", "checkpoints", "gpt2_pretrained.pt")
    save_pretrained(model, GPT2_CONFIG, save_path)

    # Quick generation test
    from src.model.generate import TextGenerator
    generator = TextGenerator(model, encoding, device)

    prompts = [
        "To be, or not to be",
        "The meaning of life is",
        "def fibonacci(n):",
        "The capital of France is",
    ]

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        text = generator.generate(prompt, max_new_tokens=80,
                                  temperature=0.8, top_k=40, top_p=0.95)
        print(f"\n{text}")
