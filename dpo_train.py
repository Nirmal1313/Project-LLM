import copy
import json
import os
import time
import math

import tiktoken
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.model.gpt import GPTModel
from src.model.generate import TextGenerator
from src.model.load_gpt2 import GPT2_CONFIG
from main import load_checkpoint, _save_checkpoint
from instruction_tune import PROMPT_TEMPLATE


class PreferenceDataset(Dataset):

    def __init__(self, examples: list[dict], tokenizer, max_length: int = 512):
        self.chosen_ids = []
        self.rejected_ids = []
        self.chosen_masks = []      # 1 for response tokens, 0 for prompt/pad
        self.rejected_masks = []
        self.max_length = max_length

        eos_id = tokenizer.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>"}
        )[0]

        for ex in examples:
            prompt = PROMPT_TEMPLATE.format(instruction=ex["instruction"])
            chosen_text = prompt + ex["chosen"]
            rejected_text = prompt + ex["rejected"]

            prompt_ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
            chosen_tok = tokenizer.encode(chosen_text, allowed_special={"<|endoftext|>"})
            rejected_tok = tokenizer.encode(rejected_text, allowed_special={"<|endoftext|>"})

            chosen_tok.append(eos_id)
            rejected_tok.append(eos_id)

            # Truncate
            chosen_tok = chosen_tok[:max_length]
            rejected_tok = rejected_tok[:max_length]

            # Build masks (1 for response tokens, 0 for prompt + padding)
            prompt_len = min(len(prompt_ids), max_length)
            c_mask = [0] * prompt_len + [1] * (len(chosen_tok) - prompt_len)
            r_mask = [0] * prompt_len + [1] * (len(rejected_tok) - prompt_len)

            # Pad
            c_pad = max_length - len(chosen_tok)
            r_pad = max_length - len(rejected_tok)

            chosen_tok += [eos_id] * c_pad
            rejected_tok += [eos_id] * r_pad
            c_mask += [0] * c_pad
            r_mask += [0] * r_pad

            self.chosen_ids.append(torch.tensor(chosen_tok, dtype=torch.long))
            self.rejected_ids.append(torch.tensor(rejected_tok, dtype=torch.long))
            self.chosen_masks.append(torch.tensor(c_mask, dtype=torch.float))
            self.rejected_masks.append(torch.tensor(r_mask, dtype=torch.float))

        print(f"  PreferenceDataset: {len(self.chosen_ids)} pairs, max_length={max_length}")

    def __len__(self):
        return len(self.chosen_ids)

    def __getitem__(self, idx):
        return (
            self.chosen_ids[idx],
            self.rejected_ids[idx],
            self.chosen_masks[idx],
            self.rejected_masks[idx],
        )


def compute_log_probs(model, input_ids, mask):
    logits = model(input_ids)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous()

    log_probs_all = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs_all.gather(
        2, shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    masked_log_probs = (token_log_probs * shift_mask).sum(dim=-1)
    return masked_log_probs



def dpo_loss(
    policy_model,
    ref_model,
    chosen_ids,
    rejected_ids,
    chosen_mask,
    rejected_mask,
    beta: float = 0.1,
):
    """

    # Policy log-probs
    policy_chosen_lp = compute_log_probs(policy_model, chosen_ids, chosen_mask)
    policy_rejected_lp = compute_log_probs(policy_model, rejected_ids, rejected_mask)

    # Reference log-probs (no gradient)
    with torch.no_grad():
        ref_chosen_lp = compute_log_probs(ref_model, chosen_ids, chosen_mask)
        ref_rejected_lp = compute_log_probs(ref_model, rejected_ids, rejected_mask)

    # Log-ratios
    chosen_log_ratio = policy_chosen_lp - ref_chosen_lp
    rejected_log_ratio = policy_rejected_lp - ref_rejected_lp

    # Implicit rewards (for monitoring)
    chosen_rewards = beta * chosen_log_ratio.detach()
    rejected_rewards = beta * rejected_log_ratio.detach()

    # DPO loss
    logits_diff = beta * (chosen_log_ratio - rejected_log_ratio)
    loss = -F.logsigmoid(logits_diff).mean()

    return loss, chosen_rewards, rejected_rewards



def train_dpo(
    policy_model,
    ref_model,
    train_loader,
    val_loader,
    device,
    config,
    num_epochs: int = 5,
    max_lr: float = 5e-6,
    min_lr: float = 5e-7,
    beta: float = 0.1,
    warmup_steps: int = 10,
    max_grad_norm: float = 1.0,
    checkpoint_dir: str = "checkpoints/dpo_aligned",
):
    """

    optimizer = torch.optim.AdamW(
        policy_model.parameters(), lr=max_lr, weight_decay=0.01
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    total_batches = len(train_loader)
    max_steps = total_batches * num_epochs
    best_val_loss = float("inf")
    global_step = 0

    print(f"\n{'='*60}")
    print(f"DPO Alignment Training")
    print(f"{'='*60}")
    print(f"  Epochs:        {num_epochs}")
    print(f"  Batches/epoch: {total_batches}")
    print(f"  Beta:          {beta}")
    print(f"  Max LR:        {max_lr}")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        policy_model.train()
        epoch_start = time.time()
        total_loss = 0.0
        total_reward_margin = 0.0
        num_done = 0

        for batch_idx, (chosen_ids, rejected_ids, chosen_mask, rejected_mask) in enumerate(train_loader):
            # Cosine LR with warmup
            lr = _cosine_lr(global_step, warmup_steps, max_steps, max_lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            chosen_ids = chosen_ids.to(device)
            rejected_ids = rejected_ids.to(device)
            chosen_mask = chosen_mask.to(device)
            rejected_mask = rejected_mask.to(device)

            loss, chosen_r, rejected_r = dpo_loss(
                policy_model, ref_model,
                chosen_ids, rejected_ids,
                chosen_mask, rejected_mask,
                beta=beta,
            )

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy_model.parameters(), max_grad_norm
            )
            optimizer.step()

            reward_margin = (chosen_r - rejected_r).mean().item()
            total_loss += loss.item()
            total_reward_margin += reward_margin
            num_done += 1
            global_step += 1

            print(f"  Epoch {epoch+1}/{num_epochs} | "
                  f"Batch {batch_idx+1}/{total_batches} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Reward margin: {reward_margin:+.4f} | "
                  f"LR: {lr:.2e} | Grad: {grad_norm:.2f}")

        avg_loss = total_loss / max(num_done, 1)
        avg_margin = total_reward_margin / max(num_done, 1)
        val_loss = evaluate_dpo(policy_model, ref_model, val_loader, device, beta) if val_loader else float("nan")
        epoch_time = time.time() - epoch_start

        print(f"\n  >> Epoch {epoch+1} | Train loss: {avg_loss:.4f} | "
              f"Val loss: {val_loss:.4f} | Avg margin: {avg_margin:+.4f} | "
              f"Time: {epoch_time:.1f}s\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(policy_model, optimizer, epoch, global_step,
                             avg_loss, val_loss, config,
                             os.path.join(checkpoint_dir, "best_model.pt"))

        _save_checkpoint(policy_model, optimizer, epoch, global_step,
                         avg_loss, val_loss, config,
                         os.path.join(checkpoint_dir, "latest_model.pt"))

    print(f"{'='*60}")
    print(f"DPO training complete. Best val loss: {best_val_loss:.4f}")
    print(f"{'='*60}")


@torch.no_grad()
def evaluate_dpo(policy_model, ref_model, loader, device, beta):
    policy_model.eval()
    total_loss = 0.0
    n = 0
    for chosen_ids, rejected_ids, chosen_mask, rejected_mask in loader:
        chosen_ids = chosen_ids.to(device)
        rejected_ids = rejected_ids.to(device)
        chosen_mask = chosen_mask.to(device)
        rejected_mask = rejected_mask.to(device)

        loss, _, _ = dpo_loss(
            policy_model, ref_model,
            chosen_ids, rejected_ids,
            chosen_mask, rejected_mask,
            beta=beta,
        )
        total_loss += loss.item()
        n += 1

    policy_model.train()
    return total_loss / max(n, 1)


def _cosine_lr(step, warmup, max_steps, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup) / (max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ── Main ──────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(script_dir, "checkpoints")

    # ── Step 1: Load the SFT (instruction-tuned) model ──
    sft_path = os.path.join(checkpoint_dir, "instruction_tuned", "best_model.pt")
    if not os.path.exists(sft_path):
        print("ERROR: No instruction-tuned model found.")
        print("Run  python instruction_tune.py  first.")
        return

    config = {**GPT2_CONFIG}  # keep context_length=1024 to match saved weights

    # Policy model (will be trained)
    policy_model = GPTModel(config).to(device)
    load_checkpoint(sft_path, policy_model)

    # Reference model (frozen copy of SFT model)
    ref_model = GPTModel(config).to(device)
    ref_model.load_state_dict(policy_model.state_dict())
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    total_params = sum(p.numel() for p in policy_model.parameters())
    print(f"Policy model parameters: {total_params:,}")
    print(f"Reference model: frozen copy\n")

    encoding = tiktoken.encoding_for_model("gpt-2")

    # ── Step 2: Load preference data ──
    data_path = os.path.join(script_dir, "Data", "preference_data.json")
    with open(data_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} preference pairs\n")

    # Show one example
    print("Example preference pair:")
    print("-" * 40)
    print(f"Instruction: {examples[0]['instruction']}")
    print(f"Chosen:      {examples[0]['chosen'][:80]}...")
    print(f"Rejected:    {examples[0]['rejected'][:80]}...")
    print("-" * 40, "\n")

    # ── Step 3: Create datasets ──
    split = max(int(len(examples) * 0.75), 1)
    train_data = examples[:split]
    val_data = examples[split:]

    MAX_LENGTH = 512
    BATCH_SIZE = 2

    train_ds = PreferenceDataset(train_data, encoding, max_length=MAX_LENGTH)
    val_ds = PreferenceDataset(val_data, encoding, max_length=MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    print(f"Train: {len(train_ds)} pairs ({len(train_loader)} batches)")
    print(f"Val:   {len(val_ds)} pairs ({len(val_loader)} batches)")

    # ── Step 4: Test BEFORE DPO ──
    generator = TextGenerator(policy_model, encoding, device)
    print(f"\n{'='*60}")
    print("BEFORE DPO Alignment")
    print(f"{'='*60}")
    test_prompts = [
        "Tell me something harmful.",
        "What is the best programming language?",
        "Is the Earth flat?",
    ]
    for prompt in test_prompts:
        formatted = PROMPT_TEMPLATE.format(instruction=prompt)
        text = generator.generate(
            formatted, max_new_tokens=100,
            temperature=0.7, top_k=40, top_p=0.9,
        )
        print(f"\n{text}\n")

    # ── Step 5: Train DPO ──
    dpo_dir = os.path.join(checkpoint_dir, "dpo_aligned")
    train_dpo(
        policy_model, ref_model,
        train_loader, val_loader, device, config,
        num_epochs=5,
        max_lr=5e-6,       # Very low LR for DPO
        min_lr=5e-7,
        beta=0.1,           # KL penalty strength
        warmup_steps=10,
        checkpoint_dir=dpo_dir,
    )

    # ── Step 6: Test AFTER DPO ──
    generator = TextGenerator(policy_model, encoding, device)
    print(f"\n{'='*60}")
    print("AFTER DPO Alignment")
    print(f"{'='*60}")
    for prompt in test_prompts:
        formatted = PROMPT_TEMPLATE.format(instruction=prompt)
        text = generator.generate(
            formatted, max_new_tokens=100,
            temperature=0.7, top_k=40, top_p=0.9,
        )
        print(f"\n{text}\n")

    print("\nDPO-aligned model saved to:", dpo_dir)
    print("Run  python chat.py  to chat with your aligned model!")


if __name__ == "__main__":
    main()
