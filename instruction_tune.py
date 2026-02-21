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
from src.model.load_gpt2 import GPT2_CONFIG, save_pretrained
from main import load_checkpoint, _save_checkpoint


PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

def format_example(instruction: str, response: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=instruction) + response


class InstructionDataset(Dataset):



    def __init__(self, examples: list[dict], tokenizer, max_length: int = 256):
        self.input_ids = []
        self.target_ids = []
        self.loss_masks = []
        self.max_length = max_length

        eos_id = tokenizer.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>"}
        )[0]

        skipped = 0
        for ex in examples:
            prompt = PROMPT_TEMPLATE.format(instruction=ex["instruction"])
            full_text = format_example(ex["instruction"], ex["response"])

            prompt_ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
            full_ids = tokenizer.encode(full_text, allowed_special={"<|endoftext|>"})
            full_ids.append(eos_id)  # ensure ends with <|endoftext|>

            # Truncate to max_length + 1  (need +1 for target shift)
            if len(full_ids) > max_length + 1:
                full_ids = full_ids[: max_length + 1]

            if len(full_ids) < 3:
                skipped += 1
                continue

            input_seq = full_ids[:-1]
            target_seq = full_ids[1:]

            prompt_len = min(len(prompt_ids), len(input_seq))
            mask = [0] * (prompt_len - 1) + [1] * (len(target_seq) - prompt_len + 1)

            pad_len = max_length - len(input_seq)
            input_seq  += [eos_id] * pad_len
            target_seq += [eos_id] * pad_len
            mask       += [0]      * pad_len

            self.input_ids.append(torch.tensor(input_seq, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_seq, dtype=torch.long))
            self.loss_masks.append(torch.tensor(mask, dtype=torch.float))

        if skipped:
            print(f"  (skipped {skipped} examples that were too short)")
        print(f"  InstructionDataset: {len(self.input_ids)} examples, "
              f"max_length={max_length}")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx], self.loss_masks[idx]


def train_instruction(
    model, train_loader, val_loader, device, config,
    num_epochs: int = 5,
    max_lr: float = 2e-5,
    min_lr: float = 2e-6,
    warmup_steps: int = 20,
    max_grad_norm: float = 1.0,
    checkpoint_dir: str = "checkpoints/instruction_tuned",
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)
    model.train()

    os.makedirs(checkpoint_dir, exist_ok=True)
    total_batches = len(train_loader)
    max_steps = total_batches * num_epochs
    best_val_loss = float("inf")
    global_step = 0

    print(f"\n{'='*60}")
    print(f"Instruction Tuning (SFT)")
    print(f"{'='*60}")
    print(f"  Epochs:        {num_epochs}")
    print(f"  Batches/epoch: {total_batches}")
    print(f"  Total steps:   {max_steps}")
    print(f"  Max LR:        {max_lr}")
    print(f"  Warmup:        {warmup_steps} steps")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        total_loss = 0.0
        num_done = 0

        for batch_idx, (input_ids, target_ids, loss_mask) in enumerate(train_loader):
            # Cosine LR with warmup
            lr = _cosine_lr(global_step, warmup_steps, max_steps, max_lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            loss_mask = loss_mask.to(device)

            logits = model(input_ids)                         # (B, T, V)
            # Flatten for cross_entropy, then apply per-token mask
            loss_per_token = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                reduction="none",
            )                                                  # (B*T,)
            loss_per_token = loss_per_token.view_as(target_ids)  # (B, T)
            # Masked mean
            masked_loss = (loss_per_token * loss_mask).sum() / loss_mask.sum().clamp(min=1)

            optimizer.zero_grad()
            masked_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += masked_loss.item()
            num_done += 1
            global_step += 1

            if batch_idx % 5 == 0 or batch_idx == total_batches - 1:
                print(f"  Epoch {epoch+1}/{num_epochs} | "
                      f"Batch {batch_idx+1}/{total_batches} | "
                      f"Loss: {masked_loss.item():.4f} | "
                      f"LR: {lr:.2e} | Grad: {grad_norm:.2f}")

        avg_loss = total_loss / max(num_done, 1)
        val_loss = evaluate_masked(model, val_loader, device) if val_loader else float("nan")
        epoch_time = time.time() - epoch_start

        print(f"\n  >> Epoch {epoch+1} DONE | Train: {avg_loss:.4f} | "
              f"Val: {val_loss:.4f} | Time: {epoch_time:.1f}s\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, global_step,
                             avg_loss, val_loss, config,
                             os.path.join(checkpoint_dir, "best_model.pt"))

        _save_checkpoint(model, optimizer, epoch, global_step,
                         avg_loss, val_loss, config,
                         os.path.join(checkpoint_dir, "latest_model.pt"))

    print(f"{'='*60}")
    print(f"Instruction tuning complete. Best val loss: {best_val_loss:.4f}")
    print(f"{'='*60}")


@torch.no_grad()
def evaluate_masked(model, loader, device):
    """Validation loss using the same masked approach."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for input_ids, target_ids, loss_mask in loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        loss_mask = loss_mask.to(device)

        logits = model(input_ids)
        loss_per_token = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            reduction="none",
        ).view_as(target_ids)

        total_loss += (loss_per_token * loss_mask).sum().item()
        total_tokens += loss_mask.sum().item()

    model.train()
    return total_loss / max(total_tokens, 1)


def _cosine_lr(step, warmup, max_steps, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup) / (max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ── Main ─────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(script_dir, "checkpoints")

    # ── Step 1: Load pretrained GPT-2 ──
    pretrained_path = os.path.join(checkpoint_dir, "gpt2_pretrained.pt")
    if not os.path.exists(pretrained_path):
        print("ERROR: No pretrained GPT-2 weights found.")
        print("Run  python -m src.model.load_gpt2  first to download GPT-2.")
        return

    config = {**GPT2_CONFIG}  # keep context_length=1024 to match pretrained weights
    model = GPTModel(config).to(device)
    load_checkpoint(pretrained_path, model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"GPT-2 parameters: {total_params:,}\n")

    encoding = tiktoken.encoding_for_model("gpt-2")

    # ── Step 2: Load instruction data ──
    data_path = os.path.join(script_dir, "Data", "instruction_data.json")
    with open(data_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} instruction/response pairs\n")

    # Show one formatted example
    print("Example formatted input:")
    print("-" * 40)
    print(format_example(examples[0]["instruction"], examples[0]["response"]))
    print("-" * 40, "\n")

    # ── Step 3: Create datasets ──
    # 80/20 train/val split
    split = int(len(examples) * 0.8)
    train_data = examples[:split]
    val_data = examples[split:]

    MAX_LENGTH = 512
    BATCH_SIZE = 2      # Small dataset → small batch

    train_ds = InstructionDataset(train_data, encoding, max_length=MAX_LENGTH)
    val_ds = InstructionDataset(val_data, encoding, max_length=MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    print(f"Train: {len(train_ds)} examples ({len(train_loader)} batches)")
    print(f"Val:   {len(val_ds)} examples ({len(val_loader)} batches)")

    # ── Step 4: Test BEFORE instruction tuning ──
    generator = TextGenerator(model, encoding, device)
    print(f"\n{'='*60}")
    print("BEFORE Instruction Tuning")
    print(f"{'='*60}")
    test_prompts = [
        "What is machine learning?",
        "Write a Python function to compute factorial.",
    ]
    for prompt in test_prompts:
        formatted = PROMPT_TEMPLATE.format(instruction=prompt)
        text = generator.generate(
            formatted, max_new_tokens=100,
            temperature=0.7, top_k=40, top_p=0.9,
        )
        print(f"\n{text}\n")

    # ── Step 5: Train ──
    sft_dir = os.path.join(checkpoint_dir, "instruction_tuned")
    train_instruction(
        model, train_loader, val_loader, device, config,
        num_epochs=8,       # More epochs since dataset is tiny
        max_lr=2e-5,
        min_lr=2e-6,
        warmup_steps=20,
        checkpoint_dir=sft_dir,
    )

    # ── Step 6: Test AFTER instruction tuning ──
    generator = TextGenerator(model, encoding, device)
    print(f"\n{'='*60}")
    print("AFTER Instruction Tuning")
    print(f"{'='*60}")
    for prompt in test_prompts:
        formatted = PROMPT_TEMPLATE.format(instruction=prompt)
        text = generator.generate(
            formatted, max_new_tokens=150,
            temperature=0.7, top_k=40, top_p=0.9,
        )
        print(f"\n{text}\n")

    # Additional test prompts
    extra = [
        "Explain what a neural network is.",
        "How do I reverse a string in Python?",
        "What is the capital of France?",
    ]
    for prompt in extra:
        formatted = PROMPT_TEMPLATE.format(instruction=prompt)
        text = generator.generate(
            formatted, max_new_tokens=120,
            temperature=0.7, top_k=40, top_p=0.9,
        )
        print(f"\n{text}\n")

    print("\nInstruction-tuned model saved to:", sft_dir)
    print("Run  python chat.py  to chat with your model!")


if __name__ == "__main__":
    main()
