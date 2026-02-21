import os
import re
import time
import math
import unicodedata
import tiktoken
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from src.model.gpt import GPTModel, GPT_CONFIGS
from src.model.generate import TextGenerator


def load_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def clean_text(text: str) -> str:
    text = unicodedata.normalize('NFKD', text)

    replacements = {
        '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '-',
        '\u2026': '...',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = ''.join(char if ord(char) < 128 else '' for char in text)
    return text


def split_documents(text: str) -> list[str]:
    play_pattern = r'^([A-Z][A-Z\' ]+[A-Z])$'

    documents = []
    current_doc = []
    lines = text.split('\n')

    for i, line in enumerate(lines):
        if re.match(play_pattern, line.strip()) and len(line.strip()) > 10:
            prev_blank = i > 0 and not lines[i-1].strip()
            next_blank = i < len(lines)-1 and not lines[i+1].strip()

            if prev_blank or next_blank:
                if current_doc:
                    doc_text = '\n'.join(current_doc).strip()
                    if len(doc_text) > 100:
                        documents.append(doc_text)
                current_doc = [line]
                continue

        current_doc.append(line)

    if current_doc:
        doc_text = '\n'.join(current_doc).strip()
        if len(doc_text) > 100:
            documents.append(doc_text)

    return documents


def prepare_data(file_path: str, batch_size: int = 32, max_length: int = 256,
                 stride: int = 256, val_split: float = 0.1):
    """Prepare training and validation dataloaders."""
    raw_text = load_text(file_path)
    text = clean_text(raw_text)
    documents = split_documents(text)
    print(f"Found {len(documents)} documents")

    # Split documents into train / validation
    split_idx = int(len(documents) * (1 - val_split))
    train_docs = documents[:split_idx]
    val_docs = documents[split_idx:]

    train_text = '<|endoftext|>'.join(train_docs)
    val_text = '<|endoftext|>'.join(val_docs)

    encoding = tiktoken.encoding_for_model("gpt-4o")

    train_tokens = encoding.encode(train_text, allowed_special={"<|endoftext|>"})
    val_tokens = encoding.encode(val_text, allowed_special={"<|endoftext|>"})
    print(f"Train: {len(train_docs)} docs, {len(train_tokens):,} tokens")
    print(f"Val:   {len(val_docs)} docs, {len(val_tokens):,} tokens")

    train_loader = create_dataloader(train_text, encoding,
                                     batch_size=batch_size,
                                     max_length=max_length,
                                     stride=stride)
    val_loader = create_dataloader(val_text, encoding,
                                   batch_size=batch_size,
                                   max_length=max_length,
                                   stride=stride)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader, encoding


class GPTDataset(Dataset):

    def __init__(self, txt, tokenizer, max_length, stride) -> None:
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(txt, tokenizer, max_length, stride, batch_size) -> DataLoader:
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=0)
    return dataloader


def train(model, train_loader, val_loader, device, config,
          num_epochs: int = 5, max_lr: float = 3e-4, min_lr: float = 3e-5,
          max_batches: int | None = None, warmup_steps: int = 100,
          max_grad_norm: float = 1.0, checkpoint_dir: str = None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.1)
    model.train()

    total_batches = min(len(train_loader), max_batches) if max_batches else len(train_loader)
    max_steps = total_batches * num_epochs
    training_start = time.time()
    epoch_losses = []
    val_losses = []
    global_step = 0
    best_val_loss = float('inf')

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training Config")
    print(f"{'='*60}")
    print(f"  Epochs:           {num_epochs}")
    print(f"  Batches/epoch:    {total_batches}")
    print(f"  Total steps:      {max_steps}")
    print(f"  Warmup steps:     {warmup_steps}")
    print(f"  Max LR:           {max_lr}")
    print(f"  Min LR:           {min_lr}")
    print(f"  Grad clip:        {max_grad_norm}")
    print(f"  Optimizer:        AdamW (weight_decay=0.1)")
    if max_batches:
        print(f"  (limited to {max_batches} of {len(train_loader)} total batches)")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        total_loss = 0.0
        num_batches_done = 0

        for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
            if max_batches and batch_idx >= max_batches:
                break

            # Cosine LR with linear warmup
            lr = _get_lr(global_step, warmup_steps, max_steps, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            logits = model(input_seq)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_seq.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            total_loss += loss.item()
            num_batches_done += 1
            global_step += 1

            if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                pct = (batch_idx + 1) / total_batches * 100
                bar_len = 30
                filled = int(bar_len * (batch_idx + 1) / total_batches)
                bar = '#' * filled + '-' * (bar_len - filled)
                elapsed = time.time() - epoch_start
                eta = (elapsed / (batch_idx + 1)) * (total_batches - batch_idx - 1)
                print(f"  Epoch {epoch+1}/{num_epochs} |{bar}| {pct:5.1f}% | "
                      f"Batch {batch_idx+1}/{total_batches} | "
                      f"Loss: {loss.item():.4f} | LR: {lr:.2e} | "
                      f"Grad: {grad_norm:.2f} | ETA: {eta:.0f}s")

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / num_batches_done
        epoch_losses.append(avg_loss)

        # Validation loss
        val_loss = _evaluate(model, val_loader, device, max_batches=max_batches)
        val_losses.append(val_loss)

        overfit_flag = ""
        if len(val_losses) > 1 and val_loss > val_losses[-2]:
            overfit_flag = " !! Val loss increasing (possible overfitting)"

        print(f"\n  >> Epoch {epoch+1}/{num_epochs} DONE | "
              f"Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Time: {epoch_time:.1f}s{overfit_flag}\n")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, global_step,
                             avg_loss, val_loss, config,
                             os.path.join(checkpoint_dir, "best_model.pt"))

        # Save latest every epoch
        _save_checkpoint(model, optimizer, epoch, global_step,
                         avg_loss, val_loss, config,
                         os.path.join(checkpoint_dir, "latest_model.pt"))

    total_time = time.time() - training_start
    print(f"{'='*60}")
    print(f"Training complete in {total_time:.1f}s")
    print(f"Train losses: {' -> '.join(f'{l:.4f}' for l in epoch_losses)}")
    print(f"Val   losses: {' -> '.join(f'{l:.4f}' for l in val_losses)}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"{'='*60}")




def _get_lr(step: int, warmup_steps: int, max_steps: int,
            max_lr: float, min_lr: float) -> float:
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def _evaluate(model, val_loader, device, max_batches: int | None = None) -> float:
    """Compute average validation loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (input_seq, target_seq) in enumerate(val_loader):
        if max_batches and batch_idx >= max_batches:
            break
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)

        logits = model(input_seq)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_seq.view(-1)
        )
        total_loss += loss.item()
        num_batches += 1

    model.train()
    return total_loss / num_batches if num_batches > 0 else float('inf')


def _save_checkpoint(model, optimizer, epoch, step, train_loss, val_loss, config, path):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config,
    }, path)
    print(f"  >> Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer=None):
    """Load training checkpoint. Returns the checkpoint dict."""
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"  >> Loaded checkpoint: {path}")
    if 'train_loss' in checkpoint:
        print(f"     Epoch {checkpoint.get('epoch', '?')} | "
              f"Train Loss: {checkpoint.get('train_loss', 0):.4f} | "
              f"Val Loss: {checkpoint.get('val_loss', 0):.4f}")
    return checkpoint


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    BATCH_SIZE = 8
    CONTEXT_LENGTH = 128
    NUM_EPOCHS = 3
    MAX_BATCHES = 50
    MAX_LR = 3e-4
    MIN_LR = 3e-5
    WARMUP_STEPS = 20

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "Data",
                             "The Project The Complete Works of William Shakespeare by William Shakespeare.txt")

    train_loader, val_loader, encoding = prepare_data(
        file_path, batch_size=BATCH_SIZE,
        max_length=CONTEXT_LENGTH, stride=CONTEXT_LENGTH,
        val_split=0.1
    )

    PRESET = "micro"
    GPT_CONFIG = {**GPT_CONFIGS[PRESET], "vocab_size": encoding.n_vocab}
    GPT_CONFIG["context_length"] = CONTEXT_LENGTH

    model = GPTModel(GPT_CONFIG).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Resume from checkpoint if it exists
    checkpoint_dir = os.path.join(script_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "latest_model.pt")

    if os.path.exists(checkpoint_path):
        print(f"\nFound checkpoint — loading...")
        load_checkpoint(checkpoint_path, model)
        print("(Delete checkpoints/ folder to retrain from scratch)\n")
    else:
        train(model, train_loader, val_loader, device, GPT_CONFIG,
              num_epochs=NUM_EPOCHS, max_lr=MAX_LR, min_lr=MIN_LR,
              max_batches=MAX_BATCHES, warmup_steps=WARMUP_STEPS,
              checkpoint_dir=checkpoint_dir)

    # --- Text Generation with all sampling strategies ---
    generator = TextGenerator(model, encoding, device)

    prompt = "To be, or not to be"
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")

    predictions = generator.predict_next_token(prompt, top_k=5)
    print("\nTop-5 next-token predictions:")
    for token, prob in predictions:
        print(f"  {repr(token):>15s}  ->  {prob:.4f}")

    # Greedy
    greedy_text = generator.generate(prompt, max_new_tokens=50, temperature=0.0)
    print(f"\n[Greedy (temp=0.0)]:\n{greedy_text}")

    # Temperature sampling
    temp_text = generator.generate(prompt, max_new_tokens=50, temperature=0.8, top_k=40)
    print(f"\n[Top-k=40, temp=0.8]:\n{temp_text}")

    # Top-p (nucleus) sampling
    topp_text = generator.generate(prompt, max_new_tokens=50, temperature=0.8, top_p=0.9)
    print(f"\n[Top-p=0.9, temp=0.8]:\n{topp_text}")

    # Combined: top-k + top-p + repetition penalty
    combined_text = generator.generate(prompt, max_new_tokens=50,
                                       temperature=0.8, top_k=40, top_p=0.95,
                                       repetition_penalty=1.2)
    print(f"\n[Top-k=40 + Top-p=0.95 + rep_penalty=1.2, temp=0.8]:\n{combined_text}")