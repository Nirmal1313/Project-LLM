import os
import re
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


def prepare_data(file_path: str, batch_size: int = 32, max_length: int = 256, stride: int = 256):
    raw_text = load_text(file_path)
    text = clean_text(raw_text)
    documents = split_documents(text)
    print(f"Found {len(documents)} documents")

    text_with_boundaries = '<|endoftext|>'.join(documents)

    encoding = tiktoken.encoding_for_model("gpt-4o")
    tokens = encoding.encode(text_with_boundaries, allowed_special={"<|endoftext|>"})
    print(f"Total tokens: {len(tokens)}")

    dataloader = create_dataloader(text_with_boundaries, encoding,
                                   batch_size=batch_size,
                                   max_length=max_length,
                                   stride=stride)
    print(f"Batches per epoch: {len(dataloader)}")
    return dataloader, encoding


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
                            drop_last=True, num_workers=4)
    return dataloader


def train(model, dataloader, device, num_epochs: int = 5, lr: float = 3e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            logits = model(input_seq)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_seq.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} — Avg Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 32
    CONTEXT_LENGTH = 256
    NUM_EPOCHS = 5
    LR = 3e-4

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "Data",
                             "The Project The Complete Works of William Shakespeare by William Shakespeare.txt")

    dataloader, encoding = prepare_data(file_path, batch_size=BATCH_SIZE,
                                        max_length=CONTEXT_LENGTH, stride=CONTEXT_LENGTH)

    PRESET = "tiny"
    GPT_CONFIG = {**GPT_CONFIGS[PRESET], "vocab_size": encoding.n_vocab}
    GPT_CONFIG["context_length"] = CONTEXT_LENGTH

    model = GPTModel(GPT_CONFIG).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    train(model, dataloader, device, num_epochs=NUM_EPOCHS, lr=LR)

    # --- Next-token prediction / text generation ---
    generator = TextGenerator(model, encoding, device)

    prompt = "To be, or not to be"
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")

    # Show top-5 predicted next tokens
    predictions = generator.predict_next_token(prompt, top_k=5)
    print("\nTop-5 next-token predictions:")
    for token, prob in predictions:
        print(f"  {repr(token):>15s}  →  {prob:.4f}")

    # Generate continuation (greedy)
    greedy_text = generator.generate(prompt, max_new_tokens=50, temperature=0.0)
    print(f"\nGreedy:\n{greedy_text}")

    # Generate continuation (sampled with temperature)
    sampled_text = generator.generate(prompt, max_new_tokens=50, temperature=0.8, top_k=40)
    print(f"\nSampled (temp=0.8, top_k=40):\n{sampled_text}")