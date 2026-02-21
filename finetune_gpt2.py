import os
import tiktoken
import torch

from main import (
    load_text, clean_text, split_documents, create_dataloader,
    train, _evaluate, load_checkpoint
)
from src.model.gpt import GPTModel
from src.model.generate import TextGenerator
from src.model.load_gpt2 import (
    load_gpt2_into_model, GPT2_CONFIG, save_pretrained
)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(script_dir, "checkpoints")
    pretrained_path = os.path.join(checkpoint_dir, "gpt2_pretrained.pt")

    model = GPTModel(GPT2_CONFIG).to(device)
        print("Loading cached GPT-2 weights...")
        model = GPTModel(GPT2_CONFIG).to(device)
        load_checkpoint(pretrained_path, model)
        encoding = tiktoken.encoding_for_model("gpt-2")
    else:
        print("Downloading GPT-2 weights (first time only)...")
        model, encoding = load_gpt2_into_model(device)
        save_pretrained(model, GPT2_CONFIG, pretrained_path)

    print(f"GPT-2 parameters: {total_params:,}\n")
    generator = TextGenerator(model, encoding, device)
    print("=" * 60)
    print("BEFORE Fine-tuning")
    print("=" * 60)
    for prompt in ["To be, or not to be", "Friends, Romans, countrymen"]:
        text = generator.generate(prompt, max_new_tokens=60,
                                  temperature=0.8, top_k=40, top_p=0.95)
        print(f"\nPrompt: {prompt}\n{text}\n")

    file_path = os.path.join(script_dir, "Data",
                             "The Project The Complete Works of William Shakespeare by William Shakespeare.txt")

    raw_text = load_text(file_path)
    text_clean = clean_text(raw_text)
    documents = split_documents(text_clean)
    print(f"Found {len(documents)} documents")

    split_idx = int(len(documents) * 0.9)
    train_text = '<|endoftext|>'.join(documents[:split_idx])
    val_text = '<|endoftext|>'.join(documents[split_idx:])

    # Fine-tuning hyperparameters
    BATCH_SIZE = 4
    WARMUP_STEPS = 50

    train_loader = create_dataloader(train_text, encoding,
                                     batch_size=BATCH_SIZE,
                                     max_length=CONTEXT_LENGTH,
                                     stride=CONTEXT_LENGTH)
    val_loader = create_dataloader(val_text, encoding,
                                   batch_size=BATCH_SIZE,
                                   max_length=CONTEXT_LENGTH,
                                   stride=CONTEXT_LENGTH)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")

    config = {**GPT2_CONFIG, "context_length": CONTEXT_LENGTH}
    ft_dir = os.path.join(checkpoint_dir, "gpt2_shakespeare")

    train(model, train_loader, val_loader, device, config,
          num_epochs=NUM_EPOCHS, max_lr=MAX_LR, min_lr=MIN_LR,
          max_batches=MAX_BATCHES, warmup_steps=WARMUP_STEPS,
          checkpoint_dir=ft_dir)

    # ── Step 5: Test AFTER fine-tuning ──
    generator = TextGenerator(model, encoding, device)
    print("\n" + "=" * 60)
    print("AFTER Fine-tuning on Shakespeare")
    print("=" * 60)

    prompts = [
        "To be, or not to be",
        "Friends, Romans, countrymen",
        "All the world's a stage",
        "Now is the winter of our discontent",
    ]
    for prompt in prompts:
        text = generator.generate(prompt, max_new_tokens=80,
                                  temperature=0.8, top_k=40, top_p=0.95,
                                  repetition_penalty=1.2)
        print(f"\nPrompt: {prompt}\n{text}\n")


if __name__ == "__main__":
    main()
