"""
Interactive Chat Interface

Talk to your instruction-tuned (and optionally DPO-aligned) GPT-2 model
in a simple terminal chat loop.

Usage:
    python chat.py                  # loads best available model
    python chat.py --model sft      # force instruction-tuned model
    python chat.py --model dpo      # force DPO-aligned model
    python chat.py --model pretrained  # raw GPT-2 (no instruction tuning)

Commands inside chat:
    /quit   or  /exit   — leave the chat
    /reset              — clear conversation history
    /temp <value>       — set temperature  (e.g. /temp 0.8)
    /topk <value>       — set top-k        (e.g. /topk 50)
    /topp <value>       — set top-p        (e.g. /topp 0.95)
    /tokens <value>     — set max tokens   (e.g. /tokens 200)
"""

import argparse
import os
import sys

import tiktoken
import torch

from src.model.gpt import GPTModel
from src.model.generate import TextGenerator
from src.model.load_gpt2 import GPT2_CONFIG
from main import load_checkpoint


# Same format used in instruction_tune.py
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def find_best_model(checkpoint_dir: str) -> tuple[str, str]:
    """
    Auto-detect the best available model.
    Priority: DPO-aligned > instruction-tuned > pretrained GPT-2.
    Returns (path, label).
    """
    candidates = [
        (os.path.join(checkpoint_dir, "dpo_aligned", "best_model.pt"), "DPO-aligned"),
        (os.path.join(checkpoint_dir, "dpo_aligned", "latest_model.pt"), "DPO-aligned (latest)"),
        (os.path.join(checkpoint_dir, "instruction_tuned", "best_model.pt"), "Instruction-tuned (SFT)"),
        (os.path.join(checkpoint_dir, "instruction_tuned", "latest_model.pt"), "Instruction-tuned (latest)"),
        (os.path.join(checkpoint_dir, "gpt2_pretrained.pt"), "Pretrained GPT-2 (no SFT)"),
    ]
    for path, label in candidates:
        if os.path.exists(path):
            return path, label
    return "", "none"


def load_model(model_type: str, checkpoint_dir: str, device: torch.device):
    """Load model based on type selection."""
    if model_type == "dpo":
        paths = [
            os.path.join(checkpoint_dir, "dpo_aligned", "best_model.pt"),
            os.path.join(checkpoint_dir, "dpo_aligned", "latest_model.pt"),
        ]
    elif model_type == "sft":
        paths = [
            os.path.join(checkpoint_dir, "instruction_tuned", "best_model.pt"),
            os.path.join(checkpoint_dir, "instruction_tuned", "latest_model.pt"),
        ]
    elif model_type == "pretrained":
        paths = [
            os.path.join(checkpoint_dir, "gpt2_pretrained.pt"),
        ]
    else:  # auto
        path, label = find_best_model(checkpoint_dir)
        if not path:
            print("No model checkpoint found! Run training first.")
            sys.exit(1)
        config = {**GPT2_CONFIG}
        model = GPTModel(config).to(device)
        load_checkpoint(path, model)
        return model, config, label

    for p in paths:
        if os.path.exists(p):
            config = {**GPT2_CONFIG}
            model = GPTModel(config).to(device)
            load_checkpoint(p, model)
            label = model_type.upper()
            return model, config, label

    print(f"No checkpoint found for model type '{model_type}'.")
    print("Available checkpoints:")
    for name in ["dpo_aligned", "instruction_tuned"]:
        d = os.path.join(checkpoint_dir, name)
        if os.path.isdir(d):
            print(f"  {name}/: {os.listdir(d)}")
    sys.exit(1)


def handle_command(cmd: str, settings: dict) -> bool:
    """
    Handle slash commands. Returns True if command was handled.
    """
    parts = cmd.strip().split()
    command = parts[0].lower()

    if command in ("/quit", "/exit"):
        print("\nGoodbye!")
        sys.exit(0)

    if command == "/reset":
        print("  [Conversation reset]")
        return True

    if command == "/temp" and len(parts) == 2:
        try:
            settings["temperature"] = float(parts[1])
            print(f"  [Temperature set to {settings['temperature']}]")
        except ValueError:
            print("  [Invalid value. Usage: /temp 0.8]")
        return True

    if command == "/topk" and len(parts) == 2:
        try:
            val = int(parts[1])
            settings["top_k"] = val if val > 0 else None
            print(f"  [Top-k set to {settings['top_k']}]")
        except ValueError:
            print("  [Invalid value. Usage: /topk 50]")
        return True

    if command == "/topp" and len(parts) == 2:
        try:
            val = float(parts[1])
            settings["top_p"] = val if 0 < val < 1 else None
            print(f"  [Top-p set to {settings['top_p']}]")
        except ValueError:
            print("  [Invalid value. Usage: /topp 0.9]")
        return True

    if command == "/tokens" and len(parts) == 2:
        try:
            settings["max_tokens"] = int(parts[1])
            print(f"  [Max tokens set to {settings['max_tokens']}]")
        except ValueError:
            print("  [Invalid value. Usage: /tokens 200]")
        return True

    if command.startswith("/"):
        print(f"  [Unknown command: {command}]")
        print("  Commands: /quit /reset /temp /topk /topp /tokens")
        return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Chat with your GPT-2 assistant")
    parser.add_argument(
        "--model", type=str, default="auto",
        choices=["auto", "dpo", "sft", "pretrained"],
        help="Which model checkpoint to load (default: auto = best available)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(script_dir, "checkpoints")

    # Load model
    model, config, label = load_model(args.model, checkpoint_dir, device)
    encoding = tiktoken.encoding_for_model("gpt-2")
    generator = TextGenerator(model, encoding, device)

    total_params = sum(p.numel() for p in model.parameters())

    # Generation settings (adjustable via /commands)
    settings = {
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "max_tokens": 150,
    }

    # Welcome message
    print(f"\n{'='*60}")
    print(f"  GPT-2 Chat Interface")
    print(f"{'='*60}")
    print(f"  Model:       {label}")
    print(f"  Parameters:  {total_params:,}")
    print(f"  Device:      {device}")
    print(f"  Context:     {config['context_length']} tokens")
    print(f"{'='*60}")
    print(f"  Commands: /quit /reset /temp /topk /topp /tokens")
    print(f"  Current:  temp={settings['temperature']}, "
          f"top_k={settings['top_k']}, top_p={settings['top_p']}")
    print(f"{'='*60}\n")

    print("NOTE: This is a 124M parameter model for educational purposes.")
    print("Responses will be limited compared to larger models.\n")

    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            handle_command(user_input, settings)
            continue

        # Format as instruction prompt
        formatted_prompt = PROMPT_TEMPLATE.format(instruction=user_input)

        # Generate response
        full_output = generator.generate(
            formatted_prompt,
            max_new_tokens=settings["max_tokens"],
            temperature=settings["temperature"],
            top_k=settings["top_k"],
            top_p=settings["top_p"],
            repetition_penalty=1.2,
        )

        # Extract just the response part (after "### Response:\n")
        response_marker = "### Response:\n"
        if response_marker in full_output:
            response = full_output.split(response_marker, 1)[1]
        else:
            response = full_output[len(formatted_prompt):]

        # Clean up: remove trailing special tokens and extra whitespace
        response = response.replace("<|endoftext|>", "").strip()

        # If the model generated another "### Instruction:" block, cut it off
        if "### Instruction:" in response:
            response = response.split("### Instruction:")[0].strip()

        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    main()
