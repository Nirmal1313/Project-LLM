"""
Semantic Kernel-based Chat Interface

Enhanced chat interface using Semantic Kernel for orchestration and memory management.

Usage:
    python sk_chat.py                  # loads best available model
    python sk_chat.py --model sft      # force instruction-tuned model
    python sk_chat.py --model dpo      # force DPO-aligned model

Commands inside chat:
    /quit or /exit        — leave the chat
    /reset               — clear conversation history
    /info                — show model information
    /settings            — show current settings
    /temp <value>        — set temperature (e.g. /temp 0.8)
    /topk <value>        — set top-k (e.g. /topk 50)
    /topp <value>        — set top-p (e.g. /topp 0.95)
    /tokens <value>      — set max tokens (e.g. /tokens 200)
    /memory              — show conversation memory summary
    /refine              — refine last response with iteration
    /multi <steps>       — execute multi-step task
"""

import argparse
import sys

from src.sk_integration.orchestrator import SKOrchestrator


def handle_command(cmd: str, orchestrator: SKOrchestrator) -> bool:
    """
    Handle slash commands. Returns True if command was handled.
    """
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()

    if command in ("/quit", "/exit"):
        print("\nGoodbye!")
        sys.exit(0)

    if command == "/reset":
        orchestrator.clear_memory()
        print("  [Conversation reset]")
        return True

    if command == "/info":
        print(f"\n{orchestrator.process_command('info')}\n")
        return True

    if command == "/settings":
        print(f"\n{orchestrator.process_command('settings')}\n")
        return True

    if command == "/memory":
        summary = orchestrator.get_conversation_summary()
        print(f"\n{summary}\n")
        return True

    if command == "/temp" and len(parts) == 2:
        try:
            result = orchestrator.plugins.update_settings("temperature", parts[1])
            print(f"  [{result}]")
        except ValueError:
            print("  [Invalid value. Usage: /temp 0.8]")
        return True

    if command == "/topk" and len(parts) == 2:
        try:
            result = orchestrator.plugins.update_settings("top_k", parts[1])
            print(f"  [{result}]")
        except ValueError:
            print("  [Invalid value. Usage: /topk 50]")
        return True

    if command == "/topp" and len(parts) == 2:
        try:
            result = orchestrator.plugins.update_settings("top_p", parts[1])
            print(f"  [{result}]")
        except ValueError:
            print("  [Invalid value. Usage: /topp 0.9]")
        return True

    if command == "/tokens" and len(parts) == 2:
        try:
            result = orchestrator.plugins.update_settings("max_tokens", parts[1])
            print(f"  [{result}]")
        except ValueError:
            print("  [Invalid value. Usage: /tokens 200]")
        return True

    if command == "/refine":
        history = orchestrator.get_full_history()
        if history:
            last_response = history[-1]["assistant"]
            print("\n  [Refining last response with 2 iterations...]")
            refined = orchestrator.execute_with_refinement(last_response, max_iterations=2)
            print(f"\nAssistant (refined): {refined}\n")
        else:
            print("  [No previous response to refine]")
        return True

    if command.startswith("/"):
        print(f"  [Unknown command: {command}]")
        print("  Commands: /quit /reset /info /settings /temp /topk /topp /tokens /memory /refine")
        return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Semantic Kernel-enhanced GPT-2 chat")
    parser.add_argument(
        "--model", type=str, default="auto",
        choices=["auto", "dpo", "sft", "pretrained"],
        help="Which model checkpoint to load (default: auto = best available)",
    )
    parser.add_argument(
        "--no-memory", action="store_true",
        help="Disable conversation memory",
    )
    args = parser.parse_args()

    # Initialize orchestrator with SK integration
    print("\n[Initializing Semantic Kernel orchestrator...]")
    orchestrator = SKOrchestrator(
        model_type=args.model,
        enable_memory=not args.no_memory
    )

    # Welcome message
    model_info = orchestrator.plugins.get_model_info()
    print(f"\n{'='*60}")
    print(f"  Semantic Kernel Chat Interface")
    print(f"{'='*60}")
    print(model_info)
    print(f"{'='*60}")
    print(f"  Commands: /quit /reset /info /settings /temp /topk /topp /tokens /memory /refine")
    print(f"  Memory: {'Enabled' if orchestrator.memory else 'Disabled'}")
    print(f"{'='*60}\n")

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
            handle_command(user_input, orchestrator)
            continue

        # Process with orchestrator
        print("\n  [Generating response...]")
        response, metadata = orchestrator.execute_task(user_input)

        print(f"\nAssistant: {response}\n")

        # Optionally show metadata
        if False:  # Set to True for debugging
            print(f"Metadata: {metadata}\n")


if __name__ == "__main__":
    main()
