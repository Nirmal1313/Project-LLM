"""
Quick test script for Semantic Kernel integration.
"""

import sys

print("Testing SK Integration Imports...")
print("-" * 60)

try:
    from src.sk_integration.memory_manager import ConversationMemory
    print("✓ ConversationMemory imported")
except Exception as e:
    print(f"✗ ConversationMemory failed: {e}")
    sys.exit(1)

try:
    from src.sk_integration.plugins import ChatPlugins
    print("✓ ChatPlugins imported")
except Exception as e:
    print(f"✗ ChatPlugins failed: {e}")
    sys.exit(1)

try:
    from src.sk_integration.orchestrator import SKOrchestrator
    print("✓ SKOrchestrator imported")
except Exception as e:
    print(f"✗ SKOrchestrator failed: {e}")
    sys.exit(1)

print("-" * 60)
print("\n✓ All SK modules imported successfully!")
print("\nYou can now run:")
print("  python sk_chat.py              # Use auto-detected model")
print("  python sk_chat.py --model sft  # Use instruction-tuned model")
