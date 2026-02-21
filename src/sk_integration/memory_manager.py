"""Conversation memory management for SK integration."""

from datetime import datetime
from typing import Optional


class ConversationMemory:
    """Manages conversation history and context for multi-turn conversations."""

    def __init__(self, max_history: int = 10, context_window: int = 4096):
        """
        Initialize conversation memory.
        
        Args:
            max_history: Maximum number of exchanges to keep
            context_window: Maximum token count for memory context
        """
        self.max_history = max_history
        self.context_window = context_window
        self.history = []
        self.metadata = {
            "session_start": datetime.now().isoformat(),
            "total_messages": 0,
        }

    def add_exchange(self, user_input: str, assistant_response: str, metadata: Optional[dict] = None):
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "assistant": assistant_response,
            "metadata": metadata or {},
        }
        self.history.append(exchange)
        self.metadata["total_messages"] += 1

        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_context(self) -> str:
        """Get formatted conversation context for prompt enrichment."""
        if not self.history:
            return ""

        context_lines = ["## Conversation History:"]
        for exchange in self.history[-3:]:  # Use last 3 exchanges
            context_lines.append(f"User: {exchange['user']}")
            context_lines.append(f"Assistant: {exchange['assistant']}")

        return "\n".join(context_lines)

    def get_full_history(self) -> list:
        """Get full conversation history."""
        return self.history.copy()

    def clear(self):
        """Clear conversation history."""
        self.history = []
        self.metadata["total_messages"] = 0

    def get_summary(self) -> dict:
        """Get memory summary statistics."""
        return {
            "exchanges": len(self.history),
            "total_messages": self.metadata["total_messages"],
            "session_start": self.metadata["session_start"],
            "recent_topics": self._extract_topics(),
        }

    def _extract_topics(self) -> list[str]:
        """Extract topics from recent exchanges."""
        topics = []
        for exchange in self.history[-3:]:
            # Simple topic extraction: first few words of user input
            words = exchange["user"].split()[:3]
            topics.append(" ".join(words))
        return topics
