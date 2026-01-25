"""
Data models and schemas (DTOs).

These dataclasses represent the data structures used throughout the package.
Using frozen=True where appropriate for immutability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from src.tokenizer.core.constants import Defaults


@dataclass(frozen=True)
class SpecialTokens:
    """
    Configuration for special tokens.
    
    Frozen dataclass for immutable token configuration.
    """
    
    end_of_text: str = Defaults.END_OF_TEXT_TOKEN
    unknown: str = Defaults.UNKNOWN_TOKEN
    
    def as_list(self) -> List[str]:
        """Return all special tokens as a list."""
        return [self.end_of_text, self.unknown]
    
    def __contains__(self, token: str) -> bool:
        """Check if a token is a special token."""
        return token in self.as_list()


@dataclass
class TokenizerResult:
    """
    Result of an encoding operation.
    
    Contains both token IDs and the corresponding tokens for inspection.
    """
    
    token_ids: List[int]
    tokens: List[str]
    
    @property
    def num_tokens(self) -> int:
        """Return the number of tokens."""
        return len(self.token_ids)
    
    def __len__(self) -> int:
        """Support len() on result."""
        return self.num_tokens
    
    def __iter__(self):
        """Iterate over (token, id) pairs."""
        return zip(self.tokens, self.token_ids)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "token_ids": self.token_ids,
            "tokens": self.tokens,
            "num_tokens": self.num_tokens,
        }


@dataclass
class VocabularyInfo:
    """
    Information about a vocabulary.
    
    Used to report vocabulary statistics and samples.
    """
    
    size: int
    special_tokens: List[str]
    sample: List[Tuple[str, int]] = field(default_factory=list)
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"VocabularyInfo(size={self.size}, "
            f"special_tokens={self.special_tokens})"
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "size": self.size,
            "special_tokens": self.special_tokens,
            "sample": [{"token": t, "id": i} for t, i in self.sample],
        }
