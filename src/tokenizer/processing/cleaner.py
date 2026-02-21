"""Text cleaning module."""

from __future__ import annotations

import unicodedata

from src.tokenizer.core.constants import SmartQuoteMapping
from src.tokenizer.core.logging import LoggerMixin


class TextCleaner(LoggerMixin):

    
    def __init__(self, remove_non_ascii: bool = True) -> None:
        """
        Initialize the text cleaner.
        
        Args:
            remove_non_ascii: If True, remove non-ASCII characters after normalization
        """
        self._remove_non_ascii = remove_non_ascii
        self._replacement_map = SmartQuoteMapping.get_replacement_map()
    
    def clean(self, text: str) -> str:
        if not text:
            return text
        
        # Normalize unicode (NFKD decomposes characters)
        text = unicodedata.normalize('NFKD', text)
        
        # Replace smart quotes and special punctuation
        for old, new in self._replacement_map.items():
            text = text.replace(old, new)
        
        # Remove non-ASCII if configured
        if self._remove_non_ascii:
            text = self._remove_non_ascii_chars(text)
        
        return text
    
    @staticmethod
    def _remove_non_ascii_chars(text: str) -> str:
        """Remove all non-ASCII characters from text."""
        return ''.join(c if ord(c) < 128 else '' for c in text)
    
    def __repr__(self) -> str:
        return f"TextCleaner(remove_non_ascii={self._remove_non_ascii})"
