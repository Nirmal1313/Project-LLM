"""Text splitting module."""

from __future__ import annotations

from typing import Optional

from src.tokenizer.core.constants import Patterns
from src.tokenizer.core.logging import LoggerMixin
from src.tokenizer.protocols.interfaces import ITextCleaner
from src.tokenizer.processing.cleaner import TextCleaner


class TextSplitter(LoggerMixin):
    
    def __init__(self, text_cleaner: Optional[ITextCleaner] = None) -> None:
        """
        Initialize the text splitter.
        
        Args:
            text_cleaner: Optional text cleaner (Dependency Injection)
        """
        self._cleaner = text_cleaner or TextCleaner()
        self._pattern = Patterns.SPLIT_PATTERN
    
    def split(self, text: str, clean_first: bool = True) -> list[str]:
        if clean_first:
            text = self._cleaner.clean(text)
        
        tokens = self._pattern.split(text)
        return [t.strip() for t in tokens if t.strip()]
    
    def split_with_special_tokens(
        self,
        text: str,
        special_tokens: list[str],
        clean_first: bool = True,
    ) -> list[str]:
        placeholders: dict[str, str] = {}
        for i, token in enumerate(special_tokens):
            placeholder = Patterns.PLACEHOLDER_TEMPLATE.format(index=i)
            placeholders[placeholder] = token
            text = text.replace(token, f' {placeholder} ')
        
        # Split
        tokens = self.split(text, clean_first=clean_first)
        
        # Restore special tokens
        return [placeholders.get(t, t) for t in tokens]
    
    def __repr__(self) -> str:
        return f"TextSplitter(cleaner={self._cleaner!r})"
