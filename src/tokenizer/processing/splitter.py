"""
Text splitting module.

Single Responsibility: Only handles splitting text into tokens.
"""

from __future__ import annotations

from typing import Optional

from src.tokenizer.core.constants import Patterns
from src.tokenizer.core.logging import LoggerMixin
from src.tokenizer.protocols.interfaces import ITextCleaner
from src.tokenizer.processing.cleaner import TextCleaner


class TextSplitter(LoggerMixin):
    """
    Splits text into tokens using regex.
    
    Single Responsibility: Only handles text splitting.
    Dependency Injection: TextCleaner is injected.
    
    Example:
        splitter = TextSplitter()
        tokens = splitter.split("Hello, world!")  # ['Hello', ',', 'world', '!']
    """
    
    def __init__(self, text_cleaner: Optional[ITextCleaner] = None) -> None:
        """
        Initialize the text splitter.
        
        Args:
            text_cleaner: Optional text cleaner (Dependency Injection)
        """
        self._cleaner = text_cleaner or TextCleaner()
        self._pattern = Patterns.SPLIT_PATTERN
    
    def split(self, text: str, clean_first: bool = True) -> list[str]:
        """
        Split text into tokens.
        
        Args:
            text: Input text
            clean_first: If True, clean text before splitting
            
        Returns:
            List of tokens (non-empty, stripped)
        """
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
        """
        Split text while preserving special tokens.
        
        Special tokens are replaced with placeholders before splitting,
        then restored after to prevent them from being split apart.
        
        Args:
            text: Input text
            special_tokens: List of special tokens to preserve
            clean_first: If True, clean text before splitting
            
        Returns:
            List of tokens with special tokens preserved
        """
        # Replace special tokens with placeholders
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
