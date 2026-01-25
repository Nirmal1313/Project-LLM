"""
Tokenizer factory module.

Factory Pattern: Encapsulates tokenizer creation logic.
"""

from __future__ import annotations

from typing import Optional, Type

from src.tokenizer.core.constants import Defaults
from src.tokenizer.core.logging import LoggerMixin
from src.tokenizer.models.schemas import SpecialTokens
from src.tokenizer.protocols.interfaces import ITextSplitter
from src.tokenizer.tokenizers.base import BaseTokenizer
from src.tokenizer.tokenizers.with_unknown import TokenizerWithUnknown


class TokenizerFactory(LoggerMixin):
    """
    Factory for creating tokenizers.
    
    Factory Pattern: Encapsulates tokenizer creation logic.
    
    Supports:
        - "with_unknown": TokenizerWithUnknown (replaces unknown tokens with <UNK>)
    
    Example:
        tokenizer = TokenizerFactory.create("with_unknown", vocabulary)
    """
    
    # Registry of tokenizer types
    _registry: dict[str, Type[BaseTokenizer]] = {
        Defaults.TOKENIZER_WITH_UNKNOWN: TokenizerWithUnknown,
    }
    
    @classmethod
    def create(
        cls,
        tokenizer_type: str,
        vocabulary: dict[str, int],
        text_splitter: Optional[ITextSplitter] = None,
        special_tokens: Optional[SpecialTokens] = None,
    ) -> BaseTokenizer:
        """
        Create a tokenizer of the specified type.
        
        Args:
            tokenizer_type: Type of tokenizer ("simple" or "with_unknown")
            vocabulary: Token to ID mapping
            text_splitter: Optional text splitter
            special_tokens: Optional special tokens config
            
        Returns:
            Configured tokenizer instance
            
        Raises:
            ValueError: If tokenizer_type is not recognized
        """
        if tokenizer_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown tokenizer type: '{tokenizer_type}'. "
                f"Available types: {available}"
            )
        
        tokenizer_class = cls._registry[tokenizer_type]
        
        return tokenizer_class(
            vocabulary=vocabulary,
            text_splitter=text_splitter,
            special_tokens=special_tokens,
        )
    
    @classmethod
    def register(cls, name: str, tokenizer_class: Type[BaseTokenizer]) -> None:
        """
        Register a new tokenizer type.
        
        Args:
            name: Name for the tokenizer type
            tokenizer_class: Tokenizer class to register
        """
        cls._registry[name] = tokenizer_class
    
    @classmethod
    def available_types(cls) -> list[str]:
        """Return list of available tokenizer types."""
        return list(cls._registry.keys())
