"""
Base tokenizer module.

Open-Closed Principle: Extend by subclassing, not modifying.
Liskov Substitution: All subclasses can replace this base.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.tokenizer.core.logging import LoggerMixin
from src.tokenizer.models.schemas import TokenizerResult
from src.tokenizer.protocols.interfaces import ITextSplitter
from src.tokenizer.processing.splitter import TextSplitter
from src.tokenizer.vocabulary.builder import VocabularyBuilder


class BaseTokenizer(ABC, LoggerMixin):
    """
    Abstract base tokenizer.
    
    Open-Closed: Extend by subclassing, not modifying.
    Liskov Substitution: All subclasses can replace this base.
    Dependency Inversion: Depends on ITextSplitter abstraction.
    
    Subclasses must implement:
        - encode(text: str) -> list[int]
        - decode(ids: list[int]) -> str
    """
    
    def __init__(
        self,
        vocabulary: dict[str, int],
        text_splitter: Optional[ITextSplitter] = None,
    ) -> None:
        """
        Initialize the tokenizer.
        
        Args:
            vocabulary: Token to ID mapping
            text_splitter: Text splitter instance (Dependency Injection)
        """
        self._vocabulary = vocabulary
        self._id_to_token = VocabularyBuilder.invert(vocabulary)
        self._splitter = text_splitter or TextSplitter()
    
    @property
    def vocabulary(self) -> dict[str, int]:
        """Return the vocabulary mapping."""
        return self._vocabulary
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self._vocabulary)
    
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        pass
    
    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text
        """
        pass
    
    def encode_with_tokens(self, text: str) -> TokenizerResult:
        """
        Encode and return both IDs and tokens.
        
        Args:
            text: Input text
            
        Returns:
            TokenizerResult with token_ids and tokens
        """
        ids = self.encode(text)
        tokens = [self._id_to_token[i] for i in ids]
        return TokenizerResult(token_ids=ids, tokens=tokens)
    
    def get_token(self, token_id: int) -> Optional[str]:
        """
        Get token string for a given ID.
        
        Args:
            token_id: Token ID
            
        Returns:
            Token string or None if not found
        """
        return self._id_to_token.get(token_id)
    
    def get_id(self, token: str) -> Optional[int]:
        """
        Get ID for a given token.
        
        Args:
            token: Token string
            
        Returns:
            Token ID or None if not found
        """
        return self._vocabulary.get(token)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(vocab_size={self.vocab_size})"
