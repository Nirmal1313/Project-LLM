"""
Protocol interfaces for dependency inversion.

These protocols define the contracts that implementations must follow,
enabling loose coupling and easier testing through dependency injection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ITextCleaner(Protocol):
    """
    Protocol for text cleaning operations.
    
    Implementations should normalize and clean text before tokenization.
    """
    
    def clean(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text ready for tokenization
        """
        ...


@runtime_checkable
class ITextSplitter(Protocol):
    """
    Protocol for text splitting operations.
    
    Implementations should split text into individual tokens.
    """
    
    def split(self, text: str) -> list[str]:
        """
        Split text into tokens.
        
        Args:
            text: Input text (may or may not be cleaned)
            
        Returns:
            List of tokens
        """
        ...


@runtime_checkable
class IVocabularyBuilder(Protocol):
    """
    Protocol for vocabulary building operations.
    
    Implementations should create token-to-id mappings from token lists.
    """
    
    def build(self, tokens: list[str]) -> dict[str, int]:
        """
        Build vocabulary from tokens.
        
        Args:
            tokens: List of tokens (may contain duplicates)
            
        Returns:
            Dictionary mapping tokens to unique integer IDs
        """
        ...


@runtime_checkable
class IFileReader(Protocol):
    """
    Protocol for file reading operations.
    
    Implementations should handle file I/O with appropriate encoding.
    """
    
    def read(self, file_path: Path) -> str:
        """
        Read file contents.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File contents as string
        """
        ...


@runtime_checkable
class ITokenizer(Protocol):
    """
    Protocol for tokenizer operations.
    
    Implementations should encode text to IDs and decode IDs back to text.
    """
    
    def encode(self, text: str) -> list[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        ...
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text
        """
        ...
    
    @property
    def vocabulary(self) -> dict[str, int]:
        """Return the vocabulary mapping."""
        ...
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        ...
