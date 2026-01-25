"""
Tokenizer with unknown token handling.

Strategy Pattern: Different strategy that replaces unknown tokens with <UNK>.
Open-Closed: Extends base tokenizer behavior without modifying it.
"""

from __future__ import annotations

import re
from typing import Optional

from src.tokenizer.core.constants import Patterns
from src.tokenizer.core.exceptions import DecodingError
from src.tokenizer.models.schemas import SpecialTokens
from src.tokenizer.protocols.interfaces import ITextSplitter
from src.tokenizer.processing.splitter import TextSplitter
from src.tokenizer.tokenizers.base import BaseTokenizer


class TokenizerWithUnknown(BaseTokenizer):
    """
    Tokenizer that handles unknown tokens with <UNK>.
    
    Open-Closed: Extends SimpleTokenizer behavior without modifying it.
    Strategy Pattern: Different strategy for unknown tokens.
    
    Use this when:
        - Text may contain tokens not in vocabulary
        - You want graceful handling of unknown tokens
    
    Example:
        tokenizer = TokenizerWithUnknown(vocabulary)
        ids = tokenizer.encode("Hello xyz")  # xyz -> <UNK>
    """
    
    def __init__(
        self,
        vocabulary: dict[str, int],
        text_splitter: Optional[ITextSplitter] = None,
        special_tokens: Optional[SpecialTokens] = None,
    ) -> None:
        """
        Initialize tokenizer with unknown token handling.
        
        Args:
            vocabulary: Token to ID mapping
            text_splitter: Text splitter instance
            special_tokens: Special tokens configuration
        """
        super().__init__(vocabulary, text_splitter)
        self._special_tokens = special_tokens or SpecialTokens()
        
        # Validate that UNK token exists in vocabulary
        if self._special_tokens.unknown not in vocabulary:
            raise ValueError(
                f"Unknown token '{self._special_tokens.unknown}' not in vocabulary"
            )
        
        self._unk_id = vocabulary[self._special_tokens.unknown]
    
    @property
    def unknown_token(self) -> str:
        """Return the unknown token string."""
        return self._special_tokens.unknown
    
    @property
    def unknown_token_id(self) -> int:
        """Return the unknown token ID."""
        return self._unk_id
    
    def encode(self, text: str) -> list[int]:
        """
        Encode text, replacing unknown tokens with <UNK>.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        # Use special token handling if splitter supports it
        if isinstance(self._splitter, TextSplitter):
            tokens = self._splitter.split_with_special_tokens(
                text, self._special_tokens.as_list()
            )
        else:
            tokens = self._splitter.split(text)
        
        ids = []
        for token in tokens:
            if token in self._vocabulary:
                ids.append(self._vocabulary[token])
            else:
                self.logger.debug(f"Unknown token replaced: '{token}'")
                ids.append(self._unk_id)
        
        return ids
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode token IDs to text with punctuation cleanup.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text
            
        Raises:
            DecodingError: If a token ID is not in vocabulary
        """
        tokens = []
        for token_id in ids:
            if token_id not in self._id_to_token:
                raise DecodingError(
                    f"Unknown token ID: {token_id}",
                    token_ids=ids,
                )
            tokens.append(self._id_to_token[token_id])
        
        text = " ".join(tokens)
        
        # Clean up spacing around special tokens
        text = Patterns.SPECIAL_TOKEN_CLEANUP.sub(r' \1 ', text)
        
        # Clean up spacing around punctuation
        text = Patterns.PUNCTUATION_CLEANUP.sub(r' \1 ', text)
        
        return text.strip()
    
    def encode_with_unknown_count(self, text: str) -> tuple[list[int], int]:
        """
        Encode and return count of unknown tokens.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (token IDs, unknown token count)
        """
        ids = self.encode(text)
        unknown_count = sum(1 for i in ids if i == self._unk_id)
        return ids, unknown_count
