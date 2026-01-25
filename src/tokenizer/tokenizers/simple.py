"""
Simple tokenizer implementation.

Strategy Pattern: Basic strategy that raises errors on unknown tokens.
"""

from __future__ import annotations

from src.tokenizer.core.exceptions import UnknownTokenError, DecodingError
from src.tokenizer.tokenizers.base import BaseTokenizer


class SimpleTokenizer(BaseTokenizer):
    """
    Basic tokenizer that raises errors on unknown tokens.
    
    Strategy Pattern: One strategy for handling tokens.
    
    Use this when:
        - You know all tokens are in the vocabulary
        - You want to fail fast on unknown tokens
    
    Example:
        tokenizer = SimpleTokenizer(vocabulary)
        ids = tokenizer.encode("Hello world")  # May raise UnknownTokenError
    """
    
    def encode(self, text: str) -> list[int]:
        """
        Encode text, raising error for unknown tokens.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
            
        Raises:
            UnknownTokenError: If a token is not in vocabulary
        """
        tokens = self._splitter.split(text)
        
        ids = []
        for token in tokens:
            if token not in self._vocabulary:
                raise UnknownTokenError(token, vocab_size=self.vocab_size)
            ids.append(self._vocabulary[token])
        
        return ids
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text (space-separated tokens)
            
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
        
        return " ".join(tokens)
