"""
Vocabulary building module.

Single Responsibility: Only handles vocabulary construction.
"""

from __future__ import annotations

from typing import Optional

from src.tokenizer.core.logging import LoggerMixin
from src.tokenizer.models.schemas import SpecialTokens


class VocabularyBuilder(LoggerMixin):
    """
    Builds vocabulary mappings from tokens.
    
    Single Responsibility: Only builds vocabulary mappings.
    
    Example:
        builder = VocabularyBuilder()
        vocab = builder.build(["hello", "world", "hello"])
        # {'hello': 0, 'world': 1, '|<ENDOFTEXT>|': 2, '|<UNK>|': 3}
    """
    
    def __init__(self, special_tokens: Optional[SpecialTokens] = None) -> None:
        """
        Initialize the vocabulary builder.
        
        Args:
            special_tokens: Configuration for special tokens
        """
        self._special_tokens = special_tokens or SpecialTokens()
    
    def build(self, tokens: list[str]) -> dict[str, int]:
        """
        Build vocabulary from tokens.
        
        Args:
            tokens: List of tokens (may contain duplicates)
            
        Returns:
            Dictionary mapping tokens to unique integer IDs
        """
        # Get unique sorted tokens
        unique_tokens = sorted(set(tokens))
        
        self.logger.debug(
            f"Building vocabulary: {len(tokens)} tokens -> {len(unique_tokens)} unique"
        )
        
        # Create mapping
        vocab = {token: idx for idx, token in enumerate(unique_tokens)}
        
        # Add special tokens at the end
        for token in self._special_tokens.as_list():
            if token not in vocab:
                vocab[token] = len(vocab)
        
        self.logger.info(f"Vocabulary built with {len(vocab)} entries")
        return vocab
    
    @staticmethod
    def invert(vocab: dict[str, int]) -> dict[int, str]:
        """
        Create inverse mapping (id -> token).
        
        Args:
            vocab: Token to ID mapping
            
        Returns:
            ID to token mapping
        """
        return {idx: token for token, idx in vocab.items()}
    
    @staticmethod
    def get_vocab_stats(vocab: dict[str, int]) -> dict[str, int]:
        """
        Get statistics about the vocabulary.
        
        Args:
            vocab: Token to ID mapping
            
        Returns:
            Dictionary with vocabulary statistics
        """
        return {
            "size": len(vocab),
            "min_id": min(vocab.values()) if vocab else 0,
            "max_id": max(vocab.values()) if vocab else 0,
        }
    
    def __repr__(self) -> str:
        return f"VocabularyBuilder(special_tokens={self._special_tokens})"
