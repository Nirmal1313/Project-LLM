"""Simple tokenizer implementation."""

from __future__ import annotations

from src.tokenizer.core.exceptions import UnknownTokenError, DecodingError
from src.tokenizer.tokenizers.base import BaseTokenizer


class SimpleTokenizer(BaseTokenizer):
    
    def encode(self, text: str) -> list[int]:
        tokens = self._splitter.split(text)
        
        ids = []
        for token in tokens:
            if token not in self._vocabulary:
                raise UnknownTokenError(token, vocab_size=self.vocab_size)
            ids.append(self._vocabulary[token])
        
        return ids
    
    def decode(self, ids: list[int]) -> str:
        tokens = []
        for token_id in ids:
            if token_id not in self._id_to_token:
                raise DecodingError(
                    f"Unknown token ID: {token_id}",
                    token_ids=ids,
                )
            tokens.append(self._id_to_token[token_id])
        
        return " ".join(tokens)
