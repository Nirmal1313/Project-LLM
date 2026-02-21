"""Main application module."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.tokenizer.core.constants import Defaults
from src.tokenizer.core.exceptions import VocabularyError
from src.tokenizer.core.logging import LoggerMixin
from src.tokenizer.models.schemas import SpecialTokens, VocabularyInfo
from src.tokenizer.protocols.interfaces import (
    IFileReader,
    ITextCleaner,
    ITextSplitter,
    IVocabularyBuilder,
)
from src.tokenizer.processing.cleaner import TextCleaner
from src.tokenizer.processing.splitter import TextSplitter
from src.tokenizer.processing.file_reader import FileReader
from src.tokenizer.vocabulary.builder import VocabularyBuilder
from src.tokenizer.tokenizers.base import BaseTokenizer
from src.tokenizer.tokenizers.factory import TokenizerFactory
from src.tokenizer.tokenizers.with_unknown import TokenizerWithUnknown


class TokenizerApplication(LoggerMixin):
    
    def __init__(
        self,
        file_reader: Optional[IFileReader] = None,
        text_cleaner: Optional[ITextCleaner] = None,
        text_splitter: Optional[ITextSplitter] = None,
        vocab_builder: Optional[IVocabularyBuilder] = None,
        special_tokens: Optional[SpecialTokens] = None,
    ) -> None:
        self._special_tokens = special_tokens or SpecialTokens()
        self._text_cleaner = text_cleaner or TextCleaner()
        self._text_splitter = text_splitter or TextSplitter(self._text_cleaner)
        self._file_reader = file_reader or FileReader()
        self._vocab_builder = vocab_builder or VocabularyBuilder(self._special_tokens)
        
        self._vocabulary: dict[str, int] = {}
        self._tokenizer: Optional[BaseTokenizer] = None
    
    @property
    def is_loaded(self) -> bool:
        """Check if vocabulary has been loaded."""
        return bool(self._vocabulary)
    
    @property
    def vocabulary(self) -> dict[str, int]:
        """Return the vocabulary mapping."""
        return self._vocabulary
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self._vocabulary)
    
    def load_vocabulary_from_file(self, file_path: Path) -> VocabularyInfo:
        """
        Load and build vocabulary from a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            VocabularyInfo with size and sample tokens
        """
        self.logger.info(f"Loading vocabulary from: {file_path}")
        
        # Read file
        text = self._file_reader.read(Path(file_path))
        
        # Split into tokens
        tokens = self._text_splitter.split(text)
        
        # Build vocabulary
        self._vocabulary = self._vocab_builder.build(tokens)
        
        # Create default tokenizer
        self._tokenizer = TokenizerWithUnknown(
            self._vocabulary,
            self._text_splitter,
            self._special_tokens,
        )
        
        self.logger.info(f"Vocabulary loaded: {len(self._vocabulary)} tokens")
        
        return VocabularyInfo(
            size=len(self._vocabulary),
            special_tokens=self._special_tokens.as_list(),
            sample=list(self._vocabulary.items())[-5:],
        )
    
    def load_vocabulary_from_tokens(self, tokens: list[str]) -> VocabularyInfo:
        """
        Build vocabulary from a list of tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            VocabularyInfo with size and sample tokens
        """
        self._vocabulary = self._vocab_builder.build(tokens)
        
        self._tokenizer = TokenizerWithUnknown(
            self._vocabulary,
            self._text_splitter,
            self._special_tokens,
        )
        
        return VocabularyInfo(
            size=len(self._vocabulary),
            special_tokens=self._special_tokens.as_list(),
            sample=list(self._vocabulary.items())[-5:],
        )
    
    def set_vocabulary(self, vocabulary: dict[str, int]) -> None:
        """
        Set vocabulary directly.
        
        Args:
            vocabulary: Token to ID mapping
        """
        self._vocabulary = vocabulary
        self._tokenizer = TokenizerWithUnknown(
            self._vocabulary,
            self._text_splitter,
            self._special_tokens,
        )
    
    def get_tokenizer(
        self,
        tokenizer_type: str = Defaults.TOKENIZER_WITH_UNKNOWN,
    ) -> BaseTokenizer:
        """
        Get a tokenizer instance.
        
        Args:
            tokenizer_type: Type of tokenizer ("simple" or "with_unknown")
            
        Returns:
            Configured tokenizer instance
            
        Raises:
            VocabularyError: If vocabulary not loaded
        """
        if not self._vocabulary:
            raise VocabularyError(
                "Vocabulary not loaded. Call load_vocabulary_from_file first."
            )
        
        return TokenizerFactory.create(
            tokenizer_type,
            self._vocabulary,
            self._text_splitter,
            self._special_tokens,
        )
    
    def __repr__(self) -> str:
        return (
            f"TokenizerApplication("
            f"vocab_size={self.vocab_size}, "
            f"is_loaded={self.is_loaded})"
        )
