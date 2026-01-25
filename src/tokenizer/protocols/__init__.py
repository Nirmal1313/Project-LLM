"""Protocol interfaces for dependency inversion."""

from src.tokenizer.protocols.interfaces import (
    ITextCleaner,
    ITextSplitter,
    IVocabularyBuilder,
    IFileReader,
    ITokenizer,
)

__all__ = [
    "ITextCleaner",
    "ITextSplitter",
    "IVocabularyBuilder",
    "IFileReader",
    "ITokenizer",
]
