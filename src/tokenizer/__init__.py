"""
LLM Tokenizer Package

A production-ready tokenizer library with SOLID principles.

Usage:
    from src.tokenizer import TokenizerApplication, SpecialTokens
    
    app = TokenizerApplication()
    app.load_vocabulary_from_file("data.txt")
    tokenizer = app.get_tokenizer("with_unknown")
"""

from src.tokenizer.models.schemas import SpecialTokens, TokenizerResult, VocabularyInfo
from src.tokenizer.protocols.interfaces import (
    ITextCleaner,
    ITextSplitter,
    IVocabularyBuilder,
    IFileReader,
)
from src.tokenizer.processing.cleaner import TextCleaner
from src.tokenizer.processing.splitter import TextSplitter
from src.tokenizer.processing.file_reader import FileReader
from src.tokenizer.vocabulary.builder import VocabularyBuilder
from src.tokenizer.tokenizers.base import BaseTokenizer
from src.tokenizer.tokenizers.simple import SimpleTokenizer
from src.tokenizer.tokenizers.with_unknown import TokenizerWithUnknown
from src.tokenizer.tokenizers.factory import TokenizerFactory
from src.tokenizer.app.application import TokenizerApplication

__all__ = [
    # Models
    "SpecialTokens",
    "TokenizerResult",
    "VocabularyInfo",
    # Protocols
    "ITextCleaner",
    "ITextSplitter",
    "IVocabularyBuilder",
    "IFileReader",
    # Processing
    "TextCleaner",
    "TextSplitter",
    "FileReader",
    # Vocabulary
    "VocabularyBuilder",
    # Tokenizers
    "BaseTokenizer",
    "SimpleTokenizer",
    "TokenizerWithUnknown",
    "TokenizerFactory",
    # Application
    "TokenizerApplication",
]

__version__ = "1.0.0"
