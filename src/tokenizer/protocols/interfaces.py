"""Protocol interfaces for dependency inversion."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ITextCleaner(Protocol):
    def clean(self, text: str) -> str:
        ...


@runtime_checkable
class ITextSplitter(Protocol):
    def split(self, text: str) -> list[str]:
        ...


@runtime_checkable
class IVocabularyBuilder(Protocol):
    def build(self, tokens: list[str]) -> dict[str, int]:
        ...


@runtime_checkable
class IFileReader(Protocol):
    def read(self, file_path: Path) -> str:
        ...


@runtime_checkable
class ITokenizer(Protocol):
    def encode(self, text: str) -> list[int]:
        ...
    
    def decode(self, ids: list[int]) -> str:
        ...
    
    @property
    def vocabulary(self) -> dict[str, int]:
        """Return the vocabulary mapping."""
        ...
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        ...
