"""Custom exceptions for the tokenizer package."""

from __future__ import annotations

from typing import Any, Optional


class TokenizerBaseError(Exception):
    
    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v!r}" for k, v in self.details.items())
            return f"{self.message} [{details_str}]"
        return self.message
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r}, details={self.details!r})"


class TokenizerError(TokenizerBaseError):
    """General tokenizer operation errors."""
    pass


class EncodingError(TokenizerError):
    """Error during text encoding."""
    
    def __init__(self, message: str, text: Optional[str] = None) -> None:
        details = {"text_preview": text[:50] + "..." if text and len(text) > 50 else text}
        super().__init__(message, details)


class DecodingError(TokenizerError):
    """Error during token decoding."""
    
    def __init__(self, message: str, token_ids: Optional[list[int]] = None) -> None:
        details = {"token_ids": token_ids[:10] if token_ids else None}
        super().__init__(message, details)


class UnknownTokenError(TokenizerError):
    """Error when encountering an unknown token."""
    
    def __init__(self, token: str, vocab_size: Optional[int] = None) -> None:
        self.token = token
        details = {"token": token}
        if vocab_size is not None:
            details["vocab_size"] = vocab_size
        super().__init__(f"Unknown token: '{token}'", details)


class VocabularyError(TokenizerBaseError):
    """Vocabulary-related errors."""
    
    def __init__(
        self,
        message: str,
        vocab_size: Optional[int] = None,
        missing_tokens: Optional[list[str]] = None,
    ) -> None:
        details: dict[str, Any] = {}
        if vocab_size is not None:
            details["vocab_size"] = vocab_size
        if missing_tokens:
            details["missing_tokens"] = missing_tokens[:5]  # Limit for readability
        super().__init__(message, details)


class FileReadError(TokenizerBaseError):
    """File reading errors."""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        details: dict[str, Any] = {}
        if file_path:
            details["file_path"] = file_path
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(message, details)
        self.original_error = original_error
