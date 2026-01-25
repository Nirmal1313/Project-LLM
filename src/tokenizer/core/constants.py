"""
Global constants and configuration values.

Centralizes all magic strings, regex patterns, and default values
for easy maintenance and consistency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final


class Patterns:
    """Regex patterns used throughout the package."""
    
    # Pattern for splitting text into tokens (punctuation and whitespace)
    SPLIT_PATTERN: Final[re.Pattern[str]] = re.compile(
        r'([,.:;!<>?"()\[\]{}\-_@#$%^&*+=~`|\\\/]|--|\s)'
    )
    
    # Pattern for cleaning special tokens in decoded text
    SPECIAL_TOKEN_CLEANUP: Final[re.Pattern[str]] = re.compile(
        r'\s*(\|<ENDOFTEXT>\||\|<UNK>\|)\s*'
    )
    
    # Pattern for punctuation cleanup in decoded text
    PUNCTUATION_CLEANUP: Final[re.Pattern[str]] = re.compile(
        r'([,.:;!?"()\[\]{}\-_@#$%^&*+=~`\\\/]|--)'
    )
    
    # Placeholder pattern for special tokens during processing
    PLACEHOLDER_TEMPLATE: Final[str] = '\x00SPECIAL{index}\x00'


class Defaults:
    """Default values used throughout the package."""
    
    # Special tokens
    END_OF_TEXT_TOKEN: Final[str] = "|<ENDOFTEXT>|"
    UNKNOWN_TOKEN: Final[str] = "|<UNK>|"
    
    # File reading
    FILE_ENCODING: Final[str] = "utf-8"
    
    # Tokenizer types
    TOKENIZER_WITH_UNKNOWN: Final[str] = "with_unknown"


@dataclass(frozen=True)
class SmartQuoteMapping:
    """Mapping of smart quotes to ASCII equivalents."""
    
    LEFT_SINGLE_QUOTE: str = '\u2018'    # '
    RIGHT_SINGLE_QUOTE: str = '\u2019'   # '
    LEFT_DOUBLE_QUOTE: str = '\u201c'    # "
    RIGHT_DOUBLE_QUOTE: str = '\u201d'   # "
    EN_DASH: str = '\u2013'              # –
    EM_DASH: str = '\u2014'              # —
    ELLIPSIS: str = '\u2026'             # …
    
    @classmethod
    def get_replacement_map(cls) -> dict[str, str]:
        """Get mapping of unicode chars to ASCII replacements."""
        return {
            cls.LEFT_SINGLE_QUOTE: "'",
            cls.RIGHT_SINGLE_QUOTE: "'",
            cls.LEFT_DOUBLE_QUOTE: '"',
            cls.RIGHT_DOUBLE_QUOTE: '"',
            cls.EN_DASH: '-',
            cls.EM_DASH: '-',
            cls.ELLIPSIS: '...',
        }


# Convenience instance
SMART_QUOTES = SmartQuoteMapping()
