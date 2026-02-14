"""
Custom Tokenizers — Learning / Intuition Reference
====================================================

This file contains the early tokenizer implementations built from scratch
to understand how tokenization works before moving to production tokenizers
like tiktoken (BPE).

These are NOT used in the main pipeline. They exist purely for learning.

How it works:
    1. VocabularySort   — takes a list of words, deduplicates & sorts them
    2. dictionary       — maps each unique token to an integer ID
    3. SimpleTokenizerV1 — basic encode/decode using the vocabulary
    4. SimpleTokenizerV2 — adds <UNK> and <ENDOFTEXT> special token handling

Usage example:
    from intuition.custom_tokenizers import (
        VocabularySort, dictionary, SimpleTokenizerV1, SimpleTokenizerV2
    )

    words = ["To", "be", "or", "not", "to", "be"]
    sorted_vocab = VocabularySort(words)
    words_dict = dictionary(sorted_vocab)

    tokenizer = SimpleTokenizerV1(words_dict)
    ids = tokenizer.encode("To be or not to be")
    text = tokenizer.tokenize(ids)
"""

import re


# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

def VocabularySort(words: list) -> list:
    """Calculate the vocabulary size from a list of words."""
    unique_words = sorted(set(words))
    return unique_words


def dictionary(words: list) -> dict:
    """Return a token-to-id mapping with special tokens appended."""
    words.extend(['|<ENDOFTEXT>|', '|<UNK>|'])
    return {token: idx for idx, token in enumerate(words)}


# ---------------------------------------------------------------------------
# SimpleTokenizerV1 — basic word-level tokenizer
# ---------------------------------------------------------------------------

class SimpleTokenizerV1:
    """
    A simple tokenizer that maps words to integer IDs.
    
    - encode(): text  → list of token IDs
    - tokenize(): list of token IDs → text
    
    Limitation: crashes on unknown words (not in vocabulary).
    That's why V2 exists.
    """

    def __init__(self, vocabulary: dict):
        self.vocabulary = vocabulary
        self.int_to_str = {idx: token for token, idx in vocabulary.items()}

    def encode(self, text):
        # Split on punctuation and whitespace, keeping delimiters
        tokens = re.split(r'([,.:;!<>?"()\[\]{}\-_@#$%^&*+=~`|\\\/]|--|\s)', text)
        tokens = [item.strip() for item in tokens if item.strip()]
        ids = [self.vocabulary[token] for token in tokens]
        return ids

    def tokenize(self, ids: list) -> str:
        """Decode token IDs back into a string."""
        words = " ".join(self.int_to_str[id] for id in ids)
        words = re.sub(r'([,.:;!<>?"()\[\]{}\-_@#$%^&*+=~`|\\\/]|--|\s)', r' \1 ', words)
        return words


# ---------------------------------------------------------------------------
# SimpleTokenizerV2 — handles unknown tokens with <UNK>
# ---------------------------------------------------------------------------

class SimpleTokenizerV2:
    """
    Improved tokenizer that handles unknown words gracefully.

    - Unknown words are replaced with |<UNK>| instead of crashing.
    - Special tokens like |<ENDOFTEXT>| are preserved during encoding.
    """

    def __init__(self, vocabulary: dict):
        self.vocabulary = vocabulary
        self.int_to_str = {idx: token for token, idx in vocabulary.items()}

    def encode(self, text):
        # Handle special tokens first — replace them with placeholders
        special_tokens = ['|<ENDOFTEXT>|', '|<UNK>|']
        placeholders = {}
        for i, token in enumerate(special_tokens):
            placeholder = f'\x00SPECIAL{i}\x00'  # Won't appear in real text
            placeholders[placeholder] = token
            text = text.replace(token, f' {placeholder} ')

        # Split on punctuation and whitespace
        preprocessed = re.split(r'([,.:;!<>?"()\[\]{}\-_@#$%^&*+=~`|\\\/]|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # Restore special tokens from placeholders
        preprocessed = [placeholders.get(item, item) for item in preprocessed]

        # Replace unknown tokens with |<UNK>|
        preprocessed = [item if item in self.vocabulary else "|<UNK>|" for item in preprocessed]
        ids = [self.vocabulary[token] for token in preprocessed]
        return ids

    def tokenize(self, ids: list) -> str:
        """Decode token IDs back into a string."""
        words = " ".join(self.int_to_str[id] for id in ids)
        return words.strip()
