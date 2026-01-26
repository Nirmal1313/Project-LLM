"""
LLM Tokenizer Project - Entry Point

This is the main entry point that uses the modular tokenizer package.
"""
"""
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.tokenizer import TokenizerApplication, SpecialTokens
from src.tokenizer.core import setup_logging, get_logger
import logging


# Initialize logging (logs to both console and logs/tokenizer.log)
log_file = setup_logging(level=logging.INFO)
logger = get_logger(__name__)

if log_file:
    logger.info(f"Logging to file: {log_file.absolute()}")


def greet(name: str) -> str:
    return f"Hello, {name}!"


def demo_tokenizer(app: TokenizerApplication) -> None:
    
    logger.info("--- TokenizerWithUnknown Demo ---")
    tokenizer = app.get_tokenizer("with_unknown")
    print(tokenizer)
    # sample_text1 = "Tao be, e question."
    # sample_text2 = "Thesaurus Functions: Synonyms and antonyms are often included."
    # sample_text3 = f"{sample_text1}|<ENDOFTEXT>|{sample_text2}"
    
    # logger.info(f"Input: {sample_text3}")
    
    # # Encode
    # encoded_ids = tokenizer.encode(sample_text3)
    # logger.info(f"Encoded IDs: {encoded_ids}")
    
    # # Decode
    # decoded_text = tokenizer.decode(encoded_ids)
    # logger.info(f"Decoded text: {decoded_text}")


def main() -> None:
    #logger.info(greet("World"))
    
    # Setup paths
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    data_file = script_dir / "Data" / "The Project The Complete Works of William Shakespeare by William Shakespeare.txt"
    
    # Create application with injected dependencies
    #logger.info("Creating TokenizerApplication...")
    app = TokenizerApplication(
        special_tokens=SpecialTokens(),
    )
    
    # Load vocabulary
    vocab_info = app.load_vocabulary_from_file(data_file)
    
    #logger.info(f"Vocabulary size: {vocab_info.size}")
    #logger.info(f"Special tokens: {vocab_info.special_tokens}")
    #logger.debug("Last 5 tokens in vocabulary:")
    #for token, idx in vocab_info.sample:
        #logger.debug(f"  {token!r}: {idx}")
    
    # Run demo
    #demo_tokenizer(app)
    
    print(vocab_info)
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    main() """
    

import os
import re
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
def greet(name: str) -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"


def readme() -> str:
    """Return the README content."""
    return """
    # Simple Python Test Project

    This is a simple Python project that demonstrates basic functions such as greeting, addition, and multiplication.

    ## Functions

    - `greet(name: str) -> str`: Returns a greeting message.
    """
    
def readtxtFile() -> str:
    """Return the content of a sample text file."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "Data", "The Project The Complete Works of William Shakespeare by William Shakespeare.txt")
    
    with open(file_path, "r", encoding="utf-8") as file:
        raw_data = file.read()
    
    #print("total number of characters in text file:", len(raw_data))
    #print("first 99 characters from the text file:\n", raw_data)
    split_into_words(raw_data);
    return "0";

def clean_text(text: str) -> str:
    """Clean text by replacing smart quotes and special characters with standard ones."""
    import unicodedata
    
    # Normalize unicode characters (NFKD decomposes characters)
    text = unicodedata.normalize('NFKD', text)
    
    # Replace common smart quotes and dashes
    replacements = {
        '\u2018': "'",  # Left single quote '
        '\u2019': "'",  # Right single quote '
        '\u201c': '"',  # Left double quote "
        '\u201d': '"',  # Right double quote "
        '\u2013': '-',  # En-dash –
        '\u2014': '-',  # Em-dash —
        '\u2026': '...',  # Ellipsis …
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove any remaining non-ASCII characters (like �, accented chars, etc.)
    # Keep only ASCII printable characters
    text = ''.join(char if ord(char) < 128 else '' for char in text)
    
    return text

def split_into_words(text: str) -> list:
    """Split text into words using regex."""
    # Clean the text first
#     text = clean_text(text1)
#     words = re.split(r'([,.:;!<>?"()\[\]{}\-_@#$%^&*+=~`|\\\/]|--|\s)', text)
#     #print("Words extracted from the text snippet:", words);
#     result = [item for item in words if item.strip()]
#     sortedData=VocabularySort(result);
#     words_dict=dictionary(sortedData);
   
#     SimpleTokenizerV1_instance = SimpleTokenizerV1(words_dict);
#     sample_text = "To be, or not to be: that is the question."
#     encoded_ids = SimpleTokenizerV1_instance.encode(sample_text);
    
#    # print("Encoded IDs for sample text:", encoded_ids);
#     decoded_text = SimpleTokenizerV1_instance.tokenize(encoded_ids);
    #print("Decoded text from IDs:", decoded_text);
    
    # sample_text1 = "Tao be, e question."
    # sample_text2 = "Thesaurus Functions: Synonyms and antonyms are often included."
    # sample_text3 = "|<ENDOFTEXT>|".join((sample_text1, sample_text2));
    # SimpleTokenizerV2_instance = SimpleTokenizerV2(words_dict);
    # encoded_ids1 = SimpleTokenizerV2_instance.encode(sample_text3);
    # #print("Encoded IDs for sample text:", encoded_ids1);
    # decoded_text = SimpleTokenizerV2_instance.tokenize(encoded_ids1);
    # #print("Decoded text from IDs:", decoded_text);
    
    
    encoding = tiktoken.encoding_for_model("gpt-4o")

    # Encode text into a list of token integers
    tokens = encoding.encode(text)
    print(f"Tokens: {len(tokens)} tokens")

    # Decode tokens back into text
    decoded_text = encoding.decode(tokens)
   # print(f"Decoded text: {len(decoded_text)} characters")

    # Count the number of tokens
    token_count = len(tokens)
    #print(f"Token count: {token_count}")
    dataloader=create_dataloader(text, encoding, max_length=256, stride=128, batch_size=4);
    data_itr=iter(dataloader);
    input_seq, target_seq=next(data_itr);
    print("Input Sequence Tokens:", input_seq)
    print("Target Sequence Tokens:", target_seq)
   
    return tokens
 
# def VocabularySort(words: list) -> list:
#     ""Calculate the vocabulary size from a list of words.""
#     unique_words = sorted(set(words))
#     return unique_words

# def dictionary(words: list) -> dict:
#     ""Return a sample dictionary.""
#     words.extend(['|<ENDOFTEXT>|', '|<UNK>|'])
#     return {token: idx for idx, token in enumerate(words)}

# class SimpleTokenizerV1:
#     ""A simple tokenizer class.""
    
#     def __init__(self, vocabulary: dict):
#         self.vocabulary = vocabulary
#         self.int_to_str = {idx: token for token, idx in vocabulary.items()}
    
#     def encode(self, text):
#         preprocessed = (text)
#         # Use the same regex split as split_into_words to ensure consistent tokenization
#         #tokens = re.split(r'([,.:;!<>?"()\[\]{}\-_@#$%^&*+=~`|\\\/]|--|\s)', preprocessed)
#         tokens = [item.strip() for item in tokens if item.strip()]
#         ids = [self.vocabulary[token] for token in tokens]
#         return ids
    
#     def tokenize(self,  ids: list) -> str:
#         ""Tokenize the input text into a list of tokens.""
#         words = " ".join(self.int_to_str[id] for id in ids)
#         words =re.sub(r'([,.:;!<>?"()\[\]{}\-_@#$%^&*+=~`|\\\/]|--|\s)', r' \1 ', words)
#         return words

# class SimpleTokenizerV2:
#     ""A simple tokenizer class.""
    
#     def __init__(self, vocabulary: dict):
#         self.vocabulary = vocabulary
#         self.int_to_str = {idx: token for token, idx in vocabulary.items()}
    
#     def encode(self, text):
#         # Handle special tokens first - replace them with placeholders
#         special_tokens = ['|<ENDOFTEXT>|', '|<UNK>|']
#         placeholders = {}
#         for i, token in enumerate(special_tokens):
#             placeholder = f'\x00SPECIAL{i}\x00'  # Use null char as delimiter (won't appear in text)
#             placeholders[placeholder] = token
#             text = text.replace(token, f' {placeholder} ')
        
#         # Use the same regex split as split_into_words to ensure consistent tokenization
#         #preprocessed = re.split(r'([,.:;!<>?"()\[\]{}\-_@#$%^&*+=~`|\\\/]|--|\s)', text)
#         preprocessed = [item.strip() for item in preprocessed if item.strip()]
        
#         # Restore special tokens from placeholders
#         preprocessed = [placeholders.get(item, item) for item in preprocessed]
        
#         # Replace unknown tokens with |<UNK>|
#         preprocessed = [item if item in self.vocabulary else "|<UNK>|" for item in preprocessed]
#         ids = [self.vocabulary[token] for token in preprocessed]
#         return ids
    
#     def tokenize(self,  ids: list) -> str:
#         ""Tokenize the input text into a list of tokens.""
#         words = " ".join(self.int_to_str[id] for id in ids)
#         # Don't split special tokens
#         #words = re.sub(r'\s*(\|<ENDOFTEXT>\||\|<UNK>\|)\s*', r' \1 ', words)
#         #words = re.sub(r'([,.:;!?"()\[\]{}\-_@#$%^&*+=~`\\\/]|--)', r' \1 ', words)
#         return words.strip()
    
class GPTDataset(Dataset):
    """Custom Dataset for GPT tokenized data."""
    
    def __init__(self, txt, tokenizer, max_length, stride) -> None:
        self.input_ids = []
        self.target_ids=[]
        
        token_ids=tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        
        for i in range(0, len(token_ids)- max_length, stride):
            inputchunk = token_ids[i:i + max_length]
            targetchunk = token_ids[i +1:i + max_length + 1]
            self.input_ids.append(torch.tensor(inputchunk))
            self.target_ids.append(torch.tensor(targetchunk))
            
    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        input_seq = self.input_ids[idx]
        target_seq = self.target_ids[idx]
        return input_seq, target_seq
    

def create_dataloader(txt, tokenizer, max_length=256, stride=128, batch_size=4) -> DataLoader:
    """Create DataLoader for the GPT dataset."""
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    return dataloader

if __name__ == "__main__":
    print(greet("World"))
    print(readme())
    print(readtxtFile())