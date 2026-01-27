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

def split_documents(text: str) -> list[str]:
    """Split text into documents by play boundaries."""
    # Regex to match play titles (all caps, multiple words, at start of line)
    play_pattern = r'^([A-Z][A-Z\' ]+[A-Z])$'
    
    documents = []
    current_doc = []
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        # Check if this line is a play title
        if re.match(play_pattern, line.strip()) and len(line.strip()) > 10:
            # Check context: should have blank lines before/after
            prev_blank = i > 0 and not lines[i-1].strip()
            next_blank = i < len(lines)-1 and not lines[i+1].strip()
            
            if prev_blank or next_blank:
                # Save current document if it has content
                if current_doc:
                    doc_text = '\n'.join(current_doc).strip()
                    if len(doc_text) > 100:  # Minimum length filter
                        documents.append(doc_text)
                # Start new document
                current_doc = [line]
                continue
        
        current_doc.append(line)
    
    # Add final document
    if current_doc:
        doc_text = '\n'.join(current_doc).strip()
        if len(doc_text) > 100:
            documents.append(doc_text)
    
    return documents


def split_into_words(text: str) -> list:
    """Split text into words using regex."""
    # Clean the text first
    text = clean_text(text)
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
    documents = split_documents(text)
    print(f"Found {len(documents)} documents")
    
    # Filter out very short documents and join with separator
    # filtered_docs = []
    # for doc in documents:
    #     # Remove empty/whitespace-only content
    #     doc = doc.strip()
    #     if not doc:
    #         continue
    #     # Minimum token threshold (approximate: ~4 chars per token)
    #     if len(doc) < 400:  # ~100 tokens minimum
    #         continue
    #     # Truncate very long documents (optional, adjust as needed)
    #     if len(doc) > 500000:  # ~125k tokens
    #         doc = doc[:500000]
    #     filtered_docs.append(doc)
    
    # print(f"After filtering: {len(filtered_docs)} documents")
    
    # Join with <|endoftext|> separator
    text_with_boundaries = '<|endoftext|>'.join(documents)
    
    
    encoding = tiktoken.encoding_for_model("gpt-4o")

    # Encode text into a list of token integers
    tokens = encoding.encode(text_with_boundaries, allowed_special={"<|endoftext|>"})
    print(f"Tokens: {len(tokens)} tokens")
    

    dataloader = create_dataloader(text_with_boundaries, encoding, batch_size=8, max_length=256, stride=256)  # set num_workers=0 inside
    input_seq, target_seq = next(iter(dataloader))

    vocab_size = encoding.n_vocab
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    token_embeddings = token_embedding_layer(input_seq)          # (B, T, D)
    seq_len = input_seq.size(1)
    pos_indices = torch.arange(seq_len, device=input_seq.device) # (T,)
    pos_embedding_layer = torch.nn.Embedding(seq_len, output_dim)
    pos_embeddings = pos_embedding_layer(pos_indices)            # (T, D)
    input_embeddings = token_embeddings + pos_embeddings         # (B, T, D)
    print(input_embeddings.shape)
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
    

def create_dataloader(txt, tokenizer, max_length, stride, batch_size) -> DataLoader:
    """Create DataLoader for the GPT dataset."""
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    return dataloader

if __name__ == "__main__":
    print(greet("World"))
    print(readme())
    print(readtxtFile())