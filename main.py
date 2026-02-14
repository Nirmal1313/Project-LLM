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
    documents = split_documents(text)
    print(f"Found {len(documents)} documents")
    
    # Join documents with <|endoftext|> separator
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
    pos_embedding_layer = torch.nn.Embedding(256, output_dim)  # max_length = 256

    # Process ALL batches, not just the first one
    for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
        # input_seq shape: (8, 256) — 8 sequences of 256 tokens each
        
        token_embeddings = token_embedding_layer(input_seq)            # (B, T, D)
        seq_len = input_seq.size(1)
        pos_indices = torch.arange(seq_len, device=input_seq.device)   # (T,)
        pos_embeddings = pos_embedding_layer(pos_indices)               # (T, D)
        input_embeddings = token_embeddings + pos_embeddings            # (B, T, D)
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, input_embeddings shape: {input_embeddings.shape}")

        print(f"Total batches processed: {batch_idx + 1}")
    return tokens


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