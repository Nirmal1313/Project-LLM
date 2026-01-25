"""
LLM Tokenizer Project - Entry Point

This is the main entry point that uses the modular tokenizer package.
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
    """Return a greeting message."""
    return f"Hello, {name}!"


def demo_tokenizer(app: TokenizerApplication) -> None:
    """Demonstrate TokenizerWithUnknown functionality."""
    
    logger.info("--- TokenizerWithUnknown Demo ---")
    tokenizer = app.get_tokenizer("with_unknown")
    
    sample_text1 = "Tao be, e question."
    sample_text2 = "Thesaurus Functions: Synonyms and antonyms are often included."
    sample_text3 = f"{sample_text1}|<ENDOFTEXT>|{sample_text2}"
    
    logger.info(f"Input: {sample_text3}")
    
    # Encode
    encoded_ids = tokenizer.encode(sample_text3)
    logger.info(f"Encoded IDs: {encoded_ids}")
    
    # Decode
    decoded_text = tokenizer.decode(encoded_ids)
    logger.info(f"Decoded text: {decoded_text}")


def main() -> None:
    """Main entry point."""
    logger.info(greet("World"))
    
    # Setup paths
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    data_file = script_dir / "Data" / "The Project The Complete Works of William Shakespeare by William Shakespeare.txt"
    
    # Create application with injected dependencies
    logger.info("Creating TokenizerApplication...")
    app = TokenizerApplication(
        special_tokens=SpecialTokens(),
    )
    
    # Load vocabulary
    vocab_info = app.load_vocabulary_from_file(data_file)
    
    logger.info(f"Vocabulary size: {vocab_info.size}")
    logger.info(f"Special tokens: {vocab_info.special_tokens}")
    logger.debug("Last 5 tokens in vocabulary:")
    for token, idx in vocab_info.sample:
        logger.debug(f"  {token!r}: {idx}")
    
    # Run demo
    demo_tokenizer(app)
    
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    main()