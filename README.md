# GPT-2 Training and Fine-tuning Pipeline

A comprehensive Python project for training, fine-tuning, and inference with GPT-2 models. Features instruction tuning, direct preference optimization (DPO), and multiple chat interfaces.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Training Pipeline](#training-pipeline)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

This project implements a complete pipeline for:
- **Pretraining**: Fine-tune pretrained GPT-2 (124M) on custom datasets
- **Instruction Tuning**: Supervised fine-tuning (SFT) on instruction-response pairs
- **Preference Alignment**: Direct Preference Optimization (DPO) for alignment
- **Inference**: Multiple chat interfaces with configurable generation parameters
- **Tokenization**: Custom tokenizer implementations with vocabulary management

### Dataset

The project includes Shakespeare text data (`Data/The Project The Complete Works of William Shakespeare by William Shakespeare.txt`) with:
- 30 instruction-response pairs for SFT (`Data/instruction_data.json`)
- 20 preference pairs for DPO (`Data/preference_data.json`)

## Features

**Pre-trained GPT-2 Integration** - Download and load HuggingFace GPT-2 weights
**Instruction Tuning (SFT)** - Fine-tune on instruction-response pairs with loss masking
**DPO Alignment** - Align model with human preferences without RLHF
**Multiple Chat Interfaces** - Standard and Semantic Kernel-based interfaces
**Advanced Text Generation** - Temperature, top-k, top-p, repetition penalty
**Custom Tokenizer** - Built-in tokenizer with <UNK> and <ENDOFTEXT> tokens
**Flexible Architecture** - Support for multiple model sizes (micro, tiny, GPT-2, GPT-3 variants)
**Checkpoint Management** - Save/load best and latest model states  

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (optional, but recommended for training)
- pip or conda

### Setup

1. **Clone or navigate to project directory**

```bash
cd c:\Users\nirma\source\repos\LLM\Project
```

2. **Create and activate virtual environment**

```bash
# Using venv
python -m venv .venv
.venv\Scripts\Activate.ps1

# Or using conda
conda create -n gpt2-project python=3.9
conda activate gpt2-project
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch>=2.0.0` - Deep learning framework
- `tiktoken>=0.4.0` - GPT tokenizer
- `semantic-kernel>=0.9.0` - For SK integration

## Quick Start

### 1. Download Pretrained GPT-2 Weights

```bash
python finetune_gpt2.py
```

This will:
- Download GPT-2 weights (first time only)
- Show generation before fine-tuning
- Fine-tune on Shakespeare data
- Show generation after fine-tuning
- Save checkpoints to `checkpoints/gpt2_shakespeare/`

### 2. Run Chat Interface

```bash
# Auto-detect best available model
python chat.py

# Force specific model
python chat.py --model sft        # Instruction-tuned
python chat.py --model dpo        # DPO-aligned
python chat.py --model pretrained # Raw GPT-2
```

### 3. Use Semantic Kernel Interface (with memory)

```bash
python sk_chat.py
```

## Usage Guide

### Training Pipeline Overview

```
Raw GPT-2 (Pretrained)
    |
    v
Fine-tuning (Optional)
    |
    v
Stage 1: Instruction Tuning (SFT)
  - Format: ### Instruction:\n{inst}\n\n### Response:\n{resp}
  - Loss mask on response tokens only
  - Checkpoint: checkpoints/instruction_tuned/
    |
    v
Stage 2: DPO Alignment (Optional)
  - Input: Chosen vs rejected responses
  - Objective: Maximize preference gap
  - Checkpoint: checkpoints/dpo_aligned/
    |
    v
Final Model
  - Ready for inference
  - Available via chat.py or sk_chat.py
```

### Script Descriptions

#### `main.py` - Base Training Loop

Core training utilities for all training scripts.

**Key Functions:**
- `prepare_data()` - Load, clean, and split data
- `train()` - Main training loop with validation
- `load_checkpoint()` - Resume from checkpoint
- `create_dataloader()` - Create PyTorch DataLoader

**Configuration:**
```python
BATCH_SIZE = 8
CONTEXT_LENGTH = 128
NUM_EPOCHS = 3
MAX_LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 20
```

**Usage:**
```bash
python main.py
```

#### `finetune_gpt2.py` - Pretrained GPT-2 Fine-tuning

Fine-tune pretrained GPT-2 on Shakespeare data.

**Features:**
- Automatic weight download (50GB+ warning)
- Shows generation before/after
- Lower learning rate for fine-tuning (5e-5)
- Saves to `checkpoints/gpt2_shakespeare/`

**Configuration:**
```python
BATCH_SIZE = 4
CONTEXT_LENGTH = 256
NUM_EPOCHS = 2
MAX_LR = 5e-5
MIN_LR = 5e-6
```

**Usage:**
```bash
# First run: downloads weights (~350MB)
python finetune_gpt2.py

# Subsequent runs: uses cached weights
python finetune_gpt2.py
```

#### `instruction_tune.py` - Supervised Fine-tuning

Train on instruction-response pairs with masked loss.

**Format:**
```
### Instruction:
What is the capital of France?

### Response:
Paris<|endoftext|>
```

**Loss Masking:**
- Prompt tokens: loss_mask = 0 (ignored)
- Response tokens: loss_mask = 1 (computed)
- Model sees full prompt but only learns responses

**Data Source:** `Data/instruction_data.json`

**Checkpoint:** `checkpoints/instruction_tuned/`

**Usage:**
```bash
python instruction_tune.py
```

#### `dpo_train.py` - Direct Preference Optimization

Align model with human preferences using DPO.

**Algorithm:**
```
Loss = -log σ(β · (log π(y_w|x)/π_ref - log π(y_l|x)/π_ref))
```

Where:
- π_θ = policy model (updated)
- π_ref = reference model (frozen SFT)
- y_w = chosen response
- y_l = rejected response
- β = temperature (0.1 default)

**Data Format:** `Data/preference_data.json`
```json
[
  {
    "instruction": "Question here",
    "chosen": "Good response",
    "rejected": "Bad response"
  }
]
```

**Checkpoint:** `checkpoints/dpo_aligned/`

**Usage:**
```bash
python dpo_train.py
```

#### `chat.py` - Interactive Chat

Simple terminal chat interface with configurable generation.

**Commands:**
```
/quit, /exit      - Leave chat
/reset            - Clear history
/temp <value>     - Set temperature (e.g., /temp 0.8)
/topk <value>     - Set top-k (e.g., /topk 50)
/topp <value>     - Set top-p (e.g., /topp 0.95)
/tokens <value>   - Set max tokens (e.g., /tokens 200)
```

**Default Settings:**
- Temperature: 0.7
- Top-k: 40
- Top-p: 0.9
- Max tokens: 150

**Usage:**
```bash
python chat.py                  # Auto-select best model
python chat.py --model sft      # Use instruction-tuned
python chat.py --model dpo      # Use DPO-aligned
python chat.py --model pretrained  # Use raw GPT-2
```

**Example Session:**
```
You: What is machine learning?
Assistant: Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed...

You: /temp 0.9

You: Tell me a story
Assistant: [More creative output with higher temperature]
```

#### `sk_chat.py` - Semantic Kernel Chat

Enhanced chat with conversation memory and multi-step tasks.

**Additional Commands:**
```
/info             - Show model information
/settings         - Show current settings
/memory           - Show conversation summary
/refine           - Refine last response
/multi <steps>    - Execute multi-step task
```

**Features:**
- Full conversation history
- Context-aware responses
- Session memory
- Task refinement

**Usage:**
```bash
python sk_chat.py
python sk_chat.py --model dpo
```

### Custom Tokenizer

The project includes a custom tokenizer system for vocabulary management.

**Components:**
- `TextCleaner` - Unicode normalization and special character replacement
- `TextSplitter` - Regex-based token splitting
- `VocabularyBuilder` - Create token-to-ID mappings
- `TokenizerWithUnknown` - Handle unknown tokens with <UNK>

**Usage Example:**
```python
from src.tokenizer.app.application import TokenizerApplication
from pathlib import Path

app = TokenizerApplication()
app.load_vocabulary_from_file(Path("data.txt"))
tokenizer = app.get_tokenizer("with_unknown")

result = tokenizer.encode("Hello world!")
print(result.token_ids)
```

## Project Structure

```
LLM/Project/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── pyproject.toml                      # Project metadata
│
├── main.py                             # Base training utilities
├── finetune_gpt2.py                    # GPT-2 fine-tuning
├── instruction_tune.py                 # Supervised FT (SFT)
├── dpo_train.py                        # DPO alignment
├── chat.py                             # Chat interface
├── sk_chat.py                          # Semantic Kernel chat
│
├── checkpoints/                        # Model checkpoints
│   ├── gpt2_pretrained.pt              # Pretrained GPT-2
│   ├── gpt2_shakespeare/               # Fine-tuned on Shakespeare
│   │   ├── best_model.pt
│   │   └── latest_model.pt
│   ├── instruction_tuned/              # SFT checkpoint
│   │   ├── best_model.pt
│   │   └── latest_model.pt
│   └── dpo_aligned/                    # DPO checkpoint
│       ├── best_model.pt
│       └── latest_model.pt
│
├── Data/                               # Training data
│   ├── The Project The Complete Works of William Shakespeare.txt
│   ├── instruction_data.json           # 30 instruction-response pairs
│   └── preference_data.json            # 20 preference pairs
│
├── logs/                               # Training logs
│
├── src/
│   ├── model/                          # Model implementations
│   │   ├── gpt.py                      # GPTModel
│   │   ├── attention.py                # MultiHeadAttention
│   │   ├── transformerBlock.py         # TransformerBlock
│   │   ├── generate.py                 # TextGenerator
│   │   └── load_gpt2.py                # GPT-2 weight mapping
│   │
│   ├── tokenizer/                      # Tokenizer system
│   │   ├── app/
│   │   │   └── application.py          # Main app
│   │   ├── core/
│   │   │   ├── constants.py
│   │   │   ├── exceptions.py
│   │   │   ├── logging.py
│   │   │   └── models/
│   │   ├── processing/
│   │   │   ├── cleaner.py
│   │   │   ├── splitter.py
│   │   │   └── file_reader.py
│   │   ├── tokenizers/
│   │   │   ├── base.py
│   │   │   ├── simple.py
│   │   │   ├── with_unknown.py
│   │   │   └── factory.py
│   │   ├── protocols/
│   │   │   └── interfaces.py
│   │   └── vocabulary/
│   │       └── builder.py
│   │
│   ├── sk_integration/                 # Semantic Kernel integration
│   │   ├── orchestrator.py             # Task orchestration
│   │   ├── plugins.py                  # SK plugins
│   │   └── memory_manager.py           # Conversation memory
│   │
│   └── intuition/
│       └── custom_tokenizers.py        # Learning examples
│
├── exercises/                          # Educational materials
│   ├── attention_exercises.md
│   └── hard_attention_exercises.md
│
└── test files
    ├── test_main.py
    └── test_sk_integration.py
```

## Training Pipeline

### Stage 1: Base Model (Optional)

If you want to start from scratch with your own pretrained weights:

```python
# In main.py, adjust configuration
GPT_CONFIG = {**GPT_CONFIGS["micro"], "vocab_size": encoding.n_vocab}
BATCH_SIZE = 8
NUM_EPOCHS = 3
```

### Stage 2: Instruction Tuning

Fine-tune on instruction-response pairs:

```bash
python instruction_tune.py
```

**What happens:**
1. Loads pretrained GPT-2 (or latest checkpoint)
2. Loads 30 instruction-response pairs from `Data/instruction_data.json`
3. Formats as: `### Instruction:\n{inst}\n\n### Response:\n{resp}<|endoftext|>`
4. Masks loss on instruction tokens (model only learns to generate responses)
5. Saves best/latest checkpoints to `checkpoints/instruction_tuned/`

**Output:**
```
Instruction Tuning (SFT)
============================================================
  Epochs:        5
  Batches/epoch: 8
  Total steps:   40
  Max LR:        2e-05
  Warmup:        20 steps
============================================================

  Epoch 1/5 |######-----| 50.0% | Loss: 4.2341
  >> Epoch 1/5 DONE | Train Loss: 4.1234 | Val Loss: 4.0456 | Time: 12.3s

  ...

Training complete in 65.4s
Train losses: 4.1234 -> 3.8945 -> 3.7234
Val   losses: 4.0456 -> 3.9123 -> 3.8901
Best val loss: 3.8901
```

### Stage 3: DPO Alignment

Align with human preferences:

```bash
python dpo_train.py
```

**What happens:**
1. Loads instruction-tuned model
2. Creates frozen reference model copy
3. Loads 20 preference pairs from `Data/preference_data.json`
4. Applies DPO loss to increase gap between chosen/rejected responses
5. Saves checkpoints to `checkpoints/dpo_aligned/`

**Output:**
```
DPO Alignment Training
============================================================
  Epochs:        5
  Batches/epoch: 5
  Beta:          0.1
  Max LR:        5e-06
============================================================

  Epoch 1/5 DONE | Train Loss: 0.6234 | Val Loss: 0.5891
  ...
```

## API Reference

### GPTModel

```python
from src.model.gpt import GPTModel, GPT_CONFIGS

# Available configs
configs = {
    "gpt2-124M": {"d_model": 768, "n_heads": 12, "n_layers": 12, ...},
    "tiny": {"d_model": 256, "n_heads": 4, "n_layers": 4, ...},
    "micro": {"d_model": 128, "n_heads": 4, "n_layers": 2, ...},
}

# Create model
model = GPTModel(GPT_CONFIGS["micro"])
model.to(device)

# Forward pass
logits = model(input_ids)  # (batch_size, seq_len, vocab_size)
```

### TextGenerator

```python
from src.model.generate import TextGenerator
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-2")
generator = TextGenerator(model, encoding, device)

# Simple generation
text = generator.generate("Once upon a time", max_new_tokens=50)

# With sampling
text = generator.generate(
    prompt="Once upon a time",
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    repetition_penalty=1.2
)

# Get next token predictions
predictions = generator.predict_next_token("Once", top_k=5)
# [(token_str, probability), ...]
```

### Custom Tokenizer

```python
from src.tokenizer.app.application import TokenizerApplication
from pathlib import Path

# Create app
app = TokenizerApplication()

# Load vocabulary
vocab_info = app.load_vocabulary_from_file(Path("data.txt"))
print(f"Vocabulary size: {vocab_info.size}")

# Get tokenizer
tokenizer = app.get_tokenizer("with_unknown")

# Encode
tokens = tokenizer.encode("Hello world")

# Decode
text = tokenizer.decode(tokens)

# Get both
result = tokenizer.encode_with_tokens("Hello world")
print(result.token_ids)
print(result.tokens)
```

## Examples

### Example 1: Fine-tune on Custom Data

1. **Prepare instruction data:**

```json
// Data/custom_instructions.json
[
  {
    "instruction": "What is Python?",
    "response": "Python is a high-level programming language..."
  },
  {
    "instruction": "How do I install packages?",
    "response": "You can use pip: pip install package_name"
  }
]
```

2. **Modify instruction_tune.py:**

```python
# Line 60, in main():
json_file = "Data/custom_instructions.json"

with open(json_file, 'r') as f:
    examples = json.load(f)
```

3. **Run training:**

```bash
python instruction_tune.py
```

### Example 2: Generate with Different Sampling

```python
from src.model.generate import TextGenerator

# Greedy (deterministic)
greedy = generator.generate(prompt, temperature=0.0, max_new_tokens=50)

# Creative (high temperature)
creative = generator.generate(
    prompt, temperature=1.2, top_k=50, top_p=0.95, max_new_tokens=50
)

# Balanced
balanced = generator.generate(
    prompt, temperature=0.7, top_k=40, top_p=0.9, max_new_tokens=50
)

# No repetition
no_repeat = generator.generate(
    prompt, temperature=0.8, repetition_penalty=1.3, max_new_tokens=50
)
```

### Example 3: Load Checkpoint and Resume

```python
from main import load_checkpoint

# Load checkpoint
checkpoint = load_checkpoint("checkpoints/instruction_tuned/best_model.pt", model)
print(f"Epoch: {checkpoint['epoch']}")
print(f"Loss: {checkpoint['train_loss']}")

# Resume training
for epoch in range(checkpoint['epoch'], NUM_EPOCHS):
    train(model, train_loader, val_loader, device, config, ...)
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**
- Reduce `BATCH_SIZE` in training scripts
- Reduce `CONTEXT_LENGTH`
- Use CPU: `device = torch.device('cpu')`

```python
BATCH_SIZE = 2  # Reduce from 8
CONTEXT_LENGTH = 128  # Reduce from 256
```

### Issue: Model not generating good responses

**Solutions:**
1. **Insufficient training:**
   - Increase `NUM_EPOCHS`
   - More training data
   - Lower learning rate with longer training

2. **Overfitting:**
   - Reduce `MAX_LR`
   - Increase `NUM_EPOCHS` with early stopping
   - More diverse data

3. **Generation parameters:**
   ```python
   # Try different settings
   temperature = 0.7  # Was 1.0
   top_k = 40  # Was None
   top_p = 0.9  # Was None
   ```

### Issue: Checkpoint file not found

**Solution:**
Ensure model training completes or check correct path:

```bash
# List available checkpoints
dir checkpoints

# Check specific model
dir checkpoints\instruction_tuned
```

### Issue: Semantic Kernel import error

**Solution:**
Install missing dependency:

```bash
pip install semantic-kernel>=0.9.0
```

### Issue: Tokenizer vocabulary empty

**Solution:**
Load vocabulary before using:

```python
from src.tokenizer.app.application import TokenizerApplication
from pathlib import Path

app = TokenizerApplication()
app.load_vocabulary_from_file(Path("Data/your_data.txt"))

# Now use tokenizer
tokenizer = app.get_tokenizer("with_unknown")
```

## Performance Tips

1. **Mixed Precision Training:**
```python
from torch.cuda.amp import autocast

with autocast():
    logits = model(input_ids)
    loss = F.cross_entropy(...)
```

2. **Gradient Accumulation:**
```python
accumulation_steps = 4
for batch_idx, batch in enumerate(train_loader):
    loss = compute_loss(batch)
    loss.backward()
    
    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

3. **Use Smaller Model for Testing:**
```python
# Instead of GPT-2 (124M), use:
GPT_CONFIG = GPT_CONFIGS["micro"]  # 1.3M parameters
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.0.0 | Deep learning framework |
| tiktoken | >=0.4.0 | GPT tokenizer |
| semantic-kernel | >=0.9.0 | Task orchestration |
| transformers | >=4.0.0 | HuggingFace models |

## License

This project is provided as-is for educational and research purposes.

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review example scripts in the codebase
3. Check training logs in `logs/` directory

---

**Last Updated:** February 2026  
**Python Version:** 3.9+  
**CUDA Version:** 11.8+ (optional)
