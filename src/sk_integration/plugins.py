"""Semantic Kernel plugins for GPT-2 chat functionality."""

import os
import sys
from typing import Optional
import tiktoken
import torch

from semantic_kernel.functions import kernel_function
from src.model.gpt import GPTModel
from src.model.generate import TextGenerator
from main import load_checkpoint
from src.model.load_gpt2 import GPT2_CONFIG


class ChatPlugins:
    """Collection of kernel functions for chat operations."""

    def __init__(self, model, generator, encoding, config, device):
        """
        Initialize chat plugins.
        
        Args:
            model: GPTModel instance
            generator: TextGenerator instance
            encoding: tiktoken encoding
            config: Model configuration
            device: torch device
        """
        self.model = model
        self.generator = generator
        self.encoding = encoding
        self.config = config
        self.device = device
        self.current_settings = {
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "max_tokens": 150,
        }

    @kernel_function(
        description="Generate a response from the GPT-2 model",
        name="generate_response"
    )
    def generate_response(self, prompt: str, use_instruction_format: bool = True) -> str:
        PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

        if use_instruction_format:
            formatted_prompt = PROMPT_TEMPLATE.format(instruction=prompt)
        else:
            formatted_prompt = prompt

        full_output = self.generator.generate(
            formatted_prompt,
            max_new_tokens=self.current_settings["max_tokens"],
            temperature=self.current_settings["temperature"],
            top_k=self.current_settings["top_k"],
            top_p=self.current_settings["top_p"],
            repetition_penalty=1.2,
        )

        # Extract response
        response_marker = "### Response:\n"
        if response_marker in full_output and use_instruction_format:
            response = full_output.split(response_marker, 1)[1]
        else:
            response = full_output[len(formatted_prompt):] if use_instruction_format else full_output

        response = response.replace("<|endoftext|>", "").strip()
        if "### Instruction:" in response:
            response = response.split("### Instruction:")[0].strip()

        return response

    @kernel_function(
        description="Get next token predictions",
        name="predict_next_tokens"
    )
    def predict_next_tokens(self, text: str, top_k: int = 5) -> str:
        predictions = self.generator.predict_next_token(text, top_k=top_k)
        
        result_lines = ["Top token predictions:"]
        for token, prob in predictions:
            result_lines.append(f"  {repr(token):>15s}  ->  {prob:.4f}")
        
        return "\n".join(result_lines)

    @kernel_function(
        description="Update generation settings (temperature, top_k, top_p, max_tokens)",
        name="update_settings"
    )
    def update_settings(self, setting_name: str, value: str) -> str:
        """
        Update generation settings.
        
        Args:
            setting_name: Name of setting (temperature, top_k, top_p, max_tokens)
            value: New value
            
        Returns:
            Confirmation message
        """
        try:
            if setting_name == "temperature":
                self.current_settings["temperature"] = float(value)
                return f"Temperature set to {self.current_settings['temperature']}"
            elif setting_name == "top_k":
                val = int(value)
                self.current_settings["top_k"] = val if val > 0 else None
                return f"Top-k set to {self.current_settings['top_k']}"
            elif setting_name == "top_p":
                val = float(value)
                self.current_settings["top_p"] = val if 0 < val < 1 else None
                return f"Top-p set to {self.current_settings['top_p']}"
            elif setting_name == "max_tokens":
                self.current_settings["max_tokens"] = int(value)
                return f"Max tokens set to {self.current_settings['max_tokens']}"
            else:
                return f"Unknown setting: {setting_name}"
        except ValueError as e:
            return f"Invalid value for {setting_name}: {e}"

    @kernel_function(
        description="Get current model information",
        name="get_model_info"
    )
    def get_model_info(self) -> str:
        """Get information about the current model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        
        info = [
            f"Total Parameters: {total_params:,}",
            f"Context Length: {self.config['context_length']} tokens",
            f"Device: {self.device}",
            f"Current Settings: {self.current_settings}",
        ]
        return "\n".join(info)

    @kernel_function(
        description="Get current generation settings",
        name="get_settings"
    )
    def get_settings(self) -> str:
        """Get current generation settings."""
        lines = ["Current Generation Settings:"]
        for key, value in self.current_settings.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


def load_chat_plugins(model_type: str = "auto", checkpoint_dir: Optional[str] = None) -> ChatPlugins:
    """
    Load and initialize chat plugins.
    
    Args:
        model_type: Type of model to load (auto, dpo, sft, pretrained)
        checkpoint_dir: Directory containing model checkpoints
        
    Returns:
        Initialized ChatPlugins instance
    """
    if checkpoint_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.join(script_dir, "checkpoints")

    from chat import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, label = load_model(model_type, checkpoint_dir, device)
    encoding = tiktoken.encoding_for_model("gpt-2")
    generator = TextGenerator(model, encoding, device)

    return ChatPlugins(model, generator, encoding, config, device)
