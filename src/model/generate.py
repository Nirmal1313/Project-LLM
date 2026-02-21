import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.model.gpt import GPTModel


class TextGenerator:

    def __init__(self, model: "GPTModel", tokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.context_length: int = model.cfg["context_length"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        eos_token: str | None = "<|endoftext|>",
    ) -> str:
        """
        Generate text with configurable decoding strategy.

        Args:
            prompt:             Input text to continue from
            max_new_tokens:     Maximum tokens to generate
            temperature:        Sampling temperature (0.0 = greedy)
                                - Low (0.2): more confident, deterministic
                                - Medium (0.7): balanced
                                - High (1.5): more random, creative
            top_k:              Keep only top k tokens (None = disabled)
                                - Cuts off the long tail of unlikely tokens
            top_p:              Nucleus sampling threshold (None = disabled)
                                - Keeps smallest set of tokens covering p% probability
                                - top_p=0.9 means keep tokens summing to 90% prob mass
            repetition_penalty: Penalize already-seen tokens (1.0 = no penalty)
                                - >1.0 reduces repetition (e.g. 1.2)
                                - Divides logit by penalty if logit>0, multiplies if logit<0
            eos_token:          Stop generation when this token is produced

        Common recipes:
            Greedy:       temperature=0.0
            Creative:     temperature=0.9, top_k=50, top_p=0.95
            Balanced:     temperature=0.7, top_k=40, top_p=0.9
            No repeats:   temperature=0.8, top_k=40, repetition_penalty=1.3
        """
        self.model.eval()

        token_ids = self.tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
        input_ids = torch.tensor([token_ids], device=self.device)

        eos_id = None
        if eos_token is not None:
            eos_ids = self.tokenizer.encode(eos_token, allowed_special={"<|endoftext|>"})
            if len(eos_ids) == 1:
                eos_id = eos_ids[0]

        generated_ids = self._generate_tokens(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_id=eos_id,
        )

        return self.tokenizer.decode(generated_ids.squeeze(0).tolist())

    @torch.no_grad()
    def predict_next_token(self, prompt: str, top_k: int = 5) -> list[tuple[str, float]]:
        self.model.eval()

        token_ids = self.tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
        input_ids = torch.tensor([token_ids], device=self.device)

        # Truncate to context window
        input_ids = input_ids[:, -self.context_length:]

        logits = self.model(input_ids)              # (1, seq_len, vocab_size)
        next_logits = logits[:, -1, :]              # (1, vocab_size)
        probs = F.softmax(next_logits, dim=-1)      # (1, vocab_size)

        top_probs, top_indices = torch.topk(probs, top_k, dim=-1)

        results = []
        for prob, idx in zip(top_probs.squeeze(0), top_indices.squeeze(0)):
            token_str = self.tokenizer.decode([idx.item()])
            results.append((token_str, prob.item()))

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        repetition_penalty: float,
        eos_id: int | None,
    ) -> torch.Tensor:

        for _ in range(max_new_tokens):
            # Crop to context window
            idx_cond = input_ids[:, -self.context_length:]

            logits = self.model(idx_cond)           # (B, T, V)
            next_logits = logits[:, -1, :]          # (B, V)

            # Apply repetition penalty on already-generated tokens
            if repetition_penalty != 1.0:
                next_logits = self._apply_repetition_penalty(
                    next_logits, input_ids, repetition_penalty
                )

            next_id = self._sample(next_logits, temperature, top_k, top_p)
            input_ids = torch.cat([input_ids, next_id], dim=1)

            if eos_id is not None and next_id.item() == eos_id:
                break

        return input_ids

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        generated_ids: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """
        Repetition penalty (Keskar et al., 2019).

        For tokens already generated:
          - If logit > 0: divide by penalty  (makes it less attractive)
          - If logit < 0: multiply by penalty (makes it even less attractive)

        Higher penalty = less repetition. Typical range: 1.1 - 1.5
        """
        logits = logits.clone()
        unique_ids = generated_ids.unique()
        for token_id in unique_ids:
            if logits[0, token_id] > 0:
                logits[0, token_id] /= penalty
            else:
                logits[0, token_id] *= penalty
        return logits

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
    ) -> torch.Tensor:
        # Greedy
        if temperature == 0.0:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / temperature

        # Top-k filtering
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        # Top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Mask tokens with cumulative probability above the threshold
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")

            # Scatter back to original ordering
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
