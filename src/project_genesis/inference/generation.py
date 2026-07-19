"""Bounded autoregressive decoding and sampling."""

from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor

from project_genesis.inference.config import GenerationConfig
from project_genesis.model import GPTDecoder


class FinishReason(StrEnum):
    """Reason autoregressive generation ended."""

    STOP = "stop"
    LENGTH = "length"
    CONTEXT = "context"


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Prompt-independent generated suffix and its termination reason."""

    generated_token_ids: tuple[int, ...]
    finish_reason: FinishReason


def sample_next_token(
    logits: Tensor,
    previous_token_ids: tuple[int, ...],
    config: GenerationConfig,
    *,
    generator: torch.Generator | None = None,
) -> int:
    """Choose one token using greedy or temperature/top-k/top-p sampling."""
    if logits.ndim != 1 or logits.numel() == 0:
        raise ValueError("logits must be a non-empty vocabulary vector")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("generation logits must be finite")
    vocabulary_size = logits.numel()
    if any(not 0 <= token_id < vocabulary_size for token_id in previous_token_ids):
        raise ValueError("previous token ID is outside the model vocabulary")

    scores = logits.float().clone()
    if config.repetition_penalty != 1 and previous_token_ids:
        repeated = torch.tensor(
            sorted(set(previous_token_ids)),
            device=scores.device,
        )
        selected = scores[repeated]
        scores[repeated] = torch.where(
            selected < 0,
            selected * config.repetition_penalty,
            selected / config.repetition_penalty,
        )
    if config.temperature == 0:
        return int(scores.argmax().item())

    scores /= config.temperature
    if config.top_k:
        kept = min(config.top_k, vocabulary_size)
        threshold = torch.topk(scores, kept).values[-1]
        scores.masked_fill_(scores < threshold, -torch.inf)
    if config.top_p < 1:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        cumulative = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
        remove = cumulative > config.top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scores[sorted_indices[remove]] = -torch.inf
    probabilities = torch.softmax(scores, dim=-1)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def generate(
    model: GPTDecoder,
    prompt_token_ids: tuple[int, ...],
    config: GenerationConfig,
    *,
    generator: torch.Generator | None = None,
) -> GenerationResult:
    """Generate one token sequence without exceeding the model context window."""
    if not prompt_token_ids:
        raise ValueError("prompt_token_ids must not be empty")
    if any(not 0 <= token_id < model.config.vocab_size for token_id in prompt_token_ids):
        raise ValueError("prompt token ID is outside the model vocabulary")
    if any(token_id >= model.config.vocab_size for token_id in config.stop_token_ids):
        raise ValueError("stop token ID is outside the model vocabulary")
    if len(prompt_token_ids) > model.config.context_length:
        raise ValueError("prompt exceeds model context length")

    available = model.config.context_length - len(prompt_token_ids)
    if not available:
        return GenerationResult((), FinishReason.CONTEXT)
    limit = min(config.max_new_tokens, available)
    device = next(model.parameters()).device
    tokens = list(prompt_token_ids)
    generated: list[int] = []
    reason = FinishReason.CONTEXT if available <= config.max_new_tokens else FinishReason.LENGTH
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            inputs = torch.tensor([tokens], dtype=torch.long, device=device)
            if config.use_cache:
                logits, cache = model.forward_cached(inputs)
            else:
                logits = model(inputs)
                cache = None
            for index in range(limit):
                token_id = sample_next_token(
                    logits[0, -1],
                    tuple(tokens),
                    config,
                    generator=generator,
                )
                tokens.append(token_id)
                generated.append(token_id)
                if token_id in config.stop_token_ids:
                    reason = FinishReason.STOP
                    break
                if index + 1 == limit:
                    break
                next_input = torch.tensor([[token_id]], dtype=torch.long, device=device)
                if config.use_cache:
                    logits, cache = model.forward_cached(next_input, cache)
                else:
                    logits = model(torch.tensor([tokens], dtype=torch.long, device=device))
    finally:
        model.train(was_training)
    return GenerationResult(tuple(generated), reason)
