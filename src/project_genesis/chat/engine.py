"""Tokenizer, inference, and conversation composition."""

import torch

from project_genesis.chat.conversation import Conversation, Role, format_prompt
from project_genesis.inference import GenerationConfig, GenerationResult, generate
from project_genesis.model import GPTDecoder
from project_genesis.tokenizer import ByteBPETokenizer


def generate_reply(
    model: GPTDecoder,
    tokenizer: ByteBPETokenizer,
    conversation: Conversation,
    user_message: str,
    config: GenerationConfig,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Conversation, GenerationResult]:
    """Append a user message, generate a reply, and return new immutable state."""
    with_user = conversation.append(Role.USER, user_message)
    prompt = format_prompt(with_user, add_assistant_prompt=True)
    prompt_token_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    result = generate(
        model,
        prompt_token_ids,
        config,
        generator=generator,
    )
    reply = tokenizer.decode(result.generated_token_ids)
    if not reply.strip():
        raise ValueError("model generated an empty assistant reply")
    return with_user.append(Role.ASSISTANT, reply), result
