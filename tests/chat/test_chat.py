import pytest
import torch
from torch import Tensor, nn

from project_genesis.chat import (
    Conversation,
    Message,
    Role,
    format_prompt,
    generate_reply,
)
from project_genesis.inference import FinishReason, GenerationConfig
from project_genesis.model import ModelConfig
from project_genesis.tokenizer import ByteBPETokenizer, SpecialTokens, Vocabulary


class ConstantModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = ModelConfig(
            vocab_size=260,
            context_length=128,
            d_model=4,
            n_heads=1,
            d_ff=8,
            dropout=0.0,
            bias=True,
            layer_norm_epsilon=1e-5,
            n_layers=1,
        )
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, inputs: Tensor) -> Tensor:
        logits = torch.zeros(*inputs.shape, self.config.vocab_size)
        logits[..., 69] = 1
        return logits


def _tokenizer() -> ByteBPETokenizer:
    special = SpecialTokens("<pad>", "<bos>", "<eos>", "<unk>")
    vocabulary = Vocabulary(
        special,
        (None,) * 4 + tuple(bytes((value,)) for value in range(256)),
    )
    return ByteBPETokenizer(vocabulary, ())


def test_conversation_is_immutable_and_formats_roles() -> None:
    original = Conversation((Message(Role.SYSTEM, "Be concise."),))
    with_user = original.append(Role.USER, "Hello")

    assert len(original.messages) == 1
    assert format_prompt(with_user, add_assistant_prompt=True) == (
        "<|system|>\nBe concise.\n<|user|>\nHello\n<|assistant|>\n"
    )
    with pytest.raises(ValueError, match="alternate"):
        with_user.append(Role.USER, "Again")


def test_generate_reply_composes_tokenizer_inference_and_new_state() -> None:
    config = GenerationConfig(1, 0.0, 0, 1.0, 1.0, (69,), False)

    conversation, result = generate_reply(
        ConstantModel(),  # type: ignore[arg-type]
        _tokenizer(),
        Conversation(),
        "Write code.",
        config,
    )

    assert [message.role for message in conversation.messages] == [
        Role.USER,
        Role.ASSISTANT,
    ]
    assert conversation.messages[-1].content == "A"
    assert result.finish_reason is FinishReason.STOP
