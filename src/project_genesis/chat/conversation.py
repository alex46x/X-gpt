"""Immutable conversation state and deterministic prompt formatting."""

from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    """Supported conversation roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    """One non-empty role-tagged conversation message."""

    role: Role
    content: str

    def __post_init__(self) -> None:
        """Reject empty message content."""
        if not self.content.strip():
            raise ValueError("message content must be non-empty")


@dataclass(frozen=True, slots=True)
class Conversation:
    """Ordered system and alternating user/assistant messages."""

    messages: tuple[Message, ...] = ()

    def __post_init__(self) -> None:
        """Validate system placement and role alternation."""
        messages = self.messages
        if any(message.role is Role.SYSTEM for message in messages[1:]):
            raise ValueError("system message is allowed only at the beginning")
        conversational = messages[1:] if messages and messages[0].role is Role.SYSTEM else messages
        for index, message in enumerate(conversational):
            expected = Role.USER if index % 2 == 0 else Role.ASSISTANT
            if message.role is not expected:
                raise ValueError("conversation roles must alternate user and assistant")

    def append(self, role: Role, content: str) -> "Conversation":
        """Return a new conversation with one validated message appended."""
        return Conversation((*self.messages, Message(role, content)))


def format_prompt(
    conversation: Conversation,
    *,
    add_assistant_prompt: bool = False,
) -> str:
    """Format role-delimited text for tokenizer encoding."""
    if add_assistant_prompt and (
        not conversation.messages or conversation.messages[-1].role is not Role.USER
    ):
        raise ValueError("assistant prompt requires a final user message")
    parts = [f"<|{message.role.value}|>\n{message.content}\n" for message in conversation.messages]
    if add_assistant_prompt:
        parts.append("<|assistant|>\n")
    return "".join(parts)
