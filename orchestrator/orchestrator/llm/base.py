"""Base protocol and event types for the LLM client abstraction layer."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Stream event types — a unified vocabulary for LLM streaming responses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A chunk of generated text."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolCallStart:
    """Signals the beginning of a tool call."""

    id: str
    name: str


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    """An incremental JSON fragment for a tool call's arguments."""

    id: str
    json_chunk: str


@dataclass(frozen=True, slots=True)
class ToolCallEnd:
    """Signals the end of a tool call (arguments are now complete)."""

    id: str


@dataclass(frozen=True, slots=True)
class ResponseComplete:
    """The model has finished generating.

    Attributes:
        stop_reason: Backend-specific stop reason string.  Common values include
            ``"end_turn"``, ``"tool_use"``, ``"stop"`` (OpenAI), ``"max_tokens"``,
            and ``"stop_sequence"``.
    """

    stop_reason: str | None


# Union of all event types yielded during streaming.
StreamEvent = TextDelta | ToolCallStart | ToolCallDelta | ToolCallEnd | ResponseComplete


# ---------------------------------------------------------------------------
# Tool definition schema — backend-agnostic representation.
# ---------------------------------------------------------------------------
# We re-use the Anthropic-style dict format already defined in claude_client.py:
#   {"name": str, "description": str, "input_schema": {...}}
# Each backend client is responsible for converting this into its native
# format (e.g. OpenAI ``tools`` array with ``function`` wrappers).
ToolDefinition = dict[str, Any]


# ---------------------------------------------------------------------------
# LLM Client protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMClient(Protocol):
    """Protocol that all LLM backend clients must satisfy."""

    @property
    def model(self) -> str:
        """Return the model identifier in use."""
        ...

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 1024,
        system: str | None = None,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a response from the LLM, yielding ``StreamEvent`` instances.

        Parameters:
            messages: Conversation history in Anthropic message format
                (``{"role": "user"|"assistant", "content": ...}``).
            tools: Optional tool definitions (Anthropic schema).
            max_tokens: Maximum number of tokens to generate.
            system: Optional system prompt text.
            stop_sequences: Optional sequences that halt generation.

        Yields:
            ``StreamEvent`` instances in order: text deltas, tool call
            lifecycle events, and finally a ``ResponseComplete``.
        """
        ...
        # Make the protocol method a valid async iterator stub
        if False:  # pragma: no cover
            yield  # type: ignore[misc]
