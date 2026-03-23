"""Anthropic Claude backend implementing the LLM client interface."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from .base import (
    ResponseComplete,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


class AnthropicClient:
    """Wraps the Anthropic Messages API into the unified ``LLMClient`` interface.

    This client handles only the API-calling and streaming logic.  Persona
    management, conversation history, and tool *execution* remain in the
    orchestrator layer (``ClaudeClient``).
    """

    def __init__(self, api_key: str, model: str) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 1024,
        system: str | None = None,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a response from the Anthropic Messages API.

        Translates the native Anthropic SSE stream into a sequence of
        ``StreamEvent`` instances that the orchestrator can consume
        without any Anthropic-specific knowledge.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        if stop_sequences:
            kwargs["stop_sequences"] = stop_sequences

        current_tool_id: str = ""
        current_tool_name: str = ""

        async with self._client.messages.stream(**kwargs) as api_stream:
            async for event in api_stream:
                if event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        current_tool_id = event.content_block.id
                        current_tool_name = event.content_block.name
                        yield ToolCallStart(
                            id=current_tool_id,
                            name=current_tool_name,
                        )

                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield TextDelta(text=event.delta.text)
                    elif event.delta.type == "input_json_delta":
                        yield ToolCallDelta(
                            id=current_tool_id,
                            json_chunk=event.delta.partial_json,
                        )

                elif event.type == "content_block_stop":
                    if current_tool_name:
                        yield ToolCallEnd(id=current_tool_id)
                        current_tool_name = ""
                        current_tool_id = ""

                elif event.type == "message_delta":
                    yield ResponseComplete(
                        stop_reason=event.delta.stop_reason,
                    )
