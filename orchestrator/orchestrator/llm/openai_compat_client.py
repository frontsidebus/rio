"""OpenAI-compatible backend for local inference (vLLM / SGLang).

Implements the ``LLMClient`` protocol using the ``openai`` Python library
pointed at a local server endpoint.  Tool calls use the standard OpenAI
function-calling streaming format.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

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


def _convert_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Convert Anthropic-style tool definitions to OpenAI function-calling format.

    Anthropic format::

        {"name": "...", "description": "...", "input_schema": {...}}

    OpenAI format::

        {"type": "function", "function": {"name": "...", "description": "...",
         "parameters": {...}}}
    """
    converted: list[dict[str, Any]] = []
    for tool in tools:
        converted.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return converted


def _convert_messages(
    messages: list[dict[str, Any]],
    system: str | None,
) -> list[dict[str, Any]]:
    """Convert Anthropic-style messages to OpenAI chat format.

    Key differences handled:
    - Anthropic uses a separate ``system`` parameter; OpenAI uses a system message.
    - Anthropic content can be a list of typed blocks; OpenAI expects a string
      for simple text or a list of content parts for multimodal.
    - Anthropic ``tool_use`` / ``tool_result`` blocks map to OpenAI
      ``tool_calls`` on assistant messages and ``tool`` role messages.
    """
    converted: list[dict[str, Any]] = []

    if system:
        converted.append({"role": "system", "content": system})

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # Simple string content — pass through
        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue

        # Content is a list of typed blocks (Anthropic format)
        if isinstance(content, list):
            # Check if this is a tool_result turn (user role with tool results)
            if role == "user" and content and isinstance(content[0], dict):
                if content[0].get("type") == "tool_result":
                    for block in content:
                        converted.append({
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": (
                                block["content"]
                                if isinstance(block["content"], str)
                                else json.dumps(block["content"])
                            ),
                        })
                    continue

            # Assistant turn potentially containing text + tool_use blocks
            if role == "assistant":
                text_parts: list[str] = []
                tool_calls: list[dict[str, Any]] = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block["text"])
                        elif block.get("type") == "tool_use":
                            tool_calls.append({
                                "id": block["id"],
                                "type": "function",
                                "function": {
                                    "name": block["name"],
                                    "arguments": json.dumps(block["input"]),
                                },
                            })

                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else None,
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                converted.append(assistant_msg)
                continue

            # User turn with mixed content (text + images)
            if role == "user":
                oai_content: list[dict[str, Any]] = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            oai_content.append({
                                "type": "text",
                                "text": block["text"],
                            })
                        elif block.get("type") == "image":
                            source = block.get("source", {})
                            oai_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        f"data:{source.get('media_type', 'image/jpeg')};"
                                        f"base64,{source.get('data', '')}"
                                    ),
                                },
                            })
                converted.append({"role": "user", "content": oai_content})
                continue

        # Fallback — pass through as-is
        converted.append({"role": role, "content": content})

    return converted


class OpenAICompatClient:
    """OpenAI-compatible LLM client for local vLLM/SGLang servers.

    Uses the ``openai`` library's ``AsyncOpenAI`` client pointed at a local
    endpoint.  The server must support the ``/v1/chat/completions`` endpoint
    with streaming and (optionally) function calling.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "not-needed") -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
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
        """Stream a response from an OpenAI-compatible endpoint.

        Translates the OpenAI streaming chunk format into unified
        ``StreamEvent`` instances.
        """
        oai_messages = _convert_messages(messages, system)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": oai_messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = _convert_tools(tools)
        if stop_sequences:
            kwargs["stop"] = stop_sequences

        # Track active tool calls by index within a single response.
        # OpenAI streams tool calls identified by ``index``; we map each
        # index to a synthetic ID and name for our event model.
        active_tools: dict[int, str] = {}  # index -> tool_call_id
        active_tool_names: dict[int, str] = {}  # index -> function name

        response = await self._client.chat.completions.create(**kwargs)

        async for chunk in response:
            choice = chunk.choices[0] if chunk.choices else None
            if choice is None:
                continue

            delta = choice.delta

            # Text content
            if delta.content:
                yield TextDelta(text=delta.content)

            # Tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index

                    # New tool call starting
                    if idx not in active_tools:
                        tool_id = (
                            tc.id
                            if tc.id
                            else f"call_{uuid.uuid4().hex[:24]}"
                        )
                        tool_name = tc.function.name if tc.function and tc.function.name else ""
                        active_tools[idx] = tool_id
                        active_tool_names[idx] = tool_name
                        yield ToolCallStart(id=tool_id, name=tool_name)

                    # Argument fragment
                    if tc.function and tc.function.arguments:
                        yield ToolCallDelta(
                            id=active_tools[idx],
                            json_chunk=tc.function.arguments,
                        )

            # Finish reason indicates the response (or tool call) is done
            if choice.finish_reason is not None:
                # Close any open tool calls
                for idx in sorted(active_tools):
                    yield ToolCallEnd(id=active_tools[idx])
                active_tools.clear()
                active_tool_names.clear()

                yield ResponseComplete(stop_reason=choice.finish_reason)
