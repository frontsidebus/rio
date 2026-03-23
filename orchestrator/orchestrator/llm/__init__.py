"""LLM client abstraction layer for multiple inference backends.

Provides a unified streaming interface (``LLMClient`` protocol) with concrete
implementations for Anthropic Claude and OpenAI-compatible local servers
(vLLM, SGLang, etc.).

Usage::

    from orchestrator.llm import create_llm_client
    from orchestrator.config import load_settings

    settings = load_settings()
    client = create_llm_client(settings)

    async for event in client.stream(messages, system=prompt):
        match event:
            case TextDelta(text=t):
                print(t, end="")
            case ToolCallStart(name=name):
                print(f"\\nCalling tool: {name}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import (
    LLMClient,
    ResponseComplete,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolDefinition,
)

if TYPE_CHECKING:
    from ..config import Settings

__all__ = [
    "AnthropicClient",
    "LLMClient",
    "OpenAICompatClient",
    "ResponseComplete",
    "StreamEvent",
    "TextDelta",
    "ToolCallDelta",
    "ToolCallEnd",
    "ToolCallStart",
    "ToolDefinition",
    "create_llm_client",
]


def create_llm_client(settings: Settings) -> LLMClient:
    """Factory: instantiate the appropriate LLM client based on config.

    Reads ``settings.llm_backend`` to decide which backend to use:

    - ``"anthropic"`` (default) — Anthropic Claude API.
    - ``"local"`` — OpenAI-compatible local server (vLLM / SGLang).

    Returns:
        An object satisfying the ``LLMClient`` protocol.

    Raises:
        ValueError: If ``llm_backend`` is not a recognised value.
    """
    backend = settings.llm_backend.lower().strip()

    if backend == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(
            api_key=settings.anthropic_api_key,
            model=settings.claude_model,
        )

    if backend == "local":
        from .openai_compat_client import OpenAICompatClient

        return OpenAICompatClient(
            base_url=settings.llm_api_url,
            model=settings.llm_model_local,
        )

    raise ValueError(
        f"Unknown LLM backend: {backend!r}. "
        f"Expected 'anthropic' or 'local'."
    )


# Lazy imports so the module doesn't pull in both SDKs at import time.
def __getattr__(name: str) -> type:
    if name == "AnthropicClient":
        from .anthropic_client import AnthropicClient

        return AnthropicClient
    if name == "OpenAICompatClient":
        from .openai_compat_client import OpenAICompatClient

        return OpenAICompatClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
