"""Tests for orchestrator.llm — LLM client abstraction layer.

Covers stream event types (base.py), factory function (create_llm_client),
AnthropicClient, OpenAICompatClient, and protocol compliance.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.llm.base import (
    LLMClient,
    ResponseComplete,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolDefinition,
)


# ===========================================================================
# 1. StreamEvent types (base.py)
# ===========================================================================


class TestTextDelta:
    """TextDelta dataclass correctness."""

    def test_construction(self) -> None:
        event = TextDelta(text="hello")
        assert event.text == "hello"

    def test_frozen(self) -> None:
        event = TextDelta(text="hello")
        with pytest.raises(dataclasses.FrozenInstanceError):
            event.text = "world"  # type: ignore[misc]

    def test_slots(self) -> None:
        assert hasattr(TextDelta, "__slots__")

    def test_fields(self) -> None:
        fields = {f.name for f in dataclasses.fields(TextDelta)}
        assert fields == {"text"}


class TestToolCallStart:
    """ToolCallStart dataclass correctness."""

    def test_construction(self) -> None:
        event = ToolCallStart(id="tc_001", name="get_sim_state")
        assert event.id == "tc_001"
        assert event.name == "get_sim_state"

    def test_frozen(self) -> None:
        event = ToolCallStart(id="tc_001", name="get_sim_state")
        with pytest.raises(dataclasses.FrozenInstanceError):
            event.id = "tc_002"  # type: ignore[misc]

    def test_fields(self) -> None:
        fields = {f.name for f in dataclasses.fields(ToolCallStart)}
        assert fields == {"id", "name"}


class TestToolCallDelta:
    """ToolCallDelta dataclass correctness."""

    def test_construction(self) -> None:
        event = ToolCallDelta(id="tc_001", json_chunk='{"key":')
        assert event.id == "tc_001"
        assert event.json_chunk == '{"key":'

    def test_frozen(self) -> None:
        event = ToolCallDelta(id="tc_001", json_chunk="x")
        with pytest.raises(dataclasses.FrozenInstanceError):
            event.json_chunk = "y"  # type: ignore[misc]

    def test_fields(self) -> None:
        fields = {f.name for f in dataclasses.fields(ToolCallDelta)}
        assert fields == {"id", "json_chunk"}


class TestToolCallEnd:
    """ToolCallEnd dataclass correctness."""

    def test_construction(self) -> None:
        event = ToolCallEnd(id="tc_001")
        assert event.id == "tc_001"

    def test_frozen(self) -> None:
        event = ToolCallEnd(id="tc_001")
        with pytest.raises(dataclasses.FrozenInstanceError):
            event.id = "tc_002"  # type: ignore[misc]

    def test_fields(self) -> None:
        fields = {f.name for f in dataclasses.fields(ToolCallEnd)}
        assert fields == {"id"}


class TestResponseComplete:
    """ResponseComplete dataclass correctness."""

    def test_construction_with_reason(self) -> None:
        event = ResponseComplete(stop_reason="end_turn")
        assert event.stop_reason == "end_turn"

    def test_construction_with_none(self) -> None:
        event = ResponseComplete(stop_reason=None)
        assert event.stop_reason is None

    def test_frozen(self) -> None:
        event = ResponseComplete(stop_reason="end_turn")
        with pytest.raises(dataclasses.FrozenInstanceError):
            event.stop_reason = "stop"  # type: ignore[misc]

    def test_fields(self) -> None:
        fields = {f.name for f in dataclasses.fields(ResponseComplete)}
        assert fields == {"stop_reason"}


class TestStreamEventUnion:
    """The StreamEvent union accepts all event variants."""

    def test_text_delta_is_stream_event(self) -> None:
        event: StreamEvent = TextDelta(text="hi")
        assert isinstance(event, TextDelta)

    def test_tool_call_start_is_stream_event(self) -> None:
        event: StreamEvent = ToolCallStart(id="1", name="t")
        assert isinstance(event, ToolCallStart)

    def test_tool_call_delta_is_stream_event(self) -> None:
        event: StreamEvent = ToolCallDelta(id="1", json_chunk="{")
        assert isinstance(event, ToolCallDelta)

    def test_tool_call_end_is_stream_event(self) -> None:
        event: StreamEvent = ToolCallEnd(id="1")
        assert isinstance(event, ToolCallEnd)

    def test_response_complete_is_stream_event(self) -> None:
        event: StreamEvent = ResponseComplete(stop_reason="stop")
        assert isinstance(event, ResponseComplete)


# ===========================================================================
# 2. Factory function (create_llm_client)
# ===========================================================================


class TestCreateLLMClient:
    """Test the create_llm_client factory in orchestrator.llm.__init__."""

    @pytest.fixture
    def _mock_settings(self) -> MagicMock:
        """A mock Settings with sensible defaults."""
        s = MagicMock()
        s.llm_backend = "anthropic"
        s.anthropic_api_key = "sk-ant-test"
        s.claude_model = "claude-sonnet-4-20250514"
        s.llm_api_url = "http://localhost:8081/v1"
        s.llm_model_local = "qwen3.5-35b-a3b-aviation"
        return s

    @patch("orchestrator.llm.anthropic_client.anthropic.AsyncAnthropic")
    def test_anthropic_backend(
        self, mock_anthropic_cls: MagicMock, _mock_settings: MagicMock
    ) -> None:
        from orchestrator.llm import create_llm_client

        _mock_settings.llm_backend = "anthropic"
        client = create_llm_client(_mock_settings)

        from orchestrator.llm.anthropic_client import AnthropicClient

        assert isinstance(client, AnthropicClient)

    @patch("orchestrator.llm.openai_compat_client.AsyncOpenAI")
    def test_local_backend(
        self, mock_openai_cls: MagicMock, _mock_settings: MagicMock
    ) -> None:
        from orchestrator.llm import create_llm_client

        _mock_settings.llm_backend = "local"
        client = create_llm_client(_mock_settings)

        from orchestrator.llm.openai_compat_client import OpenAICompatClient

        assert isinstance(client, OpenAICompatClient)

    def test_unknown_backend_raises_value_error(
        self, _mock_settings: MagicMock
    ) -> None:
        from orchestrator.llm import create_llm_client

        _mock_settings.llm_backend = "unknown"
        with pytest.raises(ValueError, match="Unknown LLM backend"):
            create_llm_client(_mock_settings)

    @patch("orchestrator.llm.anthropic_client.anthropic.AsyncAnthropic")
    def test_case_insensitive_anthropic(
        self, mock_cls: MagicMock, _mock_settings: MagicMock
    ) -> None:
        from orchestrator.llm import create_llm_client

        _mock_settings.llm_backend = "ANTHROPIC"
        client = create_llm_client(_mock_settings)

        from orchestrator.llm.anthropic_client import AnthropicClient

        assert isinstance(client, AnthropicClient)

    @patch("orchestrator.llm.openai_compat_client.AsyncOpenAI")
    def test_whitespace_stripped_local(
        self, mock_cls: MagicMock, _mock_settings: MagicMock
    ) -> None:
        from orchestrator.llm import create_llm_client

        _mock_settings.llm_backend = "  local  "
        client = create_llm_client(_mock_settings)

        from orchestrator.llm.openai_compat_client import OpenAICompatClient

        assert isinstance(client, OpenAICompatClient)

    @patch("orchestrator.llm.anthropic_client.anthropic.AsyncAnthropic")
    def test_case_and_whitespace_combined(
        self, mock_cls: MagicMock, _mock_settings: MagicMock
    ) -> None:
        from orchestrator.llm import create_llm_client

        _mock_settings.llm_backend = "  Anthropic  "
        client = create_llm_client(_mock_settings)

        from orchestrator.llm.anthropic_client import AnthropicClient

        assert isinstance(client, AnthropicClient)


# ===========================================================================
# 3. AnthropicClient
# ===========================================================================


class TestAnthropicClient:
    """Tests for the Anthropic Claude backend client."""

    @pytest.fixture
    def client(self) -> Any:
        with patch("orchestrator.llm.anthropic_client.anthropic.AsyncAnthropic") as mock_cls:
            from orchestrator.llm.anthropic_client import AnthropicClient

            c = AnthropicClient(api_key="sk-ant-test", model="claude-sonnet-4-20250514")
            c._mock_anthropic = mock_cls  # stash for assertions
            return c

    def test_model_property(self, client: Any) -> None:
        assert client.model == "claude-sonnet-4-20250514"

    def test_constructor_creates_async_client(self, client: Any) -> None:
        client._mock_anthropic.assert_called_once_with(api_key="sk-ant-test")

    @pytest.mark.asyncio
    async def test_stream_text_delta(self, client: Any) -> None:
        """Anthropic text_delta events map to TextDelta."""
        delta = MagicMock()
        delta.type = "text_delta"
        delta.text = "Hello, Captain"

        content_block = MagicMock()
        content_block.type = "text"

        event_start = MagicMock()
        event_start.type = "content_block_start"
        event_start.content_block = content_block

        event_delta = MagicMock()
        event_delta.type = "content_block_delta"
        event_delta.delta = delta

        event_stop = MagicMock()
        event_stop.type = "content_block_stop"

        msg_delta = MagicMock()
        msg_delta.type = "message_delta"
        msg_delta.delta = MagicMock(stop_reason="end_turn")

        # Build async iterator for the stream context manager
        async def mock_stream_iter():
            for e in [event_start, event_delta, event_stop, msg_delta]:
                yield e

        stream_ctx = AsyncMock()
        stream_ctx.__aiter__ = mock_stream_iter
        stream_ctx.__aenter__ = AsyncMock(return_value=stream_ctx)
        stream_ctx.__aexit__ = AsyncMock(return_value=False)

        client._client.messages.stream = MagicMock(return_value=stream_ctx)

        events = []
        async for ev in client.stream([{"role": "user", "content": "hello"}]):
            events.append(ev)

        # Should have TextDelta + ResponseComplete
        text_events = [e for e in events if isinstance(e, TextDelta)]
        assert len(text_events) == 1
        assert text_events[0].text == "Hello, Captain"

        complete_events = [e for e in events if isinstance(e, ResponseComplete)]
        assert len(complete_events) == 1
        assert complete_events[0].stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_stream_tool_call_sequence(self, client: Any) -> None:
        """Tool use events map to ToolCallStart -> ToolCallDelta -> ToolCallEnd."""
        # content_block_start with tool_use
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "toolu_123"
        tool_block.name = "get_sim_state"

        ev_start = MagicMock()
        ev_start.type = "content_block_start"
        ev_start.content_block = tool_block

        # content_block_delta with input_json_delta
        json_delta = MagicMock()
        json_delta.type = "input_json_delta"
        json_delta.partial_json = '{"fields":'

        ev_delta1 = MagicMock()
        ev_delta1.type = "content_block_delta"
        ev_delta1.delta = json_delta

        json_delta2 = MagicMock()
        json_delta2.type = "input_json_delta"
        json_delta2.partial_json = '["alt"]}'

        ev_delta2 = MagicMock()
        ev_delta2.type = "content_block_delta"
        ev_delta2.delta = json_delta2

        # content_block_stop
        ev_stop = MagicMock()
        ev_stop.type = "content_block_stop"

        # message_delta
        msg_delta = MagicMock()
        msg_delta.type = "message_delta"
        msg_delta.delta = MagicMock(stop_reason="tool_use")

        async def mock_stream_iter():
            for e in [ev_start, ev_delta1, ev_delta2, ev_stop, msg_delta]:
                yield e

        stream_ctx = AsyncMock()
        stream_ctx.__aiter__ = mock_stream_iter
        stream_ctx.__aenter__ = AsyncMock(return_value=stream_ctx)
        stream_ctx.__aexit__ = AsyncMock(return_value=False)

        client._client.messages.stream = MagicMock(return_value=stream_ctx)

        events = []
        async for ev in client.stream(
            [{"role": "user", "content": "what's my alt?"}],
            tools=[{"name": "get_sim_state", "description": "Get state", "input_schema": {}}],
        ):
            events.append(ev)

        # Verify sequence: ToolCallStart, ToolCallDelta, ToolCallDelta, ToolCallEnd, ResponseComplete
        assert isinstance(events[0], ToolCallStart)
        assert events[0].id == "toolu_123"
        assert events[0].name == "get_sim_state"

        assert isinstance(events[1], ToolCallDelta)
        assert events[1].json_chunk == '{"fields":'

        assert isinstance(events[2], ToolCallDelta)
        assert events[2].json_chunk == '["alt"]}'

        assert isinstance(events[3], ToolCallEnd)
        assert events[3].id == "toolu_123"

        assert isinstance(events[4], ResponseComplete)
        assert events[4].stop_reason == "tool_use"

    @pytest.mark.asyncio
    async def test_stream_stop_reason_max_tokens(self, client: Any) -> None:
        """max_tokens stop reason is passed through."""
        msg_delta = MagicMock()
        msg_delta.type = "message_delta"
        msg_delta.delta = MagicMock(stop_reason="max_tokens")

        async def mock_stream_iter():
            yield msg_delta

        stream_ctx = AsyncMock()
        stream_ctx.__aiter__ = mock_stream_iter
        stream_ctx.__aenter__ = AsyncMock(return_value=stream_ctx)
        stream_ctx.__aexit__ = AsyncMock(return_value=False)

        client._client.messages.stream = MagicMock(return_value=stream_ctx)

        events = []
        async for ev in client.stream([{"role": "user", "content": "x"}]):
            events.append(ev)

        assert len(events) == 1
        assert isinstance(events[0], ResponseComplete)
        assert events[0].stop_reason == "max_tokens"

    @pytest.mark.asyncio
    async def test_stream_passes_kwargs(self, client: Any) -> None:
        """Verify system, tools, stop_sequences, and max_tokens are passed to the API."""
        async def mock_stream_iter():
            msg_delta = MagicMock()
            msg_delta.type = "message_delta"
            msg_delta.delta = MagicMock(stop_reason="end_turn")
            yield msg_delta

        stream_ctx = AsyncMock()
        stream_ctx.__aiter__ = mock_stream_iter
        stream_ctx.__aenter__ = AsyncMock(return_value=stream_ctx)
        stream_ctx.__aexit__ = AsyncMock(return_value=False)

        client._client.messages.stream = MagicMock(return_value=stream_ctx)

        messages = [{"role": "user", "content": "hi"}]
        tools = [{"name": "t", "description": "d", "input_schema": {}}]

        events = []
        async for ev in client.stream(
            messages,
            tools=tools,
            max_tokens=512,
            system="You are MERLIN.",
            stop_sequences=["Over."],
        ):
            events.append(ev)

        call_kwargs = client._client.messages.stream.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["max_tokens"] == 512
        assert call_kwargs["messages"] == messages
        assert call_kwargs["system"] == "You are MERLIN."
        assert call_kwargs["tools"] == tools
        assert call_kwargs["stop_sequences"] == ["Over."]

    @pytest.mark.asyncio
    async def test_stream_omits_optional_kwargs_when_none(self, client: Any) -> None:
        """When system, tools, stop_sequences are None, they are not passed."""
        async def mock_stream_iter():
            msg_delta = MagicMock()
            msg_delta.type = "message_delta"
            msg_delta.delta = MagicMock(stop_reason="end_turn")
            yield msg_delta

        stream_ctx = AsyncMock()
        stream_ctx.__aiter__ = mock_stream_iter
        stream_ctx.__aenter__ = AsyncMock(return_value=stream_ctx)
        stream_ctx.__aexit__ = AsyncMock(return_value=False)

        client._client.messages.stream = MagicMock(return_value=stream_ctx)

        events = []
        async for ev in client.stream([{"role": "user", "content": "hi"}]):
            events.append(ev)

        call_kwargs = client._client.messages.stream.call_args[1]
        assert "system" not in call_kwargs
        assert "tools" not in call_kwargs
        assert "stop_sequences" not in call_kwargs


# ===========================================================================
# 4. OpenAICompatClient
# ===========================================================================


class TestConvertTools:
    """Test _convert_tools: Anthropic tool schema to OpenAI function format."""

    def test_basic_conversion(self) -> None:
        from orchestrator.llm.openai_compat_client import _convert_tools

        tools: list[ToolDefinition] = [
            {
                "name": "get_sim_state",
                "description": "Get current sim state",
                "input_schema": {
                    "type": "object",
                    "properties": {"fields": {"type": "array"}},
                    "required": [],
                },
            }
        ]
        result = _convert_tools(tools)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_sim_state"
        assert result[0]["function"]["description"] == "Get current sim state"
        assert result[0]["function"]["parameters"]["type"] == "object"

    def test_missing_description_defaults_empty(self) -> None:
        from orchestrator.llm.openai_compat_client import _convert_tools

        tools = [{"name": "test_tool", "input_schema": {"type": "object", "properties": {}}}]
        result = _convert_tools(tools)

        assert result[0]["function"]["description"] == ""

    def test_missing_input_schema_defaults(self) -> None:
        from orchestrator.llm.openai_compat_client import _convert_tools

        tools = [{"name": "test_tool", "description": "A tool"}]
        result = _convert_tools(tools)

        assert result[0]["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_multiple_tools(self) -> None:
        from orchestrator.llm.openai_compat_client import _convert_tools

        tools = [
            {"name": "tool_a", "description": "A", "input_schema": {"type": "object"}},
            {"name": "tool_b", "description": "B", "input_schema": {"type": "object"}},
        ]
        result = _convert_tools(tools)

        assert len(result) == 2
        assert result[0]["function"]["name"] == "tool_a"
        assert result[1]["function"]["name"] == "tool_b"


class TestConvertMessages:
    """Test _convert_messages: Anthropic message format to OpenAI chat format."""

    def test_system_prompt_injected(self) -> None:
        from orchestrator.llm.openai_compat_client import _convert_messages

        result = _convert_messages([], system="You are MERLIN.")
        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are MERLIN."

    def test_no_system_when_none(self) -> None:
        from orchestrator.llm.openai_compat_client import _convert_messages

        result = _convert_messages([], system=None)
        assert len(result) == 0

    def test_simple_user_text_message(self) -> None:
        from orchestrator.llm.openai_compat_client import _convert_messages

        messages = [{"role": "user", "content": "What's my altitude?"}]
        result = _convert_messages(messages, system=None)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "What's my altitude?"

    def test_simple_assistant_text_message(self) -> None:
        from orchestrator.llm.openai_compat_client import _convert_messages

        messages = [{"role": "assistant", "content": "You're at 6500 feet, Captain."}]
        result = _convert_messages(messages, system=None)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "You're at 6500 feet, Captain."

    def test_tool_use_blocks_converted_to_tool_calls(self) -> None:
        from orchestrator.llm.openai_compat_client import _convert_messages

        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "get_sim_state",
                        "input": {"fields": ["altitude"]},
                    },
                ],
            }
        ]
        result = _convert_messages(messages, system=None)

        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Let me check."
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "toolu_123"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_sim_state"
        assert json.loads(tc["function"]["arguments"]) == {"fields": ["altitude"]}

    def test_tool_result_blocks_converted_to_tool_messages(self) -> None:
        from orchestrator.llm.openai_compat_client import _convert_messages

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": '{"altitude": 6500}',
                    }
                ],
            }
        ]
        result = _convert_messages(messages, system=None)

        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "toolu_123"
        assert result[0]["content"] == '{"altitude": 6500}'

    def test_tool_result_with_dict_content_serialized(self) -> None:
        from orchestrator.llm.openai_compat_client import _convert_messages

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_456",
                        "content": {"altitude": 6500, "heading": 270},
                    }
                ],
            }
        ]
        result = _convert_messages(messages, system=None)

        assert result[0]["role"] == "tool"
        parsed = json.loads(result[0]["content"])
        assert parsed == {"altitude": 6500, "heading": 270}

    def test_multiple_tool_results_become_separate_messages(self) -> None:
        from orchestrator.llm.openai_compat_client import _convert_messages

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "result1"},
                    {"type": "tool_result", "tool_use_id": "t2", "content": "result2"},
                ],
            }
        ]
        result = _convert_messages(messages, system=None)

        assert len(result) == 2
        assert result[0]["tool_call_id"] == "t1"
        assert result[1]["tool_call_id"] == "t2"

    def test_image_content_converted_to_image_url(self) -> None:
        from orchestrator.llm.openai_compat_client import _convert_messages

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBORw0KGgo=",
                        },
                    },
                ],
            }
        ]
        result = _convert_messages(messages, system=None)

        assert len(result) == 1
        content = result[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "What do you see?"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "data:image/png;base64,iVBORw0KGgo="

    def test_assistant_tool_use_only_no_text(self) -> None:
        """Assistant message with only tool_use blocks, no text."""
        from orchestrator.llm.openai_compat_client import _convert_messages

        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_789",
                        "name": "lookup_airport",
                        "input": {"identifier": "KJFK"},
                    },
                ],
            }
        ]
        result = _convert_messages(messages, system=None)

        assert result[0]["content"] is None
        assert len(result[0]["tool_calls"]) == 1

    def test_full_conversation_with_system(self) -> None:
        """End-to-end conversion of a multi-turn conversation."""
        from orchestrator.llm.openai_compat_client import _convert_messages

        messages = [
            {"role": "user", "content": "Check my state"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Checking..."},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "get_sim_state",
                        "input": {},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": '{"alt": 6500}'},
                ],
            },
            {"role": "assistant", "content": "You're at 6500 feet."},
        ]
        result = _convert_messages(messages, system="You are MERLIN.")

        assert len(result) == 5  # system + 4 messages
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert "tool_calls" in result[2]
        assert result[3]["role"] == "tool"
        assert result[4]["role"] == "assistant"


class TestOpenAICompatClient:
    """Tests for the OpenAI-compatible local inference client."""

    @pytest.fixture
    def client(self) -> Any:
        with patch("orchestrator.llm.openai_compat_client.AsyncOpenAI") as mock_cls:
            from orchestrator.llm.openai_compat_client import OpenAICompatClient

            c = OpenAICompatClient(
                base_url="http://localhost:8081/v1",
                model="qwen3.5-35b-a3b-aviation",
            )
            c._mock_openai = mock_cls
            return c

    def test_model_property(self, client: Any) -> None:
        assert client.model == "qwen3.5-35b-a3b-aviation"

    def test_constructor_passes_base_url(self, client: Any) -> None:
        client._mock_openai.assert_called_once_with(
            base_url="http://localhost:8081/v1",
            api_key="not-needed",
        )

    @pytest.mark.asyncio
    async def test_stream_text_delta(self, client: Any) -> None:
        """OpenAI text content chunks map to TextDelta."""
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock(content="Hello", tool_calls=None)
        chunk1.choices[0].finish_reason = None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta = MagicMock(content=", Captain", tool_calls=None)
        chunk2.choices[0].finish_reason = None

        chunk_final = MagicMock()
        chunk_final.choices = [MagicMock()]
        chunk_final.choices[0].delta = MagicMock(content=None, tool_calls=None)
        chunk_final.choices[0].finish_reason = "stop"

        async def mock_response_iter():
            for c in [chunk1, chunk2, chunk_final]:
                yield c

        mock_response = mock_response_iter()
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        events = []
        async for ev in client.stream([{"role": "user", "content": "hi"}]):
            events.append(ev)

        text_events = [e for e in events if isinstance(e, TextDelta)]
        assert len(text_events) == 2
        assert text_events[0].text == "Hello"
        assert text_events[1].text == ", Captain"

        complete = [e for e in events if isinstance(e, ResponseComplete)]
        assert len(complete) == 1
        assert complete[0].stop_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_tool_call_sequence(self, client: Any) -> None:
        """OpenAI tool call chunks map to ToolCallStart/Delta/End."""
        # Chunk 1: tool call start
        tc1 = MagicMock()
        tc1.index = 0
        tc1.id = "call_abc123"
        tc1.function = MagicMock(name="get_sim_state", arguments="")
        tc1.function.name = "get_sim_state"
        tc1.function.arguments = ""

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock(content=None, tool_calls=[tc1])
        chunk1.choices[0].finish_reason = None

        # Chunk 2: argument fragment
        tc2 = MagicMock()
        tc2.index = 0
        tc2.id = None
        tc2.function = MagicMock()
        tc2.function.name = None
        tc2.function.arguments = '{"fields":'

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta = MagicMock(content=None, tool_calls=[tc2])
        chunk2.choices[0].finish_reason = None

        # Chunk 3: argument fragment
        tc3 = MagicMock()
        tc3.index = 0
        tc3.id = None
        tc3.function = MagicMock()
        tc3.function.name = None
        tc3.function.arguments = '["alt"]}'

        chunk3 = MagicMock()
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta = MagicMock(content=None, tool_calls=[tc3])
        chunk3.choices[0].finish_reason = None

        # Final chunk with finish_reason
        chunk_final = MagicMock()
        chunk_final.choices = [MagicMock()]
        chunk_final.choices[0].delta = MagicMock(content=None, tool_calls=None)
        chunk_final.choices[0].finish_reason = "tool_calls"

        async def mock_response_iter():
            for c in [chunk1, chunk2, chunk3, chunk_final]:
                yield c

        client._client.chat.completions.create = AsyncMock(
            return_value=mock_response_iter()
        )

        events = []
        async for ev in client.stream(
            [{"role": "user", "content": "check state"}],
            tools=[{"name": "get_sim_state", "description": "d", "input_schema": {}}],
        ):
            events.append(ev)

        assert isinstance(events[0], ToolCallStart)
        assert events[0].id == "call_abc123"
        assert events[0].name == "get_sim_state"

        assert isinstance(events[1], ToolCallDelta)
        assert events[1].json_chunk == '{"fields":'

        assert isinstance(events[2], ToolCallDelta)
        assert events[2].json_chunk == '["alt"]}'

        assert isinstance(events[3], ToolCallEnd)
        assert events[3].id == "call_abc123"

        assert isinstance(events[4], ResponseComplete)
        assert events[4].stop_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_stream_tool_call_index_tracking(self, client: Any) -> None:
        """Multiple tool calls with different indices are tracked separately."""
        # First tool call on index 0
        tc_a = MagicMock()
        tc_a.index = 0
        tc_a.id = "call_aaa"
        tc_a.function = MagicMock()
        tc_a.function.name = "get_sim_state"
        tc_a.function.arguments = "{}"

        # Second tool call on index 1
        tc_b = MagicMock()
        tc_b.index = 1
        tc_b.id = "call_bbb"
        tc_b.function = MagicMock()
        tc_b.function.name = "lookup_airport"
        tc_b.function.arguments = '{"identifier":"KJFK"}'

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock(content=None, tool_calls=[tc_a])
        chunk1.choices[0].finish_reason = None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta = MagicMock(content=None, tool_calls=[tc_b])
        chunk2.choices[0].finish_reason = None

        chunk_final = MagicMock()
        chunk_final.choices = [MagicMock()]
        chunk_final.choices[0].delta = MagicMock(content=None, tool_calls=None)
        chunk_final.choices[0].finish_reason = "tool_calls"

        async def mock_response_iter():
            for c in [chunk1, chunk2, chunk_final]:
                yield c

        client._client.chat.completions.create = AsyncMock(
            return_value=mock_response_iter()
        )

        events = []
        async for ev in client.stream(
            [{"role": "user", "content": "plan"}],
            tools=[
                {"name": "get_sim_state", "description": "d", "input_schema": {}},
                {"name": "lookup_airport", "description": "d", "input_schema": {}},
            ],
        ):
            events.append(ev)

        starts = [e for e in events if isinstance(e, ToolCallStart)]
        assert len(starts) == 2
        assert starts[0].name == "get_sim_state"
        assert starts[1].name == "lookup_airport"

        ends = [e for e in events if isinstance(e, ToolCallEnd)]
        assert len(ends) == 2
        assert ends[0].id == "call_aaa"
        assert ends[1].id == "call_bbb"

    @pytest.mark.asyncio
    async def test_stream_generates_synthetic_id_when_missing(self, client: Any) -> None:
        """When the server doesn't provide a tool call ID, a synthetic one is generated."""
        tc = MagicMock()
        tc.index = 0
        tc.id = None  # No ID from server
        tc.function = MagicMock()
        tc.function.name = "test_tool"
        tc.function.arguments = ""

        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock(content=None, tool_calls=[tc])
        chunk.choices[0].finish_reason = None

        chunk_final = MagicMock()
        chunk_final.choices = [MagicMock()]
        chunk_final.choices[0].delta = MagicMock(content=None, tool_calls=None)
        chunk_final.choices[0].finish_reason = "tool_calls"

        async def mock_response_iter():
            for c in [chunk, chunk_final]:
                yield c

        client._client.chat.completions.create = AsyncMock(
            return_value=mock_response_iter()
        )

        events = []
        async for ev in client.stream([{"role": "user", "content": "x"}]):
            events.append(ev)

        start = [e for e in events if isinstance(e, ToolCallStart)][0]
        assert start.id.startswith("call_")
        assert len(start.id) > 5  # "call_" + uuid hex

    @pytest.mark.asyncio
    async def test_stream_empty_choices_skipped(self, client: Any) -> None:
        """Chunks with empty choices are gracefully skipped."""
        empty_chunk = MagicMock()
        empty_chunk.choices = []

        chunk_final = MagicMock()
        chunk_final.choices = [MagicMock()]
        chunk_final.choices[0].delta = MagicMock(content=None, tool_calls=None)
        chunk_final.choices[0].finish_reason = "stop"

        async def mock_response_iter():
            for c in [empty_chunk, chunk_final]:
                yield c

        client._client.chat.completions.create = AsyncMock(
            return_value=mock_response_iter()
        )

        events = []
        async for ev in client.stream([{"role": "user", "content": "x"}]):
            events.append(ev)

        assert len(events) == 1
        assert isinstance(events[0], ResponseComplete)

    @pytest.mark.asyncio
    async def test_stream_passes_stop_sequences_as_stop(self, client: Any) -> None:
        """stop_sequences parameter is mapped to OpenAI 'stop' parameter."""
        async def mock_response_iter():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock(content=None, tool_calls=None)
            chunk.choices[0].finish_reason = "stop"
            yield chunk

        client._client.chat.completions.create = AsyncMock(
            return_value=mock_response_iter()
        )

        events = []
        async for ev in client.stream(
            [{"role": "user", "content": "hi"}],
            stop_sequences=["Over.", "Roger."],
        ):
            events.append(ev)

        call_kwargs = client._client.chat.completions.create.call_args[1]
        assert call_kwargs["stop"] == ["Over.", "Roger."]


# ===========================================================================
# 5. Protocol compliance
# ===========================================================================


class TestProtocolCompliance:
    """Both clients satisfy isinstance checks against the LLMClient protocol."""

    @patch("orchestrator.llm.anthropic_client.anthropic.AsyncAnthropic")
    def test_anthropic_client_is_llm_client(self, mock_cls: MagicMock) -> None:
        from orchestrator.llm.anthropic_client import AnthropicClient

        client = AnthropicClient(api_key="sk-test", model="test-model")
        assert isinstance(client, LLMClient)

    @patch("orchestrator.llm.openai_compat_client.AsyncOpenAI")
    def test_openai_compat_client_is_llm_client(self, mock_cls: MagicMock) -> None:
        from orchestrator.llm.openai_compat_client import OpenAICompatClient

        client = OpenAICompatClient(base_url="http://localhost:8081/v1", model="test")
        assert isinstance(client, LLMClient)

    def test_llm_client_is_runtime_checkable(self) -> None:
        """The LLMClient protocol is decorated with @runtime_checkable."""
        assert hasattr(LLMClient, "__protocol_attrs__") or hasattr(
            LLMClient, "_is_runtime_protocol"
        )
        # The canonical check: isinstance should work without error
        assert not isinstance("not a client", LLMClient)
