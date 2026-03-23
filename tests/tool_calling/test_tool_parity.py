"""
RIO Phase 1.3: Tool Calling Parity Test Harness

Validates that the local LLM selects the correct tools and produces
valid arguments compared to the Claude API baseline.

Run with:
    pytest tests/tool_calling/test_tool_parity.py -v

For integration tests against a live backend:
    pytest tests/tool_calling/test_tool_parity.py -v -m integration

Environment variables:
    TOOL_PARITY_BACKEND   - "anthropic" (default) or "openai"
    TOOL_PARITY_BASE_URL  - OpenAI-compatible base URL (e.g., http://localhost:8081/v1)
    TOOL_PARITY_MODEL     - Model name to use (default: claude-sonnet-4-20250514 for Anthropic)
    TOOL_PARITY_API_KEY   - API key for the chosen backend
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SCENARIOS_FILE = FIXTURES_DIR / "scenarios.json"
MOCK_RESPONSES_FILE = FIXTURES_DIR / "mock_tool_responses.json"

# Valid tool names from MERLIN's TOOL_DEFINITIONS
VALID_TOOL_NAMES = frozenset({
    "get_sim_state",
    "lookup_airport",
    "search_manual",
    "get_checklist",
    "create_flight_plan",
})

# Tool argument schemas for validation
TOOL_ARG_SCHEMAS: dict[str, dict[str, Any]] = {
    "get_sim_state": {
        "required": [],
        "properties": {},
    },
    "lookup_airport": {
        "required": ["identifier"],
        "properties": {
            "identifier": {"type": "string"},
        },
    },
    "search_manual": {
        "required": ["query"],
        "properties": {
            "query": {"type": "string"},
        },
    },
    "get_checklist": {
        "required": ["phase"],
        "properties": {
            "phase": {"type": "string"},
        },
    },
    "create_flight_plan": {
        "required": ["departure", "destination"],
        "properties": {
            "departure": {"type": "string"},
            "destination": {"type": "string"},
            "altitude": {"type": "integer"},
            "route": {"type": "string"},
        },
    },
}

# MERLIN tool definitions in OpenAI function-calling format
OPENAI_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_sim_state",
            "description": (
                "Retrieve the current simulator state including position, attitude, "
                "speeds, engine parameters, autopilot, radios, fuel, weather, and "
                "surface states."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_airport",
            "description": (
                "Look up airport information by ICAO or FAA identifier. Returns name, "
                "location, elevation, and basic facility data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": "Airport ICAO or FAA identifier (e.g., KJFK, KLAX, ORL)",
                    },
                },
                "required": ["identifier"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_manual",
            "description": (
                "Search the aircraft operating manual and aviation knowledge base. "
                "Use this to look up procedures, limitations, V-speeds, systems "
                "descriptions, or any aircraft-specific information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query describing what to look up",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_checklist",
            "description": (
                "Get the appropriate checklist for a given flight phase. Returns "
                "phase-specific checklist items, preferring aircraft-specific "
                "checklists when available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "phase": {
                        "type": "string",
                        "description": (
                            "Flight phase (PREFLIGHT, TAXI, TAKEOFF, CLIMB, CRUISE, "
                            "DESCENT, APPROACH, LANDING, LANDED)"
                        ),
                    },
                },
                "required": ["phase"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_flight_plan",
            "description": (
                "Create a basic flight plan between two airports. Returns a draft "
                "route structure with departure, destination, and waypoints."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "departure": {
                        "type": "string",
                        "description": "Departure airport identifier",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination airport identifier",
                    },
                    "altitude": {
                        "type": "integer",
                        "description": "Planned cruise altitude in feet MSL",
                    },
                    "route": {
                        "type": "string",
                        "description": "Optional route waypoints separated by spaces",
                    },
                },
                "required": ["departure", "destination"],
            },
        },
    },
]

# MERLIN tool definitions in Anthropic format (from claude_client.py)
ANTHROPIC_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_sim_state",
        "description": (
            "Retrieve the current simulator state including position, attitude, speeds, "
            "engine parameters, autopilot, radios, fuel, weather, and surface states."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "lookup_airport",
        "description": (
            "Look up airport information by ICAO or FAA identifier. Returns name, location, "
            "elevation, and basic facility data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "identifier": {
                    "type": "string",
                    "description": "Airport ICAO or FAA identifier (e.g., KJFK, KLAX, ORL)",
                },
            },
            "required": ["identifier"],
        },
    },
    {
        "name": "search_manual",
        "description": (
            "Search the aircraft operating manual and aviation knowledge base. Use this to "
            "look up procedures, limitations, V-speeds, systems descriptions, or any "
            "aircraft-specific information."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query describing what to look up",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_checklist",
        "description": (
            "Get the appropriate checklist for a given flight phase. Returns phase-specific "
            "checklist items, preferring aircraft-specific checklists when available."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "phase": {
                    "type": "string",
                    "description": (
                        "Flight phase (PREFLIGHT, TAXI, TAKEOFF, CLIMB, CRUISE, "
                        "DESCENT, APPROACH, LANDING, LANDED)"
                    ),
                },
            },
            "required": ["phase"],
        },
    },
    {
        "name": "create_flight_plan",
        "description": (
            "Create a basic flight plan between two airports. Returns a draft route "
            "structure with departure, destination, and waypoints."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "departure": {
                    "type": "string",
                    "description": "Departure airport identifier",
                },
                "destination": {
                    "type": "string",
                    "description": "Destination airport identifier",
                },
                "altitude": {
                    "type": "integer",
                    "description": "Planned cruise altitude in feet MSL",
                    "default": 5000,
                },
                "route": {
                    "type": "string",
                    "description": "Optional route waypoints separated by spaces",
                    "default": "",
                },
            },
            "required": ["departure", "destination"],
        },
    },
]

# Flight-phase-specific response style directives (mirrors claude_client.py)
PHASE_STYLES: dict[str, str] = {
    "PREFLIGHT": (
        "Phase: PREFLIGHT. Relaxed tone, moderate length. Good time for banter and briefings."
    ),
    "TAXI": "Phase: TAXI. Professional, concise. 1-2 sentences unless reading a checklist.",
    "TAKEOFF": "Phase: TAKEOFF. ULTRA-BRIEF. Callouts only. No humor. No filler.",
    "CLIMB": "Phase: CLIMB. Professional, moderate length. Light humor once established.",
    "CRUISE": "Phase: CRUISE. Conversational, can be more detailed. Good time to teach.",
    "DESCENT": "Phase: DESCENT. Briefing mode. Structured and clear. Minimal humor.",
    "APPROACH": "Phase: APPROACH. ULTRA-BRIEF. Concise callouts only. No humor. No filler.",
    "LANDING": "Phase: LANDING. ULTRA-BRIEF. Callouts only. Crisp and precise.",
    "LANDED": "Phase: LANDED. Relaxed debrief mode. Can use humor. Moderate length.",
}

# Minimal MERLIN persona for testing (avoids file dependency)
MERLIN_SYSTEM_PROMPT = """\
You are MERLIN, an AI co-pilot assistant for Microsoft Flight Simulator 2024.
You are a former Navy Test Pilot School graduate. You call the pilot "Captain."
You are professional, concise, and use aviation terminology naturally.
Use the available tools when you need real-time data or reference material.
Do NOT make up flight data — always use get_sim_state for current telemetry.
Do NOT guess airport info — always use lookup_airport.
For procedures and checklists, use search_manual or get_checklist.
For flight planning, use create_flight_plan.
Only use tools when the pilot's request requires them. For general knowledge,
banter, or acknowledgments, respond directly without tools.
"""


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A single tool call extracted from an LLM response."""

    name: str
    arguments: dict[str, Any]


@dataclass
class ScenarioResult:
    """Result of running a single scenario against an LLM backend."""

    scenario_id: str
    expected_tools: list[str]
    actual_tools: list[str]
    tool_calls: list[ToolCall]
    tool_selection_correct: bool
    arguments_valid: bool
    no_hallucinated_tools: bool
    errors: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    ttft_ms: float = 0.0
    total_tokens: int = 0
    response_text: str = ""


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------


def load_scenarios() -> list[dict[str, Any]]:
    """Load test scenarios from the fixtures JSON file."""
    with open(SCENARIOS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data["scenarios"]


def load_mock_responses() -> dict[str, Any]:
    """Load mock tool responses from the fixtures JSON file."""
    with open(MOCK_RESPONSES_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data["responses"]


def scenario_ids() -> list[str]:
    """Return scenario IDs for parametrize."""
    return [s["id"] for s in load_scenarios()]


def _scenarios_by_id() -> dict[str, dict[str, Any]]:
    """Return scenarios indexed by ID."""
    return {s["id"]: s for s in load_scenarios()}


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------


def build_system_prompt(flight_phase: str, sim_state: dict[str, Any]) -> str:
    """Build a MERLIN system prompt with flight phase context for testing."""
    parts = [MERLIN_SYSTEM_PROMPT]

    if flight_phase in PHASE_STYLES:
        parts.append(f"\n--- CURRENT RESPONSE STYLE ---\n{PHASE_STYLES[flight_phase]}")

    # Inject minimal flight context
    pos = sim_state.get("position", {})
    spd = sim_state.get("speeds", {})
    att = sim_state.get("attitude", {})
    parts.append(
        f"\n--- CURRENT FLIGHT STATE ---\n"
        f"Phase: {flight_phase} | "
        f"Alt: {pos.get('altitude_msl', 0):.0f}ft | "
        f"IAS: {spd.get('indicated_airspeed', 0):.0f}kt | "
        f"HDG: {att.get('heading_magnetic', 0):.0f}deg | "
        f"VS: {spd.get('vertical_speed', 0):+.0f}fpm"
    )
    parts.append(f"Aircraft: {sim_state.get('aircraft', 'Unknown')}")

    ap = sim_state.get("autopilot", {})
    if ap.get("master"):
        parts.append(
            f"Autopilot: HDG {ap.get('heading', 0):.0f} | "
            f"ALT {ap.get('altitude', 0):.0f}"
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def validate_tool_arguments(tool_name: str, arguments: dict[str, Any]) -> list[str]:
    """Validate tool call arguments against the schema. Returns a list of errors."""
    errors: list[str] = []

    if tool_name not in TOOL_ARG_SCHEMAS:
        errors.append(f"Unknown tool: {tool_name}")
        return errors

    schema = TOOL_ARG_SCHEMAS[tool_name]

    # Check required arguments
    for req in schema["required"]:
        if req not in arguments:
            errors.append(f"Missing required argument '{req}' for {tool_name}")

    # Check argument types
    for arg_name, arg_value in arguments.items():
        if arg_name not in schema["properties"]:
            # Extra arguments are not necessarily errors (LLMs may add optional ones)
            continue
        expected_type = schema["properties"][arg_name].get("type")
        if expected_type == "string" and not isinstance(arg_value, str):
            errors.append(
                f"Argument '{arg_name}' for {tool_name} should be string, "
                f"got {type(arg_value).__name__}"
            )
        elif expected_type == "integer" and not isinstance(arg_value, int):
            # Allow float if it is a whole number
            if isinstance(arg_value, float) and arg_value == int(arg_value):
                pass
            else:
                errors.append(
                    f"Argument '{arg_name}' for {tool_name} should be integer, "
                    f"got {type(arg_value).__name__}"
                )

    return errors


# ---------------------------------------------------------------------------
# Backend adapters
# ---------------------------------------------------------------------------


class BackendAdapter:
    """Base class for LLM backend adapters."""

    async def send_message(
        self,
        system_prompt: str,
        user_message: str,
    ) -> tuple[list[ToolCall], str, dict[str, float]]:
        """Send a message and return (tool_calls, response_text, timing_info).

        timing_info keys: ttft_ms, total_ms, total_tokens
        """
        raise NotImplementedError


class AnthropicAdapter(BackendAdapter):
    """Adapter for the Anthropic Claude API."""

    def __init__(self, api_key: str, model: str) -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    async def send_message(
        self,
        system_prompt: str,
        user_message: str,
    ) -> tuple[list[ToolCall], str, dict[str, float]]:
        start = time.perf_counter()
        ttft = 0.0

        tool_calls: list[ToolCall] = []
        text_parts: list[str] = []
        total_tokens = 0

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            tools=ANTHROPIC_TOOLS,
        )

        ttft = (time.perf_counter() - start) * 1000
        total_tokens = response.usage.input_tokens + response.usage.output_tokens

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        total_ms = (time.perf_counter() - start) * 1000

        return (
            tool_calls,
            "".join(text_parts),
            {"ttft_ms": ttft, "total_ms": total_ms, "total_tokens": total_tokens},
        )


class OpenAICompatibleAdapter(BackendAdapter):
    """Adapter for OpenAI-compatible APIs (vLLM, llama.cpp, Ollama, etc.)."""

    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required for OpenAI-compatible backend. "
                "Install with: pip install openai"
            )
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def send_message(
        self,
        system_prompt: str,
        user_message: str,
    ) -> tuple[list[ToolCall], str, dict[str, float]]:
        start = time.perf_counter()

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            tools=OPENAI_TOOLS,
            tool_choice="auto",
            max_tokens=1024,
            temperature=0.0,
        )

        total_ms = (time.perf_counter() - start) * 1000
        ttft = total_ms  # Non-streaming: TTFT equals total

        tool_calls: list[ToolCall] = []
        text_parts: list[str] = []
        total_tokens = 0

        if response.usage:
            total_tokens = (
                (response.usage.prompt_tokens or 0)
                + (response.usage.completion_tokens or 0)
            )

        choice = response.choices[0] if response.choices else None
        if choice and choice.message:
            if choice.message.content:
                text_parts.append(choice.message.content)
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    tool_calls.append(ToolCall(
                        name=tc.function.name,
                        arguments=args,
                    ))

        return (
            tool_calls,
            "".join(text_parts),
            {"ttft_ms": ttft, "total_ms": total_ms, "total_tokens": total_tokens},
        )


def get_backend_adapter() -> BackendAdapter:
    """Create the appropriate backend adapter from environment variables."""
    backend = os.environ.get("TOOL_PARITY_BACKEND", "anthropic").lower()
    api_key = os.environ.get("TOOL_PARITY_API_KEY", "")

    if backend == "anthropic":
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        model = os.environ.get("TOOL_PARITY_MODEL", "claude-sonnet-4-20250514")
        return AnthropicAdapter(api_key=api_key, model=model)
    elif backend == "openai":
        base_url = os.environ.get("TOOL_PARITY_BASE_URL", "http://localhost:8081/v1")
        model = os.environ.get("TOOL_PARITY_MODEL", "default")
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
        return OpenAICompatibleAdapter(api_key=api_key, model=model, base_url=base_url)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'anthropic' or 'openai'.")


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------


async def evaluate_scenario(
    adapter: BackendAdapter,
    scenario: dict[str, Any],
) -> ScenarioResult:
    """Run a single scenario against the backend and evaluate the result."""
    scenario_id = scenario["id"]
    expected_tools = scenario["expected_tools"]
    flight_phase = scenario["flight_phase"]
    user_message = scenario["user_message"]
    sim_state = scenario["sim_state"]

    system_prompt = build_system_prompt(flight_phase, sim_state)

    errors: list[str] = []
    tool_calls: list[ToolCall] = []
    response_text = ""
    timing: dict[str, float] = {"ttft_ms": 0, "total_ms": 0, "total_tokens": 0}

    try:
        tool_calls, response_text, timing = await adapter.send_message(
            system_prompt, user_message
        )
    except Exception as e:
        errors.append(f"Backend error: {e}")

    actual_tool_names = [tc.name for tc in tool_calls]

    # Evaluate: tool selection
    # For multi-tool scenarios, check that all expected tools are present (order-independent)
    expected_set = set(expected_tools)
    actual_set = set(actual_tool_names)
    tool_selection_correct = expected_set == actual_set

    # Evaluate: no hallucinated tools
    no_hallucinated = actual_set.issubset(VALID_TOOL_NAMES)
    if not no_hallucinated:
        hallucinated = actual_set - VALID_TOOL_NAMES
        errors.append(f"Hallucinated tools: {hallucinated}")

    # Evaluate: argument validity
    arguments_valid = True
    for tc in tool_calls:
        arg_errors = validate_tool_arguments(tc.name, tc.arguments)
        if arg_errors:
            arguments_valid = False
            errors.extend(arg_errors)

    # Check expected_args if provided
    expected_args = scenario.get("expected_args", {})
    for tool_name, expected in expected_args.items():
        matching_calls = [tc for tc in tool_calls if tc.name == tool_name]
        if matching_calls:
            actual_args = matching_calls[0].arguments
            for key, expected_value in expected.items():
                actual_value = actual_args.get(key)
                if actual_value is not None:
                    # Normalize strings for comparison (case-insensitive for identifiers)
                    if isinstance(expected_value, str) and isinstance(actual_value, str):
                        if expected_value.upper() != actual_value.upper():
                            errors.append(
                                f"Argument mismatch for {tool_name}.{key}: "
                                f"expected '{expected_value}', got '{actual_value}'"
                            )

    return ScenarioResult(
        scenario_id=scenario_id,
        expected_tools=expected_tools,
        actual_tools=actual_tool_names,
        tool_calls=tool_calls,
        tool_selection_correct=tool_selection_correct,
        arguments_valid=arguments_valid,
        no_hallucinated_tools=no_hallucinated,
        errors=errors,
        latency_ms=timing.get("total_ms", 0),
        ttft_ms=timing.get("ttft_ms", 0),
        total_tokens=int(timing.get("total_tokens", 0)),
        response_text=response_text,
    )


# ---------------------------------------------------------------------------
# Unit tests — fixture structure validation (always runnable, no LLM needed)
# ---------------------------------------------------------------------------


class TestFixtureIntegrity:
    """Validate that the test fixtures are well-formed and consistent."""

    def test_scenarios_file_loads(self) -> None:
        scenarios = load_scenarios()
        assert len(scenarios) >= 30, f"Expected 30+ scenarios, got {len(scenarios)}"

    def test_mock_responses_file_loads(self) -> None:
        responses = load_mock_responses()
        assert "get_sim_state" in responses
        assert "lookup_airport" in responses
        assert "search_manual" in responses
        assert "get_checklist" in responses
        assert "create_flight_plan" in responses

    def test_all_scenarios_have_required_fields(self) -> None:
        required_fields = {"id", "description", "flight_phase", "user_message",
                           "expected_tools", "sim_state", "tags"}
        for scenario in load_scenarios():
            missing = required_fields - set(scenario.keys())
            assert not missing, (
                f"Scenario '{scenario.get('id', '?')}' missing fields: {missing}"
            )

    def test_scenario_ids_are_unique(self) -> None:
        scenarios = load_scenarios()
        ids = [s["id"] for s in scenarios]
        assert len(ids) == len(set(ids)), "Duplicate scenario IDs found"

    def test_expected_tools_are_valid(self) -> None:
        for scenario in load_scenarios():
            for tool in scenario["expected_tools"]:
                assert tool in VALID_TOOL_NAMES, (
                    f"Scenario '{scenario['id']}' references unknown tool: {tool}"
                )

    def test_flight_phases_are_valid(self) -> None:
        valid_phases = {
            "PREFLIGHT", "TAXI", "TAKEOFF", "CLIMB", "CRUISE",
            "DESCENT", "APPROACH", "LANDING", "LANDED",
        }
        for scenario in load_scenarios():
            assert scenario["flight_phase"] in valid_phases, (
                f"Scenario '{scenario['id']}' has invalid phase: {scenario['flight_phase']}"
            )

    def test_sim_state_has_core_fields(self) -> None:
        core_fields = {"connected", "aircraft", "flight_phase", "position", "speeds"}
        for scenario in load_scenarios():
            state = scenario["sim_state"]
            missing = core_fields - set(state.keys())
            assert not missing, (
                f"Scenario '{scenario['id']}' sim_state missing: {missing}"
            )

    def test_scenario_tags_present(self) -> None:
        for scenario in load_scenarios():
            assert len(scenario["tags"]) > 0, (
                f"Scenario '{scenario['id']}' has no tags"
            )

    def test_category_coverage(self) -> None:
        """Verify we have sufficient coverage across categories."""
        scenarios = load_scenarios()
        single_tool = [s for s in scenarios if "single-tool" in s["tags"]]
        multi_tool = [s for s in scenarios if "multi-tool" in s["tags"]]
        no_tool = [s for s in scenarios if "no-tool" in s["tags"]]
        phase_specific = [s for s in scenarios if "phase-specific" in s["tags"]]

        assert len(single_tool) >= 8, f"Need 8+ single-tool, got {len(single_tool)}"
        assert len(multi_tool) >= 6, f"Need 6+ multi-tool, got {len(multi_tool)}"
        assert len(no_tool) >= 6, f"Need 6+ no-tool, got {len(no_tool)}"
        assert len(phase_specific) >= 5, f"Need 5+ phase-specific, got {len(phase_specific)}"

    def test_all_tools_covered_in_single_tool_scenarios(self) -> None:
        """Ensure every tool has at least one single-tool scenario."""
        scenarios = load_scenarios()
        single_tool_scenarios = [s for s in scenarios if "single-tool" in s["tags"]]
        covered_tools: set[str] = set()
        for s in single_tool_scenarios:
            covered_tools.update(s["expected_tools"])
        assert VALID_TOOL_NAMES == covered_tools, (
            f"Missing tool coverage: {VALID_TOOL_NAMES - covered_tools}"
        )


class TestArgumentValidation:
    """Validate the argument validation logic itself."""

    def test_valid_get_sim_state_args(self) -> None:
        errors = validate_tool_arguments("get_sim_state", {})
        assert errors == []

    def test_valid_lookup_airport_args(self) -> None:
        errors = validate_tool_arguments("lookup_airport", {"identifier": "KJFK"})
        assert errors == []

    def test_missing_required_lookup_airport(self) -> None:
        errors = validate_tool_arguments("lookup_airport", {})
        assert any("Missing required" in e for e in errors)

    def test_valid_search_manual_args(self) -> None:
        errors = validate_tool_arguments("search_manual", {"query": "V-speeds"})
        assert errors == []

    def test_valid_get_checklist_args(self) -> None:
        errors = validate_tool_arguments("get_checklist", {"phase": "PREFLIGHT"})
        assert errors == []

    def test_valid_create_flight_plan_args(self) -> None:
        errors = validate_tool_arguments(
            "create_flight_plan",
            {"departure": "KMCO", "destination": "KJAX"},
        )
        assert errors == []

    def test_create_flight_plan_with_optional_args(self) -> None:
        errors = validate_tool_arguments(
            "create_flight_plan",
            {"departure": "KMCO", "destination": "KJAX", "altitude": 5500, "route": "DIRECT"},
        )
        assert errors == []

    def test_wrong_type_identifier(self) -> None:
        errors = validate_tool_arguments("lookup_airport", {"identifier": 123})
        assert any("should be string" in e for e in errors)

    def test_unknown_tool(self) -> None:
        errors = validate_tool_arguments("nonexistent_tool", {})
        assert any("Unknown tool" in e for e in errors)


class TestSystemPromptBuilder:
    """Validate the system prompt builder used for parity testing."""

    def test_includes_persona(self) -> None:
        prompt = build_system_prompt("CRUISE", {"aircraft": "C172"})
        assert "MERLIN" in prompt

    def test_includes_phase_style(self) -> None:
        prompt = build_system_prompt("TAKEOFF", {"aircraft": "C172"})
        assert "ULTRA-BRIEF" in prompt

    def test_includes_flight_state(self) -> None:
        state = {
            "aircraft": "Cessna 172 Skyhawk",
            "position": {"altitude_msl": 6500},
            "speeds": {"indicated_airspeed": 120, "vertical_speed": 0},
            "attitude": {"heading_magnetic": 42},
        }
        prompt = build_system_prompt("CRUISE", state)
        assert "6500" in prompt
        assert "120" in prompt

    def test_includes_autopilot_when_active(self) -> None:
        state = {"autopilot": {"master": True, "heading": 270, "altitude": 5500}}
        prompt = build_system_prompt("CRUISE", state)
        assert "Autopilot" in prompt
        assert "270" in prompt


# ---------------------------------------------------------------------------
# Integration tests — require a running LLM backend
# ---------------------------------------------------------------------------

_SCENARIOS = load_scenarios()
_SCENARIO_IDS = [s["id"] for s in _SCENARIOS]
_SCENARIOS_BY_ID = {s["id"]: s for s in _SCENARIOS}


@pytest.mark.integration
class TestToolCallingParity:
    """Integration tests that validate tool calling against a live LLM backend.

    These tests require either an Anthropic API key or a running local LLM server.
    Configure via environment variables (see module docstring).
    """

    @pytest.fixture(scope="class")
    def adapter(self) -> BackendAdapter:
        return get_backend_adapter()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("scenario_id", _SCENARIO_IDS)
    async def test_tool_selection(
        self,
        adapter: BackendAdapter,
        scenario_id: str,
    ) -> None:
        """Validate that the LLM selects the correct tools for each scenario."""
        scenario = _SCENARIOS_BY_ID[scenario_id]
        result = await evaluate_scenario(adapter, scenario)

        # Log details for debugging
        logger.info(
            "Scenario %s: expected=%s actual=%s match=%s errors=%s",
            scenario_id,
            result.expected_tools,
            result.actual_tools,
            result.tool_selection_correct,
            result.errors,
        )

        assert result.no_hallucinated_tools, (
            f"[{scenario_id}] Hallucinated tools detected. "
            f"Actual: {result.actual_tools}, Valid: {list(VALID_TOOL_NAMES)}"
        )
        assert result.tool_selection_correct, (
            f"[{scenario_id}] Tool selection mismatch. "
            f"Expected: {result.expected_tools}, Got: {result.actual_tools}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("scenario_id", _SCENARIO_IDS)
    async def test_argument_validity(
        self,
        adapter: BackendAdapter,
        scenario_id: str,
    ) -> None:
        """Validate that tool call arguments are well-formed and schema-compliant."""
        scenario = _SCENARIOS_BY_ID[scenario_id]
        result = await evaluate_scenario(adapter, scenario)

        assert result.arguments_valid, (
            f"[{scenario_id}] Invalid arguments: {result.errors}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "scenario_id",
        [s["id"] for s in _SCENARIOS if "single-tool" in s["tags"]],
    )
    async def test_single_tool_scenarios(
        self,
        adapter: BackendAdapter,
        scenario_id: str,
    ) -> None:
        """Single-tool scenarios should call exactly one tool."""
        scenario = _SCENARIOS_BY_ID[scenario_id]
        result = await evaluate_scenario(adapter, scenario)

        assert len(result.actual_tools) == 1, (
            f"[{scenario_id}] Expected 1 tool call, got {len(result.actual_tools)}: "
            f"{result.actual_tools}"
        )
        assert result.tool_selection_correct, (
            f"[{scenario_id}] Wrong tool. Expected: {result.expected_tools}, "
            f"Got: {result.actual_tools}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "scenario_id",
        [s["id"] for s in _SCENARIOS if "no-tool" in s["tags"]],
    )
    async def test_no_tool_scenarios(
        self,
        adapter: BackendAdapter,
        scenario_id: str,
    ) -> None:
        """No-tool scenarios should not call any tools."""
        scenario = _SCENARIOS_BY_ID[scenario_id]
        result = await evaluate_scenario(adapter, scenario)

        assert len(result.actual_tools) == 0, (
            f"[{scenario_id}] Expected no tool calls, got: {result.actual_tools}"
        )
        assert result.response_text.strip(), (
            f"[{scenario_id}] Expected a text response for no-tool scenario"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "scenario_id",
        [s["id"] for s in _SCENARIOS if "multi-tool" in s["tags"]],
    )
    async def test_multi_tool_scenarios(
        self,
        adapter: BackendAdapter,
        scenario_id: str,
    ) -> None:
        """Multi-tool scenarios should call at least 2 tools from the expected set."""
        scenario = _SCENARIOS_BY_ID[scenario_id]
        result = await evaluate_scenario(adapter, scenario)

        expected_set = set(scenario["expected_tools"])
        actual_set = set(result.actual_tools)

        # At minimum, the LLM should call at least 2 of the expected tools
        overlap = expected_set & actual_set
        assert len(overlap) >= 2, (
            f"[{scenario_id}] Expected at least 2 tools from {list(expected_set)}, "
            f"got overlap: {list(overlap)} (actual: {result.actual_tools})"
        )
