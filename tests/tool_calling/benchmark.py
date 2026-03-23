#!/usr/bin/env python3
"""
RIO Phase 1.3: Tool Calling Parity Benchmark

Runs all tool calling scenarios against a specified LLM backend and generates
a markdown report with accuracy and performance metrics.

Usage:
    # Against Anthropic API (baseline)
    python benchmark.py --backend anthropic --api-key sk-ant-...

    # Against local vLLM / llama.cpp server
    python benchmark.py --backend local --url http://localhost:8081/v1 --model my-model

    # Against Ollama
    python benchmark.py --backend local --url http://localhost:11434/v1 --model llama3

    # Save report to file
    python benchmark.py --backend local --url http://localhost:8081/v1 --output report.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add parent directories to path so we can import from the test module
sys.path.insert(0, str(Path(__file__).parent))

from test_tool_parity import (
    VALID_TOOL_NAMES,
    AnthropicAdapter,
    BackendAdapter,
    OpenAICompatibleAdapter,
    ScenarioResult,
    evaluate_scenario,
    load_scenarios,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkMetrics:
    """Aggregated benchmark metrics across all scenarios."""

    total_scenarios: int = 0
    tool_selection_correct: int = 0
    arguments_valid: int = 0
    no_hallucinated: int = 0
    total_latency_ms: float = 0.0
    total_ttft_ms: float = 0.0
    total_tokens: int = 0
    errors: int = 0

    # Per-category metrics
    single_tool_correct: int = 0
    single_tool_total: int = 0
    multi_tool_correct: int = 0
    multi_tool_total: int = 0
    no_tool_correct: int = 0
    no_tool_total: int = 0
    phase_specific_correct: int = 0
    phase_specific_total: int = 0

    @property
    def tool_selection_rate(self) -> float:
        return self.tool_selection_correct / max(self.total_scenarios, 1) * 100

    @property
    def argument_validity_rate(self) -> float:
        return self.arguments_valid / max(self.total_scenarios, 1) * 100

    @property
    def hallucination_free_rate(self) -> float:
        return self.no_hallucinated / max(self.total_scenarios, 1) * 100

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.total_scenarios, 1)

    @property
    def avg_ttft_ms(self) -> float:
        return self.total_ttft_ms / max(self.total_scenarios, 1)

    @property
    def avg_tokens_per_second(self) -> float:
        if self.total_latency_ms <= 0:
            return 0.0
        return self.total_tokens / (self.total_latency_ms / 1000)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


async def run_benchmark(
    adapter: BackendAdapter,
    scenarios: list[dict[str, Any]],
    verbose: bool = False,
) -> tuple[list[ScenarioResult], BenchmarkMetrics]:
    """Run all scenarios and collect metrics."""
    results: list[ScenarioResult] = []
    metrics = BenchmarkMetrics(total_scenarios=len(scenarios))

    for i, scenario in enumerate(scenarios, 1):
        sid = scenario["id"]
        tags = scenario.get("tags", [])

        if verbose:
            print(f"  [{i}/{len(scenarios)}] {sid}...", end=" ", flush=True)

        try:
            result = await evaluate_scenario(adapter, scenario)
            results.append(result)

            # Aggregate metrics
            if result.tool_selection_correct:
                metrics.tool_selection_correct += 1
            if result.arguments_valid:
                metrics.arguments_valid += 1
            if result.no_hallucinated_tools:
                metrics.no_hallucinated += 1
            if result.errors:
                metrics.errors += 1

            metrics.total_latency_ms += result.latency_ms
            metrics.total_ttft_ms += result.ttft_ms
            metrics.total_tokens += result.total_tokens

            # Category metrics
            if "single-tool" in tags:
                metrics.single_tool_total += 1
                if result.tool_selection_correct:
                    metrics.single_tool_correct += 1
            if "multi-tool" in tags:
                metrics.multi_tool_total += 1
                if result.tool_selection_correct:
                    metrics.multi_tool_correct += 1
            if "no-tool" in tags:
                metrics.no_tool_total += 1
                if result.tool_selection_correct:
                    metrics.no_tool_correct += 1
            if "phase-specific" in tags:
                metrics.phase_specific_total += 1
                if result.tool_selection_correct:
                    metrics.phase_specific_correct += 1

            if verbose:
                status = "PASS" if result.tool_selection_correct else "FAIL"
                print(
                    f"{status} ({result.latency_ms:.0f}ms, "
                    f"tools={result.actual_tools})"
                )

        except Exception as e:
            metrics.errors += 1
            if verbose:
                print(f"ERROR: {e}")
            results.append(ScenarioResult(
                scenario_id=sid,
                expected_tools=scenario["expected_tools"],
                actual_tools=[],
                tool_calls=[],
                tool_selection_correct=False,
                arguments_valid=False,
                no_hallucinated_tools=True,
                errors=[str(e)],
            ))

    return results, metrics


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    results: list[ScenarioResult],
    metrics: BenchmarkMetrics,
    backend_name: str,
    model_name: str,
) -> str:
    """Generate a markdown benchmark report."""
    lines: list[str] = []

    lines.append("# RIO Phase 1.3 -- Tool Calling Parity Benchmark Report")
    lines.append("")
    lines.append(f"**Backend:** {backend_name}")
    lines.append(f"**Model:** {model_name}")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    lines.append(f"**Scenarios:** {metrics.total_scenarios}")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(
        f"| Tool Selection Accuracy | "
        f"{metrics.tool_selection_correct}/{metrics.total_scenarios} "
        f"({metrics.tool_selection_rate:.1f}%) |"
    )
    lines.append(
        f"| Argument Validity Rate | "
        f"{metrics.arguments_valid}/{metrics.total_scenarios} "
        f"({metrics.argument_validity_rate:.1f}%) |"
    )
    lines.append(
        f"| Hallucination-Free Rate | "
        f"{metrics.no_hallucinated}/{metrics.total_scenarios} "
        f"({metrics.hallucination_free_rate:.1f}%) |"
    )
    lines.append(f"| Avg Latency (total) | {metrics.avg_latency_ms:.0f} ms |")
    lines.append(f"| Avg TTFT | {metrics.avg_ttft_ms:.0f} ms |")
    lines.append(
        f"| Avg Tokens/Second | {metrics.avg_tokens_per_second:.1f} tok/s |"
    )
    lines.append(f"| Total Tokens | {metrics.total_tokens} |")
    lines.append(f"| Errors | {metrics.errors} |")
    lines.append("")

    # Category breakdown
    lines.append("## Category Breakdown")
    lines.append("")
    lines.append("| Category | Correct | Total | Rate |")
    lines.append("|---|---|---|---|")

    def _cat_row(label: str, correct: int, total: int) -> str:
        rate = correct / max(total, 1) * 100
        return f"| {label} | {correct} | {total} | {rate:.1f}% |"

    lines.append(_cat_row(
        "Single-tool", metrics.single_tool_correct, metrics.single_tool_total
    ))
    lines.append(_cat_row(
        "Multi-tool", metrics.multi_tool_correct, metrics.multi_tool_total
    ))
    lines.append(_cat_row(
        "No-tool", metrics.no_tool_correct, metrics.no_tool_total
    ))
    lines.append(_cat_row(
        "Phase-specific", metrics.phase_specific_correct, metrics.phase_specific_total
    ))
    lines.append("")

    # Detailed results table
    lines.append("## Detailed Results")
    lines.append("")
    lines.append(
        "| Scenario | Expected | Actual | Selection | Args | "
        "Latency (ms) | Tokens |"
    )
    lines.append("|---|---|---|---|---|---|---|")

    for r in results:
        sel_icon = "PASS" if r.tool_selection_correct else "FAIL"
        arg_icon = "PASS" if r.arguments_valid else "FAIL"
        expected_str = ", ".join(r.expected_tools) if r.expected_tools else "(none)"
        actual_str = ", ".join(r.actual_tools) if r.actual_tools else "(none)"
        lines.append(
            f"| {r.scenario_id} | {expected_str} | {actual_str} | "
            f"{sel_icon} | {arg_icon} | {r.latency_ms:.0f} | {r.total_tokens} |"
        )

    lines.append("")

    # Failures section
    failures = [r for r in results if not r.tool_selection_correct or r.errors]
    if failures:
        lines.append("## Failures and Errors")
        lines.append("")
        for r in failures:
            lines.append(f"### {r.scenario_id}")
            lines.append(f"- **Expected tools:** {r.expected_tools}")
            lines.append(f"- **Actual tools:** {r.actual_tools}")
            if r.errors:
                lines.append(f"- **Errors:** {'; '.join(r.errors)}")
            if r.response_text:
                preview = r.response_text[:200].replace("\n", " ")
                lines.append(f"- **Response preview:** {preview}")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RIO Phase 1.3: Tool Calling Parity Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--backend",
        choices=["anthropic", "local"],
        default="local",
        help="LLM backend to test (default: local)",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8081/v1",
        help="Base URL for local/OpenAI-compatible backend (default: http://localhost:8081/v1)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model name (default: auto-detect based on backend)",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="API key (reads from ANTHROPIC_API_KEY or OPENAI_API_KEY env var if omitted)",
    )
    parser.add_argument(
        "--output", "-o",
        default="",
        help="Output file for the markdown report (default: stdout)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress for each scenario",
    )
    parser.add_argument(
        "--tags",
        default="",
        help="Comma-separated tags to filter scenarios (e.g., 'single-tool,routine')",
    )
    parser.add_argument(
        "--scenarios",
        default="",
        help="Comma-separated scenario IDs to run (overrides --tags)",
    )
    return parser.parse_args()


def create_adapter(args: argparse.Namespace) -> tuple[BackendAdapter, str, str]:
    """Create the backend adapter and return (adapter, backend_name, model_name)."""
    import os

    if args.backend == "anthropic":
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("Error: Anthropic API key required. Use --api-key or ANTHROPIC_API_KEY env var.")
            sys.exit(1)
        model = args.model or "claude-sonnet-4-20250514"
        return AnthropicAdapter(api_key=api_key, model=model), "Anthropic", model
    else:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
        model = args.model or "default"
        return (
            OpenAICompatibleAdapter(api_key=api_key, model=model, base_url=args.url),
            f"OpenAI-compatible ({args.url})",
            model,
        )


async def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    # Load and filter scenarios
    all_scenarios = load_scenarios()

    if args.scenarios:
        selected_ids = {s.strip() for s in args.scenarios.split(",")}
        scenarios = [s for s in all_scenarios if s["id"] in selected_ids]
        if not scenarios:
            print(f"Error: no scenarios matched IDs: {selected_ids}")
            sys.exit(1)
    elif args.tags:
        required_tags = {t.strip() for t in args.tags.split(",")}
        scenarios = [
            s for s in all_scenarios
            if required_tags.issubset(set(s.get("tags", [])))
        ]
        if not scenarios:
            print(f"Error: no scenarios matched tags: {required_tags}")
            sys.exit(1)
    else:
        scenarios = all_scenarios

    print(f"RIO Phase 1.3 Tool Calling Benchmark")
    print(f"Backend: {args.backend}")
    print(f"Scenarios: {len(scenarios)}")
    print()

    adapter, backend_name, model_name = create_adapter(args)

    results, metrics = await run_benchmark(adapter, scenarios, verbose=args.verbose)

    print()
    report = generate_report(results, metrics, backend_name, model_name)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report, encoding="utf-8")
        print(f"Report saved to: {output_path}")
    else:
        print(report)

    # Exit with non-zero if accuracy is below threshold
    if metrics.tool_selection_rate < 80:
        print(
            f"\nWARNING: Tool selection accuracy ({metrics.tool_selection_rate:.1f}%) "
            f"is below 80% threshold."
        )
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
