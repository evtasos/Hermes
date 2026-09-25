"""
Hermes CLI I/O helpers.
Keeps printing/formatting separate from logic.
"""
import sys
from typing import Optional


def print_agent(message: str) -> None:
    """Print agent response."""
    print(f"Agent: {message}")


def print_error(message: str) -> None:
    """Print error message."""
    print(f"Error: {message}", file=sys.stderr)


def print_tool_call(name: str, args: dict) -> None:
    """Print tool invocation."""
    print(f"  [calling tool: {name}({args})]")


def print_tool_result(name: str, result: str) -> None:
    """Print tool result (truncated)."""
    truncated = result[:200] + ("…" if len(result) > 200 else "")
    print(f"  [tool result: {name} → {truncated}]")