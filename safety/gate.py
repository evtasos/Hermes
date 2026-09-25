"""
Safety gate for write tools.
Asks user for confirmation before executing actions that change state.
"""
import asyncio
from typing import Dict, Any, Optional

import structlog

from cli import print_agent
from config import REQUIRE_WRITE_CONFIRMATION

logger = structlog.get_logger()


async def write_gate(
    tool_name: str,
    tool_args: Dict[str, Any],
    user_input: str,
    mcp_client: Optional[object] = None,
) -> bool:
    """
    Prompt user for confirmation.
    Returns True if confirmed, False otherwise.
    """
    if not REQUIRE_WRITE_CONFIRMATION:
        return True

    # Build a human-readable description
    desc = f"{tool_name}"
    if tool_args:
        args_str = ", ".join(f"{k}={v}" for k, v in tool_args.items())
        desc += f"({args_str})"

    print(f"\n⚠️  About to execute: {desc}")
    print(f"   Context: {user_input[:80]}{'...' if len(user_input) > 80 else ''}")
    reply = input("   Confirm? [y/N]: ").strip().lower()

    confirmed = reply in {"y", "yes"}
    if not confirmed:
        print("   → Action cancelled by user.")
    else:
        print("   → Action confirmed.")

    return confirmed