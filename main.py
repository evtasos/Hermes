#!/usr/bin/env python3
"""
Hermes main entrypoint.
Handles startup tasks (git pull) then enters the REPL.
"""
import asyncio
import sys
from pathlib import Path

from controller.agent_loop import run_repl
from utils.git_helper import git_pull
from config import GIT_AUTO_PULL_ON_START


async def main() -> None:
    """Startup and REPL."""
    if GIT_AUTO_PULL_ON_START:
        print("🔄 Checking for updates...")
        result = git_pull()
        print(result)

    print("🤖 Hermes is ready. Type your questions or 'exit' to quit.")
    await run_repl()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)