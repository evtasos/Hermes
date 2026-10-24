"""
Generic multi-server MCP manager.

Connects to any number of MCP servers behind one AsyncExitStack (so cleanup
is correct no matter how many are connected - the old single-server client
entered its transport context manually and never retained it to close
later), aggregates every server's tools into one list, and classifies each
tool as safe-to-run or needs-confirmation using the tool's own MCP
`annotations` - never by hardcoding a specific tool name. Adding a new
server is one `add_server()` call; nothing here needs to know what that
server's tools are called or do.
"""
from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import structlog
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = structlog.get_logger()


@dataclass
class ServerConfig:
    name: str
    url: str
    headers: Optional[Dict[str, str]] = None


@dataclass
class _ToolEntry:
    tool: Any  # raw MCP Tool object: .name, .description, .inputSchema, .annotations
    server_name: str


class MCPManager:
    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._sessions: Dict[str, ClientSession] = {}
        self._tools: Dict[str, _ToolEntry] = {}

    async def add_server(self, server: ServerConfig) -> None:
        """Connect one MCP server and merge its tools into the aggregate set."""
        # streamablehttp_client's return arity isn't consistent across SDK
        # builds - a 3-tuple (read, write, get_session_id) in the classic
        # v1 API, a 2-tuple in others. read/write are always first, so take
        # them positionally instead of unpacking an assumed exact count.
        transport = await self._stack.enter_async_context(
            streamablehttp_client(server.url, headers=server.headers or {})
        )
        read, write = transport[0], transport[1]
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._sessions[server.name] = session

        tools = (await session.list_tools()).tools
        added = 0
        for tool in tools:
            if tool.name in self._tools:
                logger.warning(
                    "mcp_tool_name_collision",
                    tool=tool.name,
                    existing_server=self._tools[tool.name].server_name,
                    new_server=server.name,
                )
                continue
            self._tools[tool.name] = _ToolEntry(tool=tool, server_name=server.name)
            added += 1

        logger.info("mcp_server_added", server=server.name, tool_count=added)

    def all_tools(self) -> List[Any]:
        """Raw MCP Tool objects across every connected server, for the LLM router."""
        return [entry.tool for entry in self._tools.values()]

    def needs_confirmation(self, tool_name: str) -> bool:
        """
        Spec-conservative by design: a tool needs confirmation unless it
        explicitly declares `readOnlyHint: true`. An unknown tool, a tool
        with no annotations at all, or one with readOnlyHint false/absent
        all default to requiring confirmation - this matches the MCP spec's
        own default (destructiveHint defaults to true when unset) and is
        what makes this safe to point at a server you haven't audited.

        Annotations are hints, not a security guarantee (the spec says so
        explicitly) - this decides whether to prompt, never whether to skip
        the prompt outright for something already flagged as risky.
        """
        entry = self._tools.get(tool_name)
        if entry is None:
            return True
        annotations = getattr(entry.tool, "annotations", None)
        if annotations is None:
            return True
        return getattr(annotations, "readOnlyHint", None) is not True

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        entry = self._tools.get(tool_name)
        if entry is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        session = self._sessions[entry.server_name]
        logger.info("mcp_tool_call", tool=tool_name, server=entry.server_name, args=arguments)
        result = await session.call_tool(tool_name, arguments=arguments)
        logger.info("mcp_tool_result", tool=tool_name, has_content=bool(getattr(result, "content", None)))
        return result

    async def cleanup(self) -> None:
        await self._stack.aclose()
        self._sessions.clear()
        self._tools.clear()
