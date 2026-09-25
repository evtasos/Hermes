"""
MCP client wrapper for ha-mcp.
Manages a single StreamableHTTP session and provides helper methods.
"""
from typing import List, Dict, Any, Optional, Set

import structlog
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from config import HA_MCP_URL, HA_TOKEN

logger = structlog.get_logger()


class MCPClient:
    def __init__(self):
        self._read = None
        self._write = None
        self._session: Optional[ClientSession] = None
        self._tools: Optional[List[Any]] = None
        # Meta-tool names when ha-mcp search mode is enabled
        self._meta_tool_names: Set[str] = {
            "ha_search_tools",
            "ha_read_call",
            "ha_write_call",
            "ha_delete_call",
        }

    async def initialize(self) -> None:
        """Initialize the MCP session and discover meta-tools."""
        if self._session is not None:
            return

        try:
            client_ctx = streamablehttp_client(
                HA_MCP_URL,
                headers={"Authorization": f"Bearer {HA_TOKEN}"},
            )
            self._read, self._write, _ = await client_ctx.__aenter__()
            self._session = ClientSession(self._read, self._write)
            await self._session.initialize()
            logger.info("mcp_session_initialized")
        except Exception as e:
            logger.error("mcp_init_failed", error=str(e))
            raise

    async def list_tools(self) -> List[Any]:
        """List all available tools (cached after first call)."""
        if self._tools is None:
            if self._session is None:
                await self.initialize()
            self._tools = (await self._session.list_tools()).tools
            logger.info("mcp_tools_listed", count=len(self._tools))
        return self._tools

    async def get_meta_tools(self) -> List[Any]:
        """Return only the meta-tools needed for search-first pattern."""
        all_tools = await self.list_tools()
        meta_tools = [t for t in all_tools if t.name in self._meta_tool_names]
        logger.info("mcp_meta_tools", count=len(meta_tools))
        return meta_tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool by name."""
        if self._session is None:
            await self.initialize()
        logger.info("mcp_tool_call", tool=name, args=arguments)
        result = await self._session.call_tool(name, arguments=arguments)
        logger.info("mcp_tool_result", tool=name, has_content=bool(result.content))
        return result

    async def is_write_tool(self, tool_name: str) -> bool:
        """Heuristic: treat ha_write_call as write; others as read."""
        return tool_name == "ha_write_call"

    async def cleanup(self) -> None:
        """Close the session."""
        if self._session:
            await self._session.__aexit__(None, None, None)
            self._session = None
            self._tools = None