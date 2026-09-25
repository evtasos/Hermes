from contextlib import asynccontextmanager

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from config import HA_URL, HA_TOKEN
from tools.adapters import mcp_tools_to_openai


@asynccontextmanager
async def mcp_session():
    """Connect to the Home Assistant MCP server and yield
    (session, mcp_tools, openai_tools) for the lifetime of the `async with` block.
    """
    async with streamablehttp_client(HA_URL, headers={"Authorization": f"Bearer {HA_TOKEN}"}) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            openai_tools = mcp_tools_to_openai(mcp_tools)
            yield session, mcp_tools, openai_tools
