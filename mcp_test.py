import asyncio
import os
import json
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

load_dotenv()

HA_URL = os.environ["HA_MCP_URL"]
HA_TOKEN = os.environ["HA_TOKEN"]

async def main():
    async with streamablehttp_client(
        HA_URL,
        headers={"Authorization": f"Bearer {HA_TOKEN}"}
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected to MCP server.\n")

            tools_response = await session.list_tools()

            # Find GetLiveContext and print its expected input schema
            live_context_tool = next(t for t in tools_response.tools if t.name == "GetLiveContext")
            print("GetLiveContext input schema:")
            print(json.dumps(live_context_tool.inputSchema, indent=2))
            print()

            # Now actually call it
            print("Calling GetLiveContext...\n")
            result = await session.call_tool("GetLiveContext", arguments={})

            for content_block in result.content:
                if content_block.type == "text":
                    print(content_block.text)

if __name__ == "__main__":
    asyncio.run(main())
