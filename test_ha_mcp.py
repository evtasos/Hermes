python3 -c "
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def test():
    async with streamablehttp_client('http://192.168.1.75:9584/private_W__E8dVrYPQ62b-oTrP0og', headers={'Authorization': 'Bearer YOUR_HA_TOKEN'}) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            print(f'Tools exposed: {len(tools)}')
            for t in tools:
                print(f'  - {t.name}')

asyncio.run(test())