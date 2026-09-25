"""
Convert MCP tool definitions to OpenAI-compatible function schemas.
Adds `strict: True` for better reliability.
"""
from typing import List, Any, Dict


def mcp_tools_to_openai(mcp_tools: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert a list of MCP Tool objects to OpenAI function tool format.
    """
    openai_tools = []
    for tool in mcp_tools:
        # MCP tool has: name, description, inputSchema (JSON Schema)
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
                "strict": True,  # helps the model produce valid JSON
            },
        }
        openai_tools.append(openai_tool)
    return openai_tools