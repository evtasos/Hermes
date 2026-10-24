"""
OpenRouter provider (optional stub - unreliable tool support).
Kept for compatibility but not recommended as primary.
"""
import json
from typing import List, Dict, Any, Optional

import httpx
import structlog

from hermes_mcp.schema import mcp_tools_to_openai

logger = structlog.get_logger()


def _sanitize_for_openai(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Strip internal-only bookkeeping keys before handing conversation history
    to an OpenAI-compatible endpoint - specifically Gemini's `_gemini_id`,
    which can end up in an assistant message's tool_calls if a conversation
    started on Gemini and later fell back to OpenRouter mid-turn.
    """
    clean = []
    for msg in messages:
        msg = dict(msg)
        if msg.get("tool_calls"):
            msg["tool_calls"] = [
                {k: v for k, v in call.items() if k in {"id", "type", "function"}}
                for call in msg["tool_calls"]
            ]
        clean.append(msg)
    return clean


async def openrouter_chat(
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Any]] = None,
    api_key: str = "",
) -> Dict[str, Any]:
    """
    Call OpenRouter chat completions endpoint.
    `tools`, if given, are raw MCP tool objects - converted to OpenAI format
    here rather than upstream, since Gemini needs a different shape for the
    same tools.
    """
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": model,
        "messages": _sanitize_for_openai(messages),
    }

    if tools:
        openai_tools = mcp_tools_to_openai(tools)
        if openai_tools:
            payload["tools"] = openai_tools
            payload["tool_choice"] = "auto"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    choice = data["choices"][0]
    message = choice["message"]

    content = message.get("content")
    tool_calls = message.get("tool_calls")

    return {
        "content": content,
        "tool_calls": tool_calls or [],
    }