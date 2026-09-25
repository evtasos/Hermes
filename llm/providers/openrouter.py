"""
OpenRouter provider (optional stub - unreliable tool support).
Kept for compatibility but not recommended as primary.
"""
import json
from typing import List, Dict, Any, Optional

import httpx
import structlog

logger = structlog.get_logger()


async def openrouter_chat(
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    api_key: str = "",
) -> Dict[str, Any]:
    """
    Call OpenRouter chat completions endpoint.
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
        "messages": messages,
    }

    if tools:
        payload["tools"] = tools
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