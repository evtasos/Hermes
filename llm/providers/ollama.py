"""
Ollama provider using REST API.
Queries /api/ps to find currently loaded model in VRAM; falls back to default.
"""
import json
from typing import List, Dict, Any, Optional

import httpx
import structlog

from config import OLLAMA_HOST, OLLAMA_MODEL

logger = structlog.get_logger()


async def ollama_chat(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """
    Call Ollama /api/chat endpoint with OpenAI-compatible format.
    Uses currently loaded model if available, otherwise default.
    """
    # Find active model
    active_model = OLLAMA_MODEL
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{OLLAMA_HOST}/api/ps")
            if r.status_code == 200:
                running = r.json().get("models", [])
                if running:
                    active_model = running[0].get("name", OLLAMA_MODEL)
    except Exception:
        pass

    url = f"{OLLAMA_HOST}/api/chat"

    # Convert messages to Ollama format
    ollama_messages = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            # Ollama doesn't have system role; prepend to first user message
            continue
        elif role == "user":
            ollama_messages.append({"role": "user", "content": msg["content"]})
        elif role == "assistant":
            ollama_messages.append({"role": "assistant", "content": msg["content"]})
        elif role == "tool":
            ollama_messages.append({"role": "user", "content": f"[Tool result: {msg['content']}]"})

    payload = {
        "model": active_model,
        "messages": ollama_messages,
        "stream": False,
    }

    # Note: Ollama tool support is patchy; we ignore tools for now
    # The agent's two-step loop handles this
    if tools:
        logger.warning("ollama_tools_ignored", model=active_model, tool_count=len(tools))

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    message = data.get("message", {})
    content = message.get("content", "")

    return {
        "content": content,
        "tool_calls": [],
    }