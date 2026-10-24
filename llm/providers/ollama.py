"""
Ollama provider using REST API.
"""
import json
from typing import List, Dict, Any, Optional

import httpx
import structlog

from config import OLLAMA_HOST, OLLAMA_MODEL

logger = structlog.get_logger()


async def _resolve_model() -> str:
    """
    OLLAMA_MODEL, if set, is an explicit override and is used as-is - for
    when you've settled on a model and want it locked in. Left unset (the
    default), this uses whatever model is currently loaded in Ollama, so
    models can be swapped on the Ollama side (`ollama run <model>`) without
    touching Hermes config while testing.

    Raises clearly if nothing is loaded, rather than silently falling back
    to a placeholder model name that may not even be pulled - that silent
    fallback (to "phi3:mini", which wasn't actually available) is what
    produced a 404 the first time this ran.
    """
    if OLLAMA_MODEL:
        return OLLAMA_MODEL

    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(f"{OLLAMA_HOST}/api/ps")
        r.raise_for_status()
        running = r.json().get("models", [])

    if not running:
        raise RuntimeError(
            f"No model currently loaded in Ollama at {OLLAMA_HOST}, and "
            "OLLAMA_MODEL isn't set. Load one first (`ollama run <model>`) "
            "or set OLLAMA_MODEL to pin a specific one."
        )
    return running[0]["name"]


async def ollama_chat(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Call Ollama /api/chat endpoint."""
    model = await _resolve_model()
    url = f"{OLLAMA_HOST}/api/chat"

    # Convert messages to Ollama format.
    # /api/chat does accept a system role directly, but Ollama models vary in
    # how reliably they attend to it - folding it into the first user turn
    # is the more conservative choice. Either way it must not be dropped:
    # an earlier version's comment claimed this prepend already happened,
    # but the code just skipped system messages outright, so Hermes ran with
    # zero persona/rules whenever it fell back to Ollama.
    ollama_messages = []
    pending_system = ""
    for msg in messages:
        role = msg["role"]
        if role == "system":
            pending_system += msg["content"] + "\n\n"
        elif role == "user":
            content = pending_system + msg["content"] if pending_system else msg["content"]
            pending_system = ""
            ollama_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            ollama_messages.append({"role": "assistant", "content": msg.get("content") or ""})
        elif role == "tool":
            ollama_messages.append({"role": "user", "content": f"[Tool result: {msg['content']}]"})

    payload = {
        "model": model,
        "messages": ollama_messages,
        "stream": False,
    }

    # Ollama/qwen tool-calling was tried before and was unreliable (wrong
    # tools, hallucinated parameters) - ignored here deliberately rather than
    # silently mis-executing. IMPORTANT: this means a fallback to Ollama can
    # only ever produce a conversational answer, never a real action. The
    # agent loop must not treat an Ollama response as if the requested
    # action happened - a text-only "sure, done" from this path would be a
    # false success.
    if tools:
        logger.warning("ollama_tools_ignored", model=model, tool_count=len(tools))

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
