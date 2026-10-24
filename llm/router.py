"""
LLM Router - tries providers in order with fallbacks.
"""
import httpx
from typing import List, Dict, Any, Optional

import structlog

from llm.providers.gemini import gemini_chat_with_fallback
from llm.providers.ollama import ollama_chat
from config import OPENROUTER_API_KEY, OPENROUTER_MODEL

logger = structlog.get_logger()

_last_model_used: Optional[str] = None


def set_last_model(model: str) -> None:
    global _last_model_used
    _last_model_used = model


def get_last_model() -> Optional[str]:
    return _last_model_used


async def get_llm_response(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Try LLM providers in order until one succeeds.

    Every provider returns raw {"content", "tool_calls"} - this is the one
    place responsible for shaping that into an actual assistant message
    before it goes anywhere near conversation history. Returning it without
    a "role" key was a real bug: it got appended straight into `messages`
    with no role, and every provider that indexes msg["role"] directly
    (Gemini, Ollama) threw KeyError('role') on the very next call, while
    OpenRouter - which doesn't index it locally - just forwarded the broken
    message and got a 400 back from the API instead.
    """
    # --- 1. Gemini (free tier) with auto-fallback ---
    try:
        response = await gemini_chat_with_fallback(messages, tools)
        set_last_model("gemini")
        return {"role": "assistant", **response}
    except Exception as e:
        logger.warning("gemini_failed", error=str(e))

    # --- 2. Ollama (local) ---
    try:
        response = await ollama_chat(messages, tools)
        set_last_model("ollama")
        return {"role": "assistant", **response}
    except Exception as e:
        logger.warning("ollama_failed", error=str(e))

    # --- 3. OpenRouter (optional, unreliable) ---
    if OPENROUTER_API_KEY:
        try:
            from llm.providers.openrouter import openrouter_chat
            response = await openrouter_chat(OPENROUTER_MODEL, messages, tools, OPENROUTER_API_KEY)
            set_last_model("openrouter")
            return {"role": "assistant", **response}
        except Exception as e:
            logger.warning("openrouter_failed", error=str(e))

    raise RuntimeError("All LLM providers failed")