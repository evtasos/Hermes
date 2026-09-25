"""
Gemini provider using direct REST API.
Auto-fallback chain: gemini-1.5-flash → gemini-1.5-flash-8b → gemini-1.0-pro
"""
import json
import os
from typing import List, Dict, Any, Optional

import httpx
import structlog

from config import GEMINI_API_KEY, GEMINI_MODEL_FALLBACKS

logger = structlog.get_logger()

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


async def gemini_chat(
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Call Gemini API with OpenAI-compatible message format.
    Returns normalized OpenAI-style response dict.
    """
    api_key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    # Convert OpenAI messages to Gemini contents
    contents = []
    system_texts = []

    for msg in messages:
        role = msg["role"]
        if role == "system":
            system_texts.append(msg["content"])
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": msg["content"]}]})
        elif role == "tool":
            # Tool results become user messages with prefix
            contents.append(
                {
                    "role": "user",
                    "parts": [{"text": f"[Tool result: {msg['content']}]"}],
                }
            )

    # Prepend system texts to first user message
    if system_texts and contents:
        first_user = contents[0]
        if first_user["role"] == "user":
            first_user["parts"][0]["text"] = (
                " ".join(system_texts) + "\n\n" + first_user["parts"][0]["text"]
            )
        else:
            contents.insert(
                0,
                {"role": "user", "parts": [{"text": " ".join(system_texts)}]},
            )

    # Build payload
    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1024,
        },
    }

    if tools:
        function_declarations = []
        for tool in tools:
            if tool.get("type") == "function":
                fn = tool["function"]
                function_declarations.append(
                    {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "parameters": fn["parameters"],
                    }
                )
        if function_declarations:
            payload["tools"] = [{"functionDeclarations": function_declarations}]

    url = GEMINI_API_URL.format(model=model)
    headers = {"Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{url}?key={api_key}",
            json=payload,
            headers=headers,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(
                "gemini_http_error",
                status=e.response.status_code,
                text=e.response.text,
            )
            raise

    data = response.json()

    # Extract text and potential function calls
    try:
        candidate = data["candidates"][0]
        content_parts = candidate.get("content", {}).get("parts", [])
        text_parts = []
        function_calls = []

        for part in content_parts:
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                function_calls.append(
                    {
                        "id": f"call_{hash(str(fc))}",
                        "type": "function",
                        "function": {
                            "name": fc["name"],
                            "arguments": json.dumps(fc.get("args", {})),
                        },
                    }
                )

        text = "".join(text_parts).strip()

        if function_calls:
            return {
                "content": None,
                "tool_calls": function_calls,
            }
        else:
            return {
                "content": text,
                "tool_calls": [],
            }
    except (KeyError, IndexError, TypeError) as e:
        logger.error("gemini_parse_error", error=str(e), data=data)
        raise RuntimeError(f"Failed to parse Gemini response: {e}") from e


async def gemini_chat_with_fallback(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Try each model in GEMINI_MODEL_FALLBACKS until one succeeds.
    """
    last_error: Optional[Exception] = None
    for model in GEMINI_MODEL_FALLBACKS:
        try:
            response = await gemini_chat(model, messages, tools)
            logger.info("gemini_success", model=model)
            return response
        except Exception as e:
            last_error = e
            logger.warning("gemini_attempt_failed", model=model, error=str(e))
            # On 429 (quota) or 503 (overloaded), try next model immediately
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in {429, 503}:
                continue
            # For other errors, break and let caller try next provider
            break
    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}") from last_error