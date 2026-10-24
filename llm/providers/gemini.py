"""
Gemini provider using the native generateContent REST API.

Previous version converted `tool`-role messages into freeform
`{"role": "user", "parts": [{"text": "[Tool result: ...]"}]}` text, and
fabricated a function-call id with `hash()`. Neither is how Gemini's
function calling actually works: a model's functionCall must be preserved
in its own `model`-role turn, and the result goes back as a structured
`functionResponse` part, matched by name (and by id, only when Gemini
actually issued one - the field is optional).
"""
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx
import structlog

from config import GEMINI_API_KEY, GEMINI_MODEL_FALLBACKS
from hermes_mcp.schema import mcp_tools_to_gemini

logger = structlog.get_logger()

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Thinking is a real latency/token cost, and most home-automation turns
# ("turn on the kitchen light") don't benefit from it. "low" is the lowest
# setting that's valid across the whole fallback chain below - gemini-3.8-flash
# specifically errors on "minimal", so this is not a per-model choice, it's
# the one setting that works everywhere in the chain.
THINKING_LEVEL = "low"


def _build_contents(
    messages: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Convert internal messages into Gemini `contents`, preserving genuine
    functionCall/functionResponse round trips. Returns (contents, system_text).
    """
    contents: List[Dict[str, Any]] = []
    system_parts: List[str] = []
    call_id_to_name: Dict[str, str] = {}
    call_id_to_gemini_id: Dict[str, Optional[str]] = {}

    for msg in messages:
        role = msg["role"]

        if role == "system":
            system_parts.append(msg["content"])

        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": msg["content"]}]})

        elif role == "assistant":
            parts = []
            if msg.get("content"):
                parts.append({"text": msg["content"]})
            for call in msg.get("tool_calls") or []:
                fn = call["function"]
                args = fn["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                call_id_to_name[call["id"]] = fn["name"]
                gemini_id = call.get("_gemini_id")
                call_id_to_gemini_id[call["id"]] = gemini_id
                fc_part: Dict[str, Any] = {"name": fn["name"], "args": args}
                if gemini_id:
                    fc_part["id"] = gemini_id
                parts.append({"functionCall": fc_part})
            if parts:
                contents.append({"role": "model", "parts": parts})

        elif role == "tool":
            call_id = msg.get("tool_call_id", "")
            name = msg.get("name") or call_id_to_name.get(call_id, "unknown_function")
            fr_part: Dict[str, Any] = {
                "functionResponse": {
                    "name": name,
                    "response": {"result": msg["content"]},
                }
            }
            gemini_id = call_id_to_gemini_id.get(call_id)
            if gemini_id:
                fr_part["functionResponse"]["id"] = gemini_id
            contents.append({"role": "user", "parts": [fr_part]})

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return contents, system_instruction


async def gemini_chat(
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Call Gemini's generateContent endpoint for one model.
    `tools`, if given, are raw MCP tool objects - this function does the
    Gemini-specific schema cleaning itself rather than receiving a
    pre-converted (and possibly Gemini-incompatible) shape.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    contents, system_instruction = _build_contents(messages)

    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1024,
            "thinkingConfig": {"thinkingLevel": THINKING_LEVEL},
        },
    }
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    if tools:
        function_declarations = mcp_tools_to_gemini(tools)
        if function_declarations:
            payload["tools"] = [{"functionDeclarations": function_declarations}]

    url = GEMINI_API_URL.format(model=model)
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(
                "gemini_http_error",
                status=e.response.status_code,
                text=e.response.text[:500],
            )
            raise

    data = response.json()

    try:
        candidate = data["candidates"][0]
        content_parts = candidate.get("content", {}).get("parts", [])
    except (KeyError, IndexError) as e:
        logger.error("gemini_parse_error", error=str(e), data=data)
        raise RuntimeError(f"Failed to parse Gemini response: {e}") from e

    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for part in content_parts:
        if "text" in part:
            text_parts.append(part["text"])
        elif "functionCall" in part:
            fc = part["functionCall"]
            call_entry: Dict[str, Any] = {
                # Internal bookkeeping id (OpenRouter/OpenAI-shaped consumers
                # need *some* id). NOT Gemini's id - see _gemini_id below.
                "id": f"local-{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": fc["name"],
                    "arguments": json.dumps(fc.get("args", {})),
                },
            }
            if fc.get("id"):
                # Gemini's own id, only set when Gemini actually returned one
                # (the field is optional per Gemini's FunctionCall schema).
                # Echoed back verbatim in the matching functionResponse.
                call_entry["_gemini_id"] = fc["id"]
            tool_calls.append(call_entry)

    text = "".join(text_parts).strip()

    if tool_calls:
        return {"content": None, "tool_calls": tool_calls}
    return {"content": text, "tool_calls": []}


async def gemini_chat_with_fallback(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Try each model in GEMINI_MODEL_FALLBACKS in order. Steps to the next
    model on 429 (quota) / 503 (overloaded); anything else is a real error
    and is raised so the router can try the next *provider* instead.
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
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in {429, 503}:
                continue
            break
    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}") from last_error
