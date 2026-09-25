import json

import httpx

from config import GEMINI_KEY, GEMINI_MODEL
from llm.base import LLMBackend
from tools.adapters import clean_schema_for_gemini


class GeminiBackend(LLMBackend):
    name = "Gemini"

    def __init__(self):
        super().__init__()
        self.last_model_used = GEMINI_MODEL

    async def complete(self, messages, tools=None):
        payload = {
            "contents": self._to_gemini_messages(messages),
            "tools": self._to_gemini_tools_payload(tools),
            "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}}
        }

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}"
        )

        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=payload, timeout=30)
            if r.status_code >= 400:
                print(f"  [Gemini {r.status_code} body: {r.text[:500]}]")
            r.raise_for_status()
            data = r.json()

        return self._from_gemini_response(data)

    # NOTE: system and tool-result messages are folded into "user" turns here,
    # ported as-is from the original implementation. This is UNCONFIRMED to be
    # correct - Gemini enforces strict user/model alternation in `contents`,
    # and two consecutive "user" turns (system_msg + first user_msg) or an
    # empty-text "model" turn (after a tool call) are both plausible sources
    # of a 400 on later calls. Not changed here since we haven't reproduced
    # that specific failure yet - flag if you hit a 400 during/after a tool
    # call round-trip.
    @staticmethod
    def _to_gemini_messages(messages):
        gemini_messages = []
        for m in messages:
            role = m["role"]
            if role == "system":
                gemini_messages.append({"role": "user", "parts": [{"text": f"[System instruction: {m['content']}]"}]})
            elif role == "user":
                gemini_messages.append({"role": "user", "parts": [{"text": m["content"]}]})
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [{"text": m.get("content", "")}]})
            elif role == "tool":
                gemini_messages.append({"role": "user", "parts": [{"text": f"[Tool result: {m['content']}]"}]})
        return gemini_messages

    @staticmethod
    def _to_gemini_tools_payload(tools):
        if not tools:
            return []

        gemini_tools = []
        for t in tools:
            fn = t["function"]
            gemini_tools.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": clean_schema_for_gemini(fn.get("parameters", {"type": "object", "properties": {}}))
            })
        return [{"functionDeclarations": gemini_tools}]

    @staticmethod
    def _from_gemini_response(data):
        candidate = data["candidates"][0]
        content = candidate["content"]
        parts = content.get("parts", [])

        msg = {"role": "assistant", "content": ""}
        for part in parts:
            if "text" in part:
                msg["content"] += part["text"]
            elif "functionCall" in part:
                fc = part["functionCall"]
                msg.setdefault("tool_calls", []).append({
                    "id": fc["name"],
                    "type": "function",
                    "function": {
                        "name": fc["name"],
                        "arguments": json.dumps(fc.get("args", {}))
                    }
                })
        return msg
