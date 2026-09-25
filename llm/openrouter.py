import httpx

from config import OPENROUTER_KEY
from llm.base import LLMBackend


class OpenRouterBackend(LLMBackend):
    name = "OpenRouter"

    async def complete(self, messages, tools=None):
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
                json={"model": "openrouter/free", "messages": messages, "tools": tools},
                timeout=30
            )
            r.raise_for_status()
            data = r.json()
            self.last_model_used = data.get("model", "openrouter/unknown")
            return data["choices"][0]["message"]
