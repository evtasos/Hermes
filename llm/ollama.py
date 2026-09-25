import httpx

from config import OLLAMA_HOST, OLLAMA_MODEL
from llm.base import LLMBackend


class OllamaBackend(LLMBackend):
    name = "Ollama"

    def __init__(self):
        super().__init__()
        self.last_model_used = OLLAMA_MODEL

    async def complete(self, messages, tools=None):
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "tools": tools,
                    "stream": False,
                    "think": False
                },
                timeout=120
            )
            r.raise_for_status()
            return r.json()["message"]
