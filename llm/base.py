from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class LLMBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Returns a normalized OpenAI-compatible response dict:
        {
            "role": "assistant",
            "content": str or None,
            "tool_calls": [...] or None
        }
        """
        pass