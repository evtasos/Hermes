"""
Fact extraction and storage with better prompt and safe fallback (returns []).
"""
import json
import re
from typing import List, Optional

import structlog

from llm.router import get_llm_response
from memory.store import store_fact, get_all_facts
from memory.dedup import is_duplicate_fact
from memory.embedder import get_embedder

logger = structlog.get_logger()


FACT_EXTRACTION_PROMPT = """You are a memory extraction system.
Extract ONLY concrete, verifiable facts from the user's message.
Ignore the assistant's response unless it contains new user-directed information.

RULES:
- Extract ONLY what the user stated explicitly or what is directly observable.
- Do NOT extract generic knowledge, opinions, or assumptions.
- One fact per line, max 20 words.
- Return a JSON array of strings. If no facts, return [].
- Do NOT include markdown, explanation, or extra text.

Examples:
User: "I bought a cat" → ["Owner bought a cat"]
User: "My wife's birthday is July 15" → ["Wife's birthday is July 15"]
User: "Turn on the kitchen light" → ["Owner has a kitchen light"]
User: "What's the weather?" → []

Now extract facts from this exchange:
User: {user_msg}
Assistant: {assistant_msg}

Output ONLY a JSON array like: ["fact 1", "fact 2"]
"""


async def extract_and_store_facts(
    user_msg: str,
    assistant_msg: str,
    mcp_client: Optional[object] = None
) -> None:
    """
    Extract facts from the turn and store them if not duplicates.
    Runs as a background task; errors are logged but not propagated.
    """
    try:
        prompt = FACT_EXTRACTION_PROMPT.format(
            user_msg=user_msg, assistant_msg=assistant_msg
        )
        messages = [
            {"role": "system", "content": "You extract memories as JSON arrays."},
            {"role": "user", "content": prompt},
        ]
        response = await get_llm_response(messages, tools=None)
        content = response.get("content", "").strip()

        # Parse JSON array
        try:
            facts = json.loads(content)
            if not isinstance(facts, list):
                facts = []
        except json.JSONDecodeError:
            # Safe fallback: return empty list (no guessing)
            facts = []

        # Clean and deduplicate
        embedder = get_embedder()
        existing_facts = get_all_facts()
        for fact in facts:
            fact_str = str(fact).strip()
            if not fact_str or len(fact_str) < 5:
                continue
            if not is_duplicate_fact(fact_str, existing_facts, embedder):
                store_fact(fact_str)
                existing_facts.append(fact_str)

    except Exception as e:
        logger.error("fact_extraction_failed", error=str(e))
        # Silently ignore; don't break the turn