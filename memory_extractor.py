import json

FACT_EXTRACTION_PROMPT = """You are a memory extraction system.
Given a user message and an assistant response, extract ONLY concrete facts about the USER and their HOME.

CRITICAL RULES:
1. Extract ONLY from what the USER said. IGNORE the assistant's confirmations, filler, and restatements.
2. NEVER extract generic knowledge (e.g., "cats are pets", "water is wet"). Only extract specific facts about THIS user.
3. Output ONLY a JSON array of strings. No markdown, no explanation.

Examples:
User: "I bought a cat"
Assistant: "Great! I'll remember that."
→ ["Owner has a cat"]

User: "My wife's birthday is July 15"
Assistant: "Got it."
→ ["Wife's birthday is July 15"]

User: "is it hot in the Baby room?"
Assistant: "It's 29.7°C in Bedroom Kid."
→ ["Bedroom Kid is the area with the baby", "Bedroom Kid temperature sensor reads 29.7°C"]

BAD examples (DO NOT extract these):
User: "I bought a cat"
Assistant: "You have a cat as a pet."
→ ["Cat is a pet"]  ← WRONG: generic knowledge

Now extract facts from this exchange:
User: {user_msg}
Assistant: {assistant_msg}

Output ONLY a JSON array like: ["fact 1", "fact 2"]"""

async def extract_facts(user_msg: str, assistant_msg: str, llm_caller) -> list[str]:
    prompt = FACT_EXTRACTION_PROMPT.format(user_msg=user_msg, assistant_msg=assistant_msg)
    messages = [
        {"role": "system", "content": "You extract long-term memories as JSON arrays."},
        {"role": "user", "content": prompt}
    ]
    response = await llm_caller(messages, tools=None)
    content = response.get("content", "").strip()
    
    try:
        facts = json.loads(content)
        if isinstance(facts, list):
            return [str(f).strip() for f in facts if f and len(str(f).strip()) > 5]
    except json.JSONDecodeError:
        pass
    
    lines = [l.strip().strip('"').strip("'").strip("- ").strip("* ").strip("→ ").strip("[]")
             for l in content.split("\n") if l.strip()]
    facts = [l for l in lines if len(l) > 10 and not l.lower().startswith(("here", "output", "json", "fact", "user:", "assistant:", "extract"))]
    return facts[:5]
