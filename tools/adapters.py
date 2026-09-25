def mcp_tools_to_openai(mcp_tools):
    """Convert MCP tool definitions to OpenAI-style function-calling schema.

    Used directly by OpenRouter and Ollama, since both speak the OpenAI
    tool-calling dialect natively. Gemini needs a further conversion on top
    of this - see clean_schema_for_gemini() below and llm/gemini.py.
    """
    return [{"type": "function", "function": {
        "name": t.name, "description": t.description, "parameters": t.inputSchema
    }} for t in mcp_tools]


# Confirmed via a live Gemini 400 response that "additionalProperties" is
# rejected ("Unknown name \"additionalProperties\" ... Cannot find field").
# The rest of this set is included preemptively based on Gemini's documented
# restricted OpenAPI 3.0 subset for functionDeclarations, but hasn't
# individually been confirmed to cause a 400 - if a future error names a
# different field, add it here.
UNSUPPORTED_GEMINI_SCHEMA_KEYS = {"additionalProperties", "$schema", "title", "default", "examples"}


def clean_schema_for_gemini(schema):
    """Recursively strip JSON Schema keys that Gemini's functionDeclarations
    parameters don't accept, since Gemini validates against a restricted
    OpenAPI subset rather than full JSON Schema."""
    if isinstance(schema, dict):
        return {
            k: clean_schema_for_gemini(v)
            for k, v in schema.items()
            if k not in UNSUPPORTED_GEMINI_SCHEMA_KEYS
        }
    elif isinstance(schema, list):
        return [clean_schema_for_gemini(item) for item in schema]
    return schema
