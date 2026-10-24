"""
Convert MCP tool definitions into each provider's own function-calling schema.

Each provider gets its own converter instead of one shape being forced on
every provider - Gemini's schema dialect (OpenAPI 3.0 subset) rejects things
plain JSON Schema (what MCP servers emit) allows, and OpenAI-style strict
mode has its own separate requirements. Trying to satisfy both with one
converted shape satisfies neither.
"""
from typing import Any, Dict, List


def mcp_tools_to_openai(mcp_tools: List[Any]) -> List[Dict[str, Any]]:
    """OpenAI/OpenRouter-compatible tool format."""
    openai_tools = []
    for tool in mcp_tools:
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
        )
    return openai_tools


# Keys that plain JSON Schema (what MCP servers emit) allows but Gemini's
# OpenAPI 3.0 subset hard-rejects. `additionalProperties` is the one that's
# actually been observed to fail against a real ha-mcp schema; `$schema` and
# `title` are harmless metadata Gemini doesn't need either, stripped so the
# payload stays small.
_GEMINI_UNSUPPORTED_KEYS = {"additionalProperties", "$schema", "title"}


def clean_schema_for_gemini(schema: Any) -> Any:
    """
    Recursively strip JSON Schema keys Gemini's function-calling schema
    doesn't accept, wherever they appear in the tree. A prior version only
    recursed into `properties`/`items` and missed `additionalProperties`
    nested inside `anyOf` (the common Optional[str]-style pattern from
    pydantic-generated schemas) - confirmed by a real 400 from Gemini citing
    exactly that path. Walking every dict/list generically instead of
    enumerating JSON Schema keywords by name (properties, items, anyOf,
    oneOf, allOf, ...) means this doesn't need to be extended every time a
    new nesting shape shows up.
    """
    if isinstance(schema, dict):
        return {
            k: clean_schema_for_gemini(v)
            for k, v in schema.items()
            if k not in _GEMINI_UNSUPPORTED_KEYS
        }
    if isinstance(schema, list):
        return [clean_schema_for_gemini(item) for item in schema]
    return schema


def mcp_tools_to_gemini(mcp_tools: List[Any]) -> List[Dict[str, Any]]:
    """Gemini `functionDeclarations` format, schema-cleaned for Gemini's dialect."""
    declarations = []
    for tool in mcp_tools:
        declarations.append(
            {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": clean_schema_for_gemini(tool.inputSchema),
            }
        )
    return declarations
