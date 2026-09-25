# Placeholder for shared dataclasses/types (e.g. a Message type used across
# llm/, tools/, and main.py) instead of passing raw dicts everywhere.
#
# Not implemented yet - the current refactor is a straight move of existing
# logic into modules, not a redesign of the message shape. Every backend in
# llm/ and the tool-call loop in main.py currently pass around plain dicts
# like {"role": ..., "content": ..., "tool_calls": [...]}. Introducing a
# dataclass here means updating all of those call sites at once, so it's
# left as a deliberate follow-up rather than bundled into this pass.
