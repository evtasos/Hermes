"""
Hermes agent loop: orchestrates recall -> prompt -> LLM -> tool loop -> reply -> memory.
"""
import asyncio
import time
import json
import uuid
from typing import Any, Dict, List

import structlog

from llm.router import get_llm_response, get_last_model
from hermes_mcp.manager import MCPManager, ServerConfig
from memory.store import recall, remember, cap_conversations
from memory.facts import extract_and_store_facts
from observability.logger import TurnLogger
from safety.gate import write_gate
from cli import print_agent, print_tool_call, print_tool_result
from config import (
    MAX_CONVERSATION_TURNS,
    MAX_AGENT_ITERATIONS,
    LOG_TURNS,
    LOG_FILE,
    MCP_SERVERS,
)

logger = structlog.get_logger()

# Kept 100% static (persona + rules only, no interpolation) so the local
# Ollama/llama.cpp prefix cache can actually be reused turn to turn. Anything
# that varies (recalled memory) is folded into the user turn instead - see
# below - never appended here as a second system message the way the
# previous version did, since that broke the exact same cache.
STATIC_SYSTEM_PROMPT = (
    "You are Hermes, a helpful home assistant. "
    "Use tools to answer questions and control devices. "
    "Always confirm before executing actions that change state. "
    "Keep answers concise."
)


async def _run_tool(manager: MCPManager, fn_name: str, fn_args: Dict[str, Any]) -> str:
    try:
        result = await manager.call_tool(fn_name, fn_args)
    except Exception as e:
        logger.error("tool_call_failed", tool=fn_name, error=str(e))
        return f"Error calling {fn_name}: {e}"
    if hasattr(result, "content"):
        return "".join(c.text for c in result.content if getattr(c, "type", None) == "text")
    return str(result)


async def run_repl() -> None:
    """Main REPL loop."""
    mcp_manager = MCPManager()
    for server in MCP_SERVERS:
        await mcp_manager.add_server(ServerConfig(**server))

    turn_logger = TurnLogger(LOG_FILE) if LOG_TURNS else None
    conversation_id = str(uuid.uuid4())

    try:
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Goodbye!")
                break

            if user_input.lower() in {"exit", "quit"}:
                print("👋 Goodbye!")
                break
            if not user_input:
                continue

            turn_id = str(uuid.uuid4())[:8]
            start_time = time.perf_counter()
            stages: Dict[str, float] = {}

            # ---------- MEMORY RECALL ----------
            stage_start = time.perf_counter()
            recalled = recall(user_input, k=3)
            stages["memory_recall"] = (time.perf_counter() - stage_start) * 1000

            # ---------- BUILD PROMPT ----------
            # Recalled memory goes into the user turn, not a second system
            # message - the user turn is never cacheable prefix anyway
            # (it's new every turn), so this is the only place variable
            # context can go without invalidating the system-prompt prefix.
            stage_start = time.perf_counter()
            user_content = user_input
            if recalled:
                context_block = "Relevant memories:\n" + "\n".join(f"- {r}" for r in recalled)
                user_content = f"{context_block}\n\n{user_input}"

            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": STATIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            stages["prompt_build"] = (time.perf_counter() - stage_start) * 1000

            # ---------- LLM CALL (may trigger tool use) ----------
            # Raw MCP tool objects go to the router - each provider converts
            # to its own function-calling schema (Gemini's dialect and
            # OpenAI's are not the same shape, and forcing one converted
            # shape onto both satisfies neither).
            all_tools = mcp_manager.all_tools()
            stage_start = time.perf_counter()
            llm_response = await get_llm_response(messages, all_tools)
            stages["llm_1"] = (time.perf_counter() - stage_start) * 1000

            # ---------- TOOL EXECUTION LOOP ----------
            tool_calls_log: List[Dict[str, Any]] = []
            iterations = 0
            while llm_response.get("tool_calls"):
                iterations += 1
                if iterations > MAX_AGENT_ITERATIONS:
                    logger.warning("agent_max_iterations_hit", turn_id=turn_id)
                    messages.append(llm_response)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": "iteration-limit",
                        "name": "system",
                        "content": (
                            f"Stopped after {MAX_AGENT_ITERATIONS} tool-call rounds "
                            "without a final answer. Summarize what's been done so far "
                            "and stop calling tools."
                        ),
                    })
                    llm_response = await get_llm_response(messages, all_tools)
                    break

                # The model's own tool-call turn must be preserved in history
                # BEFORE the tool results are appended - every provider's
                # role sequence (and Gemini's function-response matching)
                # depends on it. This was the #1 protocol bug in the
                # previous version: tool-result messages were appended with
                # no corresponding assistant tool_calls turn before them.
                messages.append(llm_response)

                for call in llm_response["tool_calls"]:
                    fn_name = call["function"]["name"]
                    raw_args = call["function"]["arguments"]
                    try:
                        fn_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except json.JSONDecodeError as e:
                        # Never silently execute with {} on malformed args -
                        # tell the model it produced bad JSON and let the
                        # next loop iteration be its retry.
                        logger.warning("tool_call_bad_json", tool=fn_name, raw=raw_args, error=str(e))
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": fn_name,
                            "content": f"Error: arguments were not valid JSON ({e}). Re-call with valid JSON.",
                        })
                        continue

                    print_tool_call(fn_name, fn_args)
                    stage_start = time.perf_counter()

                    # Confirmation is decided by the tool's own MCP
                    # annotations (via the manager), never by hardcoding a
                    # tool name - this is what makes ha_delete_call (and any
                    # future server's destructive tools) safe by default
                    # instead of needing to be named explicitly.
                    if mcp_manager.needs_confirmation(fn_name):
                        confirmed = await write_gate(fn_name, fn_args, user_input)
                        if not confirmed:
                            result_text = "Action cancelled by user."
                        else:
                            result_text = await _run_tool(mcp_manager, fn_name, fn_args)
                    else:
                        result_text = await _run_tool(mcp_manager, fn_name, fn_args)

                    stages.setdefault("tool_execution", 0.0)
                    stages["tool_execution"] += (time.perf_counter() - stage_start) * 1000

                    print_tool_result(fn_name, result_text)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": fn_name,
                        "content": result_text,
                    })
                    tool_calls_log.append({"name": fn_name, "args": fn_args, "result": result_text})

                stage_start = time.perf_counter()
                llm_response = await get_llm_response(messages, all_tools)
                stages.setdefault("llm_2", 0.0)
                stages["llm_2"] += (time.perf_counter() - stage_start) * 1000

            # ---------- FINAL REPLY ----------
            reply = llm_response.get("content") or ""
            if not reply and tool_calls_log:
                # A tool already succeeded but the final narration call
                # failed - report the action, don't erase it behind a
                # generic "all providers failed" message. Whether an action
                # happened must never depend on whether the LLM call after
                # it also happened to succeed.
                last = tool_calls_log[-1]
                reply = (
                    f"Done: {last['name']} ran ({last['result'][:150]}), "
                    "but I couldn't generate a full response after that."
                )

            print_agent(reply)

            # ---------- MEMORY STORAGE & FACT EXTRACTION ----------
            stage_start = time.perf_counter()
            remember(user_input, "user")
            remember(reply, "assistant")
            cap_conversations(max_turns=MAX_CONVERSATION_TURNS)
            asyncio.create_task(extract_and_store_facts(user_input, reply))
            stages["memory_save"] = (time.perf_counter() - stage_start) * 1000

            # ---------- TURN LOGGING ----------
            total_ms = (time.perf_counter() - start_time) * 1000
            if turn_logger:
                await turn_logger.log(
                    turn_id=turn_id,
                    conversation_id=conversation_id,
                    user_input=user_input,
                    agent_reply=reply,
                    tool_calls=tool_calls_log,
                    stages=stages,
                    total_ms=total_ms,
                    model_used=get_last_model() or "unknown",
                )

            print("\n" + "─" * 60)
            print(f"Turn {turn_id} | Total: {total_ms:.0f}ms")
            for name, ms in stages.items():
                print(f"  {name:20}: {ms:6.1f}ms")
            print("─" * 60)
    finally:
        await mcp_manager.cleanup()
