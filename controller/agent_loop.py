"""
Hermes agent loop: orchestrates recall → prompt → LLM → tool loop → reply → memory.
"""
import asyncio
import time
import json
import uuid
from typing import List, Dict, Any, Optional

import structlog

from llm.router import get_llm_response, get_last_model
from mcp.client import MCPClient
from mcp.schema import mcp_tools_to_openai
from memory.store import recall, remember, cap_conversations
from memory.facts import extract_and_store_facts
from observability.logger import TurnLogger
from safety.gate import write_gate
from cli import print_agent, print_tool_call, print_tool_result
from config import (
    MAX_CONVERSATION_TURNS,
    REQUIRE_WRITE_CONFIRMATION,
    LOG_TURNS,
    LOG_FILE,
)

logger = structlog.get_logger()


async def run_repl() -> None:
    """Main REPL loop."""
    # Initialize shared resources
    mcp_client = MCPClient()
    turn_logger = TurnLogger(LOG_FILE) if LOG_TURNS else None

    await mcp_client.initialize()
    meta_tools = await mcp_client.get_meta_tools()
    openai_tools = mcp_tools_to_openai(meta_tools)

    conversation_id = str(uuid.uuid4())
    history: List[Dict[str, Any]] = []

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

        def stage(name: str):
            def stopper():
                stages[name] = (time.perf_counter() - stage_start) * 1000
            return stopper

        # ---------- MEMORY RECALL ----------
        stage_start = time.perf_counter()
        recalled = recall(user_input, k=3)
        stages["memory_recall"] = (time.perf_counter() - stage_start) * 1000
        stage("memory_recall")()

        # ---------- BUILD PROMPT ----------
        stage_start = time.perf_counter()
        system_msg = (
            "You are Hermes, a helpful home assistant. "
            "Use tools to answer questions and control devices. "
            "Always confirm before executing actions that change state. "
            "Keep answers concise."
        )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_msg},
        ]
        if recalled:
            messages.append(
                {
                    "role": "system",
                    "content": "Relevant memories:\n" + "\n".join(f"- {r}" for r in recalled),
                }
            )
        messages.append({"role": "user", "content": user_input})
        stages["prompt_build"] = (time.perf_counter() - stage_start) * 1000
        stage("prompt_build")()

        # ---------- LLM CALL (may trigger tool use) ----------
        stage_start = time.perf_counter()
        llm_response = await get_llm_response(messages, openai_tools)
        stages["llm_1"] = (time.perf_counter() - stage_start) * 1000
        stage("llm_1")()

        # ---------- TOOL EXECUTION LOOP ----------
        tool_calls: List[Dict[str, Any]] = []
        while llm_response.get("tool_calls"):
            # Execute all tool calls in parallel
            for call in llm_response["tool_calls"]:
                fn_name = call["function"]["name"]
                try:
                    fn_args = json.loads(call["function"]["arguments"])
                except json.JSONDecodeError:
                    fn_args = {}

                print_tool_call(fn_name, fn_args)
                stage_start = time.perf_counter()

                # Safety gate for write tools
                if await mcp_client.is_write_tool(fn_name):
                    if REQUIRE_WRITE_CONFIRMATION:
                        confirmed = await write_gate(
                            fn_name, fn_args, user_input, mcp_client
                        )
                        if not confirmed:
                            tool_result = "Action cancelled by user."
                        else:
                            tool_result = await mcp_client.call_tool(fn_name, fn_args)
                    else:
                        tool_result = await mcp_client.call_tool(fn_name, fn_args)
                else:
                    tool_result = await mcp_client.call_tool(fn_name, fn_args)

                stages.setdefault("tool_execution", 0.0)
                stages["tool_execution"] += (time.perf_counter() - stage_start) * 1000

                result_text = (
                    "".join(c.text for c in tool_result.content if c.type == "text")
                    if hasattr(tool_result, "content")
                    else str(tool_result)
                )
                print_tool_result(fn_name, result_text)

                # Append tool result with proper role and tool_call_id
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": result_text,
                    }
                )
                tool_calls.append(
                    {"name": fn_name, "args": fn_args, "result": result_text}
                )

            # Next LLM call to refine after tool results
            stage_start = time.perf_counter()
            llm_response = await get_llm_response(messages, openai_tools)
            stages.setdefault("llm_2", 0.0)
            stages["llm_2"] += (time.perf_counter() - stage_start) * 1000
            stage("llm_2")()

        # ---------- FINAL REPLY ----------
        stage_start = time.perf_counter()
        reply = llm_response.get("content") or ""
        stages["llm_final"] = (time.perf_counter() - stage_start) * 1000
        stage("llm_final")()

        print_agent(reply)

        # ---------- MEMORY STORAGE & FACT EXTRACTION ----------
        stage_start = time.perf_counter()
        remember(user_input, "user")
        remember(reply, "assistant")
        cap_conversations(max_turns=MAX_CONVERSATION_TURNS)

        # Extract facts in background (non-blocking)
        asyncio.create_task(
            extract_and_store_facts(user_input, reply, mcp_client)
        )
        stages["memory_save"] = (time.perf_counter() - stage_start) * 1000
        stage("memory_save")()

        # ---------- TURN LOGGING ----------
        total_ms = (time.perf_counter() - start_time) * 1000
        if turn_logger:
            await turn_logger.log(
                turn_id=turn_id,
                conversation_id=conversation_id,
                user_input=user_input,
                agent_reply=reply,
                tool_calls=tool_calls,
                stages=stages,
                total_ms=total_ms,
                model_used=get_last_model() or "unknown",
            )

        # ---------- PROFILER SUMMARY (optional) ----------
        print("\n" + "─" * 60)
        print(f"Turn {turn_id} | Total: {total_ms:.0f}ms")
        for name, ms in stages.items():
            print(f"  {name:20}: {ms:6.1f}ms")
        print("─" * 60)