import asyncio, os, json
import httpx, chromadb
from datetime import datetime
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from profiler import Profiler
from memory_extractor import extract_facts
from git_manager import git_pull, git_push, git_status, git_log
import time

load_dotenv()

OPENROUTER_KEY = os.environ["OPENROUTER_API_KEY"]
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
OLLAMA_HOST = "http://192.168.1.90:11434"
OLLAMA_MODEL = "qwen3.5:0.8b"
GEMINI_MODEL = "gemini-flash-latest"
HA_URL = os.environ["HA_MCP_URL"]
HA_TOKEN = os.environ["HA_TOKEN"]

last_model_used = "unknown"
pending_extractions = []

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma = chromadb.PersistentClient(path="./hermes_memory")
memory = chroma.get_or_create_collection("conversations")
facts_collection = chroma.get_or_create_collection("facts")


def remember(text, role):
    if text is None:
        text = ""
    emb = embedder.encode(text).tolist()
    memory.add(
        ids=[f"{datetime.now().isoformat()}-{role}"],
        embeddings=[emb], documents=[text],
        metadatas=[{"role": role, "ts": datetime.now().isoformat()}]
    )


def store_fact(text: str):
    if text is None:
        return
    if facts_collection.count() > 0:
        emb = embedder.encode(text).tolist()
        results = facts_collection.query(
            query_embeddings=[emb],
            n_results=3,
            include=["documents", "distances"]
        )
        if results.get("distances") and results["distances"][0]:
            for doc, dist in zip(results["documents"][0], results["distances"][0]):
                if dist < 0.15:
                    print(f"  [fact duplicate skipped (dist={dist:.3f}): {text}]")
                    return
                text_norm = text.lower().replace("user", "owner").replace("i ", "owner ")
                doc_norm = doc.lower().replace("user", "owner").replace("i ", "owner ")
                if text_norm == doc_norm:
                    print(f"  [fact duplicate skipped (text match): {text}]")
                    return

    emb = embedder.encode(text).tolist()
    facts_collection.add(
        ids=[f"{datetime.now().isoformat()}-fact"],
        embeddings=[emb],
        documents=[text],
        metadatas=[{
            "type": "fact",
            "created": datetime.now().isoformat(),
            "source": "conversation"
        }]
    )
    print(f"  [fact stored: {text}]")


def cap_conversations(max_turns=20):
    count = memory.count()
    if count > max_turns:
        all_data = memory.get()
        ids = all_data["ids"]
        to_delete = ids[:-max_turns]
        memory.delete(ids=to_delete)
        print(f"  [pruned {len(to_delete)} old conversation turns]")


def recall(query, k=3):
    if facts_collection.count() == 0 and memory.count() == 0:
        return []

    emb = embedder.encode(query).tolist()
    seen = set()
    results = []

    def add_unique(docs):
        for doc in docs:
            key = doc.lower().strip(" .!?,;:")
            if key not in seen and len(key) > 3:
                seen.add(key)
                results.append(doc)

    if facts_collection.count() > 0:
        fact_results = facts_collection.query(
            query_embeddings=[emb],
            n_results=min(k, facts_collection.count())
        )
        if fact_results.get("documents") and fact_results["documents"][0]:
            add_unique(fact_results["documents"][0])

    if memory.count() > 0:
        conv_results = memory.query(
            query_embeddings=[emb],
            n_results=min(k, memory.count())
        )
        if conv_results.get("documents") and conv_results["documents"][0]:
            add_unique(conv_results["documents"][0])

    return results[:k]


def mcp_tools_to_openai(mcp_tools):
    return [{"type": "function", "function": {
        "name": t.name, "description": t.description, "parameters": t.inputSchema
    }} for t in mcp_tools]


async def call_openrouter(messages, tools):
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
            json={"model": "openrouter/free", "messages": messages, "tools": tools},
            timeout=30
        )
        
        r.raise_for_status()
        data = r.json()
        global last_model_used
        last_model_used = data.get("model", "openrouter/unknown")
        return data["choices"][0]["message"]

# ----tool calling is wrong
async def call_gemini(messages, tools):
    gemini_messages = []
    for m in messages:
        role = m["role"]
        if role == "system":
            gemini_messages.append({"role": "user", "parts": [{"text": f"[System instruction: {m['content']}]"}]})
        elif role == "user":
            gemini_messages.append({"role": "user", "parts": [{"text": m["content"]}]})
        elif role == "assistant":
            gemini_messages.append({"role": "model", "parts": [{"text": m.get("content", "")}]})
        elif role == "tool":
            gemini_messages.append({"role": "user", "parts": [{"text": f"[Tool result: {m['content']}]"}]})

    gemini_tools = []
    if tools:
        for t in tools:
            fn = t["function"]
            gemini_tools.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}})
            })

    payload = {
        "contents": gemini_messages,
        "tools": [{"functionDeclarations": gemini_tools}] if gemini_tools else [],
        "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}}
    }

    model = GEMINI_MODEL     ## check generatecontent models gemini-flash-latest works
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}"

    async with httpx.AsyncClient() as client:
        r = await client.post(url, json=payload, timeout=30)
        if r.status_code >= 400:
            print(f"  [Gemini 400 body: {r.text[:500]}]")
        r.raise_for_status()
        data = r.json()

    global last_model_used
    last_model_used = model

    candidate = data["candidates"][0]
    content = candidate["content"]
    parts = content.get("parts", [])

    msg = {"role": "assistant", "content": ""}
    for part in parts:
        if "text" in part:
            msg["content"] += part["text"]
        elif "functionCall" in part:
            fc = part["functionCall"]
            msg.setdefault("tool_calls", []).append({
                "id": fc["name"],
                "type": "function",
                "function": {
                    "name": fc["name"],
                    "arguments": json.dumps(fc.get("args", {}))
                }
            })
    return msg


async def call_ollama_chat(messages, tools):
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
        global last_model_used
        last_model_used = OLLAMA_MODEL
        return r.json()["message"]


# ─── GEMINI IS PRIMARY ───
async def call_llm(messages, tools=None):
    backends = []
    if GEMINI_KEY:
        backends.append(("Gemini", call_gemini))
    if OPENROUTER_KEY:
        backends.append(("OpenRouter", call_openrouter))
    backends.append(("Ollama", call_ollama_chat))

    for name, fn in backends:
        try:
            return await fn(messages, tools)
        except Exception as e:
            print(f"  [router: {name} failed: {str(e)[:100]}]")
            continue
    
    print("  [router: ALL BACKENDS FAILED]")
    return {"role": "assistant", "content": "All LLM backends failed."}


async def get_llm_response(messages, tools):
    start = time.time()
    msg = await call_llm(messages, tools)
    elapsed = time.time() - start
    print(f"  [via {last_model_used}, {elapsed:.2f}s]")
    return msg


async def extract_and_store(user_msg: str, assistant_msg: str):
    try:
        facts = await extract_facts(user_msg, assistant_msg, call_llm)
        if not facts:
            return
        for fact in facts:
            store_fact(fact)
    except Exception as e:
        print(f"  [fact extraction failed: {e}]")


async def main():
    global pending_extractions

    print("  [checking for updates...]")
    print(f"  {git_pull()}")

    async with streamablehttp_client(HA_URL, headers={"Authorization": f"Bearer {HA_TOKEN}"}) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            openai_tools = mcp_tools_to_openai(mcp_tools)

            print(f"Connected. {len(mcp_tools)} tools available.\n")
            for t in mcp_tools:
                print(f"  - {t.name}")
            print()

            print("Hermes Agent (tool-calling enabled) — type 'exit' to quit")

            while True:
                user_input = await asyncio.to_thread(input, "\nYou: ")
                user_input = user_input.strip()

                if user_input.lower() in ("exit", "quit"):
                    if pending_extractions:
                        print(f"  [waiting for {len(pending_extractions)} fact extractions...]")
                        await asyncio.gather(*pending_extractions, return_exceptions=True)
                    break

                # ─── GIT COMMANDS ───
                if user_input.lower() == "pull":
                    print("  [pulling latest code...]")
                    print(f"  {git_pull()}")
                    continue

                if user_input.lower() == "status":
                    print(f"  {git_status()}")
                    continue

                if user_input.lower() == "log":
                    print(f"  {git_log()}")
                    continue

                if user_input.lower().startswith("push "):
                    msg = user_input[5:].strip() or "Agent update"
                    print(f"  [pushing: {msg}]")
                    print(f"  {git_push(msg)}")
                    continue

                profiler = Profiler()
                profiler.start("TOTAL")

                profiler.start("Memory Recall")
                memories = recall(user_input)
                profiler.stop("Memory Recall")

                fact_memories = [m for m in memories if not m.startswith(("You: ", "Agent: ", "do i have", "i bought"))]
                if fact_memories:
                    print(f"  [facts: {fact_memories}]")

                context = "\n".join(f"- {m}" for m in memories) if memories else "No relevant memory yet."

                profiler.start("Prompt Build")
                system_msg = {
                    "role": "system",
                    "content": (
                        "You are Hermes, a personal home assistant connected to Home Assistant.\n\n"
                        "TOOL USAGE RULES:\n"
                        "- To FIND a capability (e.g., 'how do I create an automation?'), use ha_search_tools.\n"
                        "- To READ data (e.g., list automations, get sensor state), use ha_call_read_tool with the tool name found via ha_search_tools.\n"
                        "- To CREATE or UPDATE (e.g., new automation, scene, script), use ha_call_write_tool with the tool name and arguments.\n"
                        "- To DELETE, use ha_call_delete_tool.\n"
                        "- ha_search is for searching Home Assistant ENTITIES (devices, sensors, areas), NOT for finding tools.\n"
                        "- Always search for the tool first, then execute it. Do NOT guess tool names.\n\n"
                        "You remember the following facts about the user and home:\n"
                        f"{context}\n\n"
                        "Use what you remember when answering. If you don't know something, say so."
                    )
                }
                messages = [system_msg, {"role": "user", "content": user_input}]
                profiler.stop("Prompt Build")

                profiler.set("Tools count", len(openai_tools))
                profiler.set("Tools chars", len(json.dumps(openai_tools)))
                profiler.set("Messages chars", len(json.dumps(messages)))
                profiler.set("System chars", len(system_msg["content"]))
                profiler.set("User chars", len(user_input))

                profiler.start("LLM 1")
                msg = await get_llm_response(messages, openai_tools)
                profiler.stop("LLM 1")
                profiler.set("Model", last_model_used)

                tool_calls = msg.get("tool_calls")

                if tool_calls:
                    messages.append(msg)
                    profiler.start("Tool execution")
                    for call in tool_calls:
                        fn_name = call["function"]["name"]
                        fn_args = call["function"]["arguments"]
                        if isinstance(fn_args, str):
                            fn_args = json.loads(fn_args)
                        print(f"  [calling tool: {fn_name}({fn_args})]")
                        result = await session.call_tool(fn_name, arguments=fn_args)
                        result_text = "".join(c.text for c in result.content if c.type == "text")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.get("id", fn_name),
                            "content": result_text
                        })
                    profiler.stop("Tool execution")
                    profiler.start("LLM 2")
                    final_msg = await get_llm_response(messages, openai_tools)
                    profiler.stop("LLM 2")
                    reply = final_msg.get("content") or ""
                else:
                    reply = msg.get("content") or ""

                print(f"Agent: {reply}")

                # Only extract facts if we got a real reply and no tool call errors
                if reply and not reply.startswith("Error") and len(reply) > 10:
                    task = asyncio.create_task(extract_and_store(user_input, reply))
                    pending_extractions.append(task)

                profiler.start("Memory Save")
                remember(user_input, "user")
                remember(reply, "assistant")
                cap_conversations(max_turns=20)
                profiler.stop("Memory Save")

                profiler.stop("TOTAL")
                profiler.report()

                pending_extractions = [t for t in pending_extractions if not t.done()]

if __name__ == "__main__":
    asyncio.run(main())