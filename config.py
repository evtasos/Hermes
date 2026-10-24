"""
Hermes configuration.
Loads environment variables and provides defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- MCP servers ---
HA_MCP_URL = os.environ.get("HA_MCP_URL", "http://homeassistant.local:9584")
HA_TOKEN = os.environ.get("HA_TOKEN", "")

# Every server the MCPManager connects to at startup. Add more entries here
# as new MCP servers come online - the manager treats every entry the same
# way and never needs to know what a given server's tools are called.
MCP_SERVERS = [
    {"name": "ha", "url": HA_MCP_URL, "headers": {"Authorization": f"Bearer {HA_TOKEN}"}},
]

# --- LLM Router ---
# Primary: Gemini (free tier). Fallback: Ollama (local), then OpenRouter (optional)
# Gemini model list: tried in order, newest first; verified current as of
# Sept 2026 - re-check periodically since this generation is moving fast.
GEMINI_MODEL_FALLBACKS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
]

# Ollama host and model
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://192.168.1.90:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "")  # empty = auto-detect currently loaded model

# OpenRouter (optional, unreliable tool support)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/free")

# Gemini API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# --- Memory ---
MEMORY_PATH = Path(os.environ.get("MEMORY_PATH", "./hermes_memory"))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
FACT_DEDUP_THRESHOLD = float(os.environ.get("FACT_DEDUP_THRESHOLD", "0.88"))
MAX_CONVERSATION_TURNS = int(os.environ.get("MAX_CONVERSATION_TURNS", "20"))

# --- Agent loop ---
# Hard ceiling on tool-call rounds within a single turn - the previous loop
# had no limit at all, so a model stuck re-calling the same tool (or
# chaining searches indefinitely) could hang a turn forever.
MAX_AGENT_ITERATIONS = int(os.environ.get("MAX_AGENT_ITERATIONS", "8"))

# --- Safety ---
REQUIRE_WRITE_CONFIRMATION = os.environ.get("REQUIRE_WRITE_CONFIRMATION", "true").lower() == "true"

# --- Observability ---
LOG_TURNS = os.environ.get("LOG_TURNS", "true").lower() == "true"
LOG_FILE = Path(os.environ.get("LOG_FILE", "./hermes_turns.jsonl"))

# --- Git ---
GIT_AUTO_PULL_ON_START = os.environ.get("GIT_AUTO_PULL_ON_START", "true").lower() == "true"
GIT_REMOTE = os.environ.get("GIT_REMOTE", "origin")
GIT_BRANCH = os.environ.get("GIT_BRANCH", "main")

# --- Misc ---
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"