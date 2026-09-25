"""
Hermes configuration.
Loads environment variables and provides defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Home Assistant MCP ---
HA_MCP_URL = os.environ.get("HA_MCP_URL", "http://homeassistant.local:9584")
HA_TOKEN = os.environ.get("HA_TOKEN", "")

# --- LLM Router ---
# Primary: Gemini (free tier). Fallback: Ollama (local), then OpenRouter (optional)
# Gemini model list: tried in order; first that works is used.
GEMINI_MODEL_FALLBACKS = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.0-pro",
]

# Ollama host and model
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://192.168.1.90:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:mini")  # change if you prefer another

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