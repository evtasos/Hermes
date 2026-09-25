"""
ChromaDB wrapper for conversations and facts.
"""
import uuid
from datetime import datetime
from typing import List, Optional

import chromadb
from sentence_transformers import SentenceTransformer
import structlog

from config import MEMORY_PATH, EMBEDDING_MODEL, MAX_CONVERSATION_TURNS

logger = structlog.get_logger()

# Embedder singleton (loaded once)
_embedder: Optional[SentenceTransformer] = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("loading_embedder", model=EMBEDDING_MODEL)
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(MEMORY_PATH))


def get_or_create_collection(name: str):
    client = get_chroma_client()
    return client.get_or_create_collection(name)


def remember(text: str, role: str) -> None:
    """Store a conversation turn."""
    if not text:
        return
    embedder = get_embedder()
    embedding = embedder.encode([text])[0].tolist()
    col = get_or_create_collection("conversations")
    col.add(
        ids=[f"{datetime.now().isoformat()}-{role}-{uuid.uuid4()}"],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{"role": role, "ts": datetime.now().isoformat()}],
    )


def recall(query: str, k: int = 3) -> List[str]:
    """Retrieve top-k similar conversation turns."""
    if not query.strip():
        return []
    embedder = get_embedder()
    embedding = embedder.encode([query])[0].tolist()
    col = get_or_create_collection("conversations")
    results = col.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["documents"],
    )
    docs = results.get("documents", [[]])[0]
    return [doc for doc in docs if doc]


def store_fact(text: str) -> None:
    """Store a fact (deduplication handled elsewhere)."""
    if not text:
        return
    embedder = get_embedder()
    embedding = embedder.encode([text])[0].tolist()
    col = get_or_create_collection("facts")
    col.add(
        ids=[f"fact-{datetime.now().isoformat()}-{uuid.uuid4()}"],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{"ts": datetime.now().isoformat()}],
    )


def get_all_facts() -> List[str]:
    col = get_or_create_collection("facts")
    results = col.get(include=["documents"])
    return results.get("documents", [])


def cap_conversations(max_turns: int = MAX_CONVERSATION_TURNS) -> None:
    """Keep only the most recent N turns in conversations collection."""
    col = get_or_create_collection("conversations")
    all_items = col.get(include=["metadatas", "documents", "ids"])
    ids = all_items.get("ids", [])
    metadatas = all_items.get("metadatas", [])
    sorted_pairs = sorted(
        zip(ids, metadatas),
        key=lambda pair: pair[1].get("ts", ""),
        reverse=True,
    )
    keep_ids = [pid for pid, _ in sorted_pairs[:max_turns]]
    delete_ids = [pid for pid, _ in sorted_pairs[max_turns:]]
    if delete_ids:
        col.delete(ids=delete_ids)
        logger.info("conversations_capped", kept=len(keep_ids), removed=len(delete_ids))