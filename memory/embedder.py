"""
Embedder singleton accessor.
"""
from sentence_transformers import SentenceTransformer

_embedder: "SentenceTransformer | None" = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder