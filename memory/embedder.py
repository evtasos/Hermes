"""
Single embedder singleton, shared by everything that needs one
(conversation recall, fact storage, fact dedup). Previously duplicated:
this module and memory/store.py each kept their own SentenceTransformer
instance, so a changed EMBEDDING_MODEL would silently leave conversation
memory and fact dedup comparing vectors from two different models, and
the model was loaded into memory twice for nothing.
"""
from typing import Optional

from sentence_transformers import SentenceTransformer
import structlog

from config import EMBEDDING_MODEL

logger = structlog.get_logger()

_embedder: Optional[SentenceTransformer] = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("loading_embedder", model=EMBEDDING_MODEL)
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder
