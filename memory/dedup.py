"""
Fact deduplication using exact normalization and embedding similarity.
"""
import re
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import structlog

logger = structlog.get_logger()


def normalize_text(text: str) -> str:
    """Lowercase, replace user/owner pronouns, strip whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\buser\b", "owner", text)
    text = re.sub(r"\bi\s", "owner ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def is_duplicate_fact(
    new_fact: str,
    existing_facts: List[str],
    embedder: SentenceTransformer,
    exact_match: bool = True,
    similarity_threshold: float = 0.88,
) -> bool:
    """
    Return True if new_fact is considered a duplicate of any in existing_facts.
    Uses exact normalized match first, then embedding cosine similarity.
    """
    if not existing_facts:
        return False

    norm_new = normalize_text(new_fact)

    # 1. Exact normalized match
    if exact_match:
        for ex in existing_facts:
            if normalize_text(ex) == norm_new:
                logger.debug(
                    "fact_dup_exact",
                    new=new_fact,
                    matched=ex,
                )
                return True

    # 2. Embedding similarity (only if not exact match)
    try:
        new_emb = embedder.encode([new_fact])
        existing_embs = embedder.encode(existing_facts)
        sims = cosine_similarity(new_emb, existing_embs)[0]
        max_sim = float(np.max(sims)) if len(sims) > 0 else 0.0
        if max_sim >= similarity_threshold:
            idx = int(np.argmax(sims))
            logger.debug(
                "fact_dup_embedding",
                new=new_fact,
                matched=existing_facts[idx],
                similarity=max_sim,
            )
            return True
    except Exception as e:
        logger.warning("fact_embedding_error", error=str(e))

    return False