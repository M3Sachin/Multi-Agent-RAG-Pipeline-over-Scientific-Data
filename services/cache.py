import os
import json
import time
import hashlib
from typing import Optional
from openai import OpenAI
from sqlalchemy import text

from core.config import EMBEDDING_MODEL, REASONING_MODEL, logger
from core.database import get_engine

client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"), api_key="ollama"
)

# Cache settings
CACHE_SIMILARITY_THRESHOLD = 0.92
CACHE_TTL_SECONDS = 3600  # 1 hour

# In-memory semantic cache
_semantic_cache = []
_cache_hits = 0
_cache_misses = 0


def _get_query_hash(query: str) -> str:
    """Generate hash for query."""
    return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]


def _get_embedding(text: str):
    """Get embedding for a text."""
    resp = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return resp.data[0].embedding


def _cosine_similarity(a: list, b: list) -> float:
    """Calculate cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)


def check_cache(user_query: str) -> Optional[dict]:
    """
    Check if query exists in cache with high similarity.
    Returns cached result if found, None otherwise.
    """
    global _semantic_cache, _cache_hits, _cache_misses

    if not _semantic_cache:
        _cache_misses += 1
        return None

    try:
        # Get embedding for query
        query_embedding = _get_embedding(user_query)

        # Check against cached queries
        for cached in _semantic_cache:
            # Skip expired entries
            if time.time() - cached["timestamp"] > CACHE_TTL_SECONDS:
                continue

            # Calculate similarity
            similarity = _cosine_similarity(query_embedding, cached["embedding"])

            if similarity >= CACHE_SIMILARITY_THRESHOLD:
                _cache_hits += 1
                logger.info(f"Cache hit! Similarity: {similarity:.3f}")
                return {
                    "answer": cached["answer"],
                    "sources": cached["sources"],
                    "retrieved_context": cached["retrieved_context"],
                    "latency_breakdown": cached["latency_breakdown"],
                    "verification": cached.get("verification", {}),
                    "is_hallucinated": cached.get("is_hallucinated", False),
                    "cached": True,
                }
    except Exception as e:
        logger.error(f"Cache check error: {e}")

    _cache_misses += 1
    return None


def add_to_cache(user_query: str, query_embedding: list, result: dict):
    """Add query result to cache."""
    global _semantic_cache

    # Keep only last 100 entries
    if len(_semantic_cache) >= 100:
        _semantic_cache = _semantic_cache[-99:]

    _semantic_cache.append(
        {
            "query": user_query.lower().strip(),
            "embedding": query_embedding,
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved_context": result["retrieved_context"],
            "latency_breakdown": result["latency_breakdown"],
            "timestamp": time.time(),
        }
    )


def get_cache_stats() -> dict:
    """Get cache statistics."""
    global _cache_hits, _cache_misses
    total = _cache_hits + _cache_misses
    hit_rate = (_cache_hits / total * 100) if total > 0 else 0

    return {
        "cache_size": len(_semantic_cache),
        "cache_hits": _cache_hits,
        "cache_misses": _cache_misses,
        "hit_rate_percent": round(hit_rate, 2),
    }


def clear_cache():
    """Clear the semantic cache."""
    global _semantic_cache, _cache_hits, _cache_misses
    _semantic_cache = []
    _cache_hits = 0
    _cache_misses = 0
    return {"status": "success", "message": "Cache cleared"}
