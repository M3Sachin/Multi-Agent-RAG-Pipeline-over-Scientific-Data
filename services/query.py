import os
import json
import time
from openai import OpenAI
from sqlalchemy import text

from core.config import EMBEDDING_MODEL, REASONING_MODEL, logger
from core.database import get_engine
from services.cache import check_cache, add_to_cache, get_cache_stats
from services.verifier import verify_answer
from services.python_executor import execute_python
from services.retrieval_agent import analyze_query, get_table_info, generate_sql_query
from services.entity_traversal import extract_and_traverse

client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"), api_key="ollama"
)


def query_documents(user_query: str, top_k: int = 5, use_cache: bool = True) -> dict:
    """
    Query indexed documents and structured data with intelligent retrieval.

    Args:
        user_query: The question to ask
        top_k: Number of most similar document chunks to retrieve
        use_cache: Whether to check/use semantic cache

    Returns:
        dict with answer, sources, retrieved_context, latency_breakdown, verification
    """
    start_total = time.time()
    latencies = {}

    # Check cache first
    if use_cache:
        cached_result = check_cache(user_query)
        if cached_result:
            cached_result["latency_breakdown"]["total_request_time"] = (
                time.time() - start_total
            )
            return cached_result

    engine = get_engine()

    # 1. Analyze query to determine search strategy
    start_analyze = time.time()
    retrieval_plan = analyze_query(user_query)
    latencies["query_analysis"] = round(time.time() - start_analyze, 3)

    # 2. Generate embedding for query
    start_embed = time.time()
    resp = client.embeddings.create(input=user_query, model=EMBEDDING_MODEL)
    query_embedding = resp.data[0].embedding
    vector_str = "[" + ",".join(map(str, query_embedding)) + "]"
    latencies["embedding_generation"] = round(time.time() - start_embed, 3)

    all_context = []
    all_sources = []

    # 3. Search based on strategy
    start_search = time.time()

    with engine.connect() as conn:
        # Get table info for entity traversal
        table_info = get_table_info()

        # Entity traversal - find related entities
        entity_data = extract_and_traverse(user_query)
        if entity_data.get("relationship_data"):
            for rel in entity_data["relationship_data"]:
                for rel_data in rel.get("direct_relationships", []):
                    all_context.append(
                        {
                            "source": f"[RELATIONSHIP] {rel['start_entity']}",
                            "content": json.dumps(rel_data),
                            "type": "relationship",
                            "similarity": 1.0,
                        }
                    )
                    all_sources.append(f"Relationship: {rel['start_entity']}")

        # Search documents if strategy includes documents
        if retrieval_plan.get("search_strategy") in ["documents", "both"]:
            doc_result = conn.execute(
                text(f"""
                SELECT source_file, content, (embedding <=> '{vector_str}'::vector) as distance
                FROM document_chunks ORDER BY distance ASC LIMIT {top_k}
            """)
            )

            for row in doc_result.fetchall():
                sim = 1 - row.distance if row.distance else 0
                all_context.append(
                    {
                        "source": f"[DOC] {row.source_file}",
                        "content": row.content,
                        "type": "document",
                        "similarity": round(sim, 3),
                    }
                )
                all_sources.append(row.source_file)

        # Search structured tables if strategy includes tables
        if retrieval_plan.get("search_strategy") in ["tables", "both"]:
            # Try to generate SQL query
            if table_info:
                sql_query = generate_sql_query(user_query, table_info)
                if sql_query:
                    try:
                        # Quote table/column names to handle case sensitivity
                        quoted_sql = sql_query
                        # Try to quote identifiers
                        for tbl in table_info.keys():
                            # Replace unquoted table names with quoted ones
                            import re

                            # Match table name not already in quotes
                            pattern = r"\b" + tbl + r"\b"
                            quoted_sql = re.sub(
                                pattern, f'"{tbl}"', quoted_sql, flags=re.IGNORECASE
                            )

                        # Execute generated SQL
                        result = conn.execute(text(quoted_sql))
                        rows = result.fetchall()
                        if rows:
                            keys = result.keys()
                            all_context.append(
                                {
                                    "source": "[SQL QUERY]",
                                    "content": json.dumps(
                                        [dict(zip(keys, row)) for row in rows]
                                    ),
                                    "type": "sql_result",
                                    "similarity": 1.0,
                                }
                            )
                            all_sources.append("Generated SQL Query")
                    except Exception as e:
                        logger.error(f"SQL execution error: {e}")
                        conn.rollback()  # Rollback on error

            # Also search all tables
            try:
                tables_result = conn.execute(
                    text(
                        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name != 'document_chunks'"
                    )
                )
                for r in tables_result.fetchall():
                    table = r[0]
                    try:
                        cols_result = conn.execute(
                            text(
                                f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}'"
                            )
                        )
                        columns = [c[0] for c in cols_result.fetchall()]
                        if columns:
                            rows_result = conn.execute(
                                text(f'SELECT * FROM "{table}" LIMIT 5')
                            )
                            rows = rows_result.fetchall()
                            if rows:
                                all_context.append(
                                    {
                                        "source": f"[TABLE] {table}",
                                        "content": json.dumps(
                                            {
                                                "table": table,
                                                "columns": columns,
                                                "rows": [
                                                    dict(zip(columns, row))
                                                    for row in rows
                                                ],
                                            }
                                        ),
                                        "type": "structured",
                                        "similarity": 1.0,
                                    }
                                )
                                all_sources.append(f"Table: {table}")
                    except Exception as e:
                        logger.error(f"Error querying table {table}: {e}")
            except Exception as e:
                logger.error(f"Error fetching tables: {e}")

    latencies["retrieval"] = round(time.time() - start_search, 3)

    # Sort by similarity
    all_context.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    context_text = "\n\n".join(
        [f"Source: {c['source']}\n{c['content'][:800]}" for c in all_context[:5]]
    )

    # 4. Get LLM answer
    start_llm = time.time()

    # Add retrieval strategy info to prompt
    strategy_info = f"\n\nRetrieval Strategy: {retrieval_plan.get('search_strategy', 'both')} - {retrieval_plan.get('reason', '')}"

    prompt = f"""Based on the following context, answer the question.

Context:
{context_text}
{strategy_info}

Question: {user_query}

Answer:"""

    llm_response = client.chat.completions.create(
        model=REASONING_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    answer = (
        llm_response.choices[0].message.content
        or "I couldn't find relevant information."
    )
    latencies["llm_generation"] = round(time.time() - start_llm, 3)

    # 5. Verify answer
    start_verify = time.time()
    verification = verify_answer(user_query, answer, all_context)
    latencies["verification"] = round(time.time() - start_verify, 3)

    latencies["total_request_time"] = round(time.time() - start_total, 3)

    result = {
        "answer": answer,
        "sources": list(set(all_sources)),
        "retrieved_context": all_context,
        "latency_breakdown": latencies,
        "verification": verification,
        "is_hallucinated": verification.get("hallucination_detected", False),
        "cached": False,
        "retrieval_strategy": retrieval_plan,
    }

    # Add to cache
    if use_cache:
        add_to_cache(user_query, query_embedding, result)

    return result


def execute_code(code: str) -> dict:
    """Execute Python code in sandbox."""
    return execute_python(code)


def get_stats() -> dict:
    """Get ingestion and cache stats."""
    engine = get_engine()
    with engine.connect() as conn:
        doc_count_result = conn.execute(
            text("SELECT COUNT(*) FROM document_chunks")
        ).fetchone()
        doc_count = doc_count_result[0] if doc_count_result else 0
        tables_result = conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name != 'document_chunks'"
            )
        )
        tables = [r[0] for r in tables_result.fetchall()]

    cache_stats = get_cache_stats()

    return {
        "document_chunks": doc_count,
        "structured_tables": tables,
        "cache": cache_stats,
    }
