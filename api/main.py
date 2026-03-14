from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging

from core.database import init_database
from services.ingestion import ingest_zip, clear_all_data
from services.query import query_documents, execute_code, get_stats
from services.cache import clear_cache
from core.config import logger

# Initialize database on startup
init_database()

app = FastAPI(title="Scientific RAG API")


# Models
class IngestResponse(BaseModel):
    status: str
    files_processed: int
    chunks_created: int
    message: str


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_cache: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    retrieved_context: List[dict]
    latency_breakdown: Dict[str, float]
    verification: Dict
    is_hallucinated: bool
    cached: bool
    retrieval_strategy: Dict = {}


class CodeExecutionRequest(BaseModel):
    code: str


class CodeExecutionResponse(BaseModel):
    success: bool
    output: str
    error: str
    execution_time: int


# Endpoints
@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...), clear_first: bool = Form(True)):
    """Upload a ZIP file containing documents to ingest."""
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files are accepted")

    content = await file.read()
    result = await ingest_zip(content, file.filename, clear_first=clear_first)
    return result


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Query the indexed documents and structured data."""
    return query_documents(req.query, req.top_k, req.use_cache)


@app.post("/execute")
async def execute(req: CodeExecutionRequest):
    """Execute Python code in sandbox."""
    result = execute_code(req.code)

    # Return error status code if execution failed
    if not result["success"]:
        return {
            "success": False,
            "output": result["output"],
            "error": result["error"],
            "execution_time": result["execution_time"],
            "error_type": "validation_error"
            if "Syntax" in result["error"]
            else "execution_error",
        }

    return result


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    return get_stats()


@app.delete("/clear")
def clear():
    """Clear all indexed data."""
    return clear_all_data()


@app.delete("/cache")
def clear_cache_endpoint():
    """Clear semantic cache."""
    return clear_cache()


# Evaluation endpoint
class EvalRequest(BaseModel):
    num_queries: int = 10
    sample_queries: Optional[List[str]] = None


@app.post("/eval")
def run_evaluation(req: EvalRequest):
    """
    Run evaluation on the RAG pipeline.

    Either provide custom sample_queries, or let the system
    generate queries dynamically from indexed data.
    """
    import time
    import requests
    from datetime import datetime

    logger.info(f"Starting evaluation with {req.num_queries} queries")

    # Get stats
    stats = get_stats()

    # Get sample documents to generate queries from
    documents = []
    if not req.sample_queries:
        # Fetch sample content from indexed data
        sample_terms = ["data", "list", "show", "all", "info"]
        for term in sample_terms:
            try:
                result = query_documents(term, top_k=20, use_cache=False)
                docs = result.get("retrieved_context", [])
                if docs:
                    documents.extend(docs)
                    break
            except Exception as e:
                logger.warning(f"Error fetching samples: {e}")
                continue

    # Generate or use provided queries
    if req.sample_queries:
        queries = req.sample_queries[: req.num_queries]
    else:
        queries = _generate_eval_queries(documents, req.num_queries)

    # Run evaluation
    results = []
    successful = 0
    failed = 0

    for i, query in enumerate(queries):
        logger.info(f"[{i + 1}/{len(queries)}] Evaluating: {query[:50]}...")
        start_time = time.time()

        try:
            result = query_documents(query, top_k=5, use_cache=False)
            total_time = time.time() - start_time

            results.append(
                {
                    "query": query,
                    "answer": result.get("answer", "")[:200],
                    "is_cached": result.get("cached", False),
                    "total_time": total_time,
                    "latency_breakdown": result.get("latency_breakdown", {}),
                    "sources_count": len(result.get("sources", [])),
                    "is_hallucinated": result.get("is_hallucinated", False),
                }
            )
            successful += 1

        except Exception as e:
            results.append(
                {
                    "query": query,
                    "error": str(e),
                    "total_time": time.time() - start_time,
                }
            )
            failed += 1
            logger.error(f"Error evaluating query: {e}")

    # Calculate summary
    if successful > 0:
        avg_time = (
            sum(r["total_time"] for r in results if "error" not in r) / successful
        )
        cached_count = sum(1 for r in results if r.get("is_cached"))
        avg_sources = (
            sum(r["sources_count"] for r in results if "error" not in r) / successful
        )
        hallucinated_count = sum(1 for r in results if r.get("is_hallucinated"))
    else:
        avg_time = cached_count = avg_sources = hallucinated_count = 0

    eval_report = {
        "timestamp": datetime.now().isoformat(),
        "num_queries": len(queries),
        "successful": successful,
        "failed": failed,
        "stats": stats,
        "summary": {
            "average_latency": round(avg_time, 2),
            "cache_hit_rate": f"{cached_count}/{successful} ({cached_count / successful * 100:.1f}%)"
            if successful > 0
            else "0%",
            "average_sources": round(avg_sources, 1),
            "hallucination_rate": f"{hallucinated_count}/{successful} ({hallucinated_count / successful * 100:.1f}%)"
            if successful > 0
            else "0%",
        },
        "results": results,
    }

    logger.info(f"Evaluation complete: {successful} successful, {failed} failed")
    return eval_report


def _generate_eval_queries(documents: List[Dict], num_queries: int) -> List[str]:
    """Generate generic evaluation queries from documents."""
    import re
    import random

    templates = {
        "factual": [
            "What can you tell me about {subject}?",
            "Tell me about {subject}",
            "What is {subject}?",
        ],
        "comparative": [
            "Compare {a} and {b}",
            "What is the difference between {a} and {b}?",
        ],
        "numeric": [
            "What are the numerical values for {subject}?",
            "Show me the data for {subject}",
        ],
        "count": [
            "How many {subject} are there?",
            "Count the {subject}",
        ],
        "list": [
            "List all {subject}",
            "What are all the {subject}?",
        ],
    }

    # Extract entities from documents
    entities = []
    for doc in documents:
        content = doc.get("content", "")
        words = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", content)
        entities.extend(words[:3])

    entities = list(set(entities))[:30]

    if not entities:
        return ["What data is available?", "Tell me about the content"]

    queries = []
    for _ in range(num_queries):
        template_type = random.choice(list(templates.keys()))
        template = random.choice(templates[template_type])

        if "{subject}" in template:
            subject = random.choice(entities)
            queries.append(template.format(subject=subject))
        elif "{a}" in template and "{b}" in template:
            a = random.choice(entities)
            b = random.choice([e for e in entities if e != a][:1] or entities)
            queries.append(template.format(a=a, b=b))
        else:
            subject = random.choice(entities)
            queries.append(template.format(subject=subject))

    return queries[:num_queries]
