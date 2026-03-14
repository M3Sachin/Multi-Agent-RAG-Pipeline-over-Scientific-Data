"""
Evaluation Script for Scientific RAG Pipeline

This script dynamically generates evaluation queries based on the actual
indexed data and validates the pipeline's performance.
"""

import argparse
import logging
import json
import random
import re
import requests
import time
from datetime import datetime
from typing import Any, Optional, List, Dict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000"
DEFAULT_NUM_QUERIES = 10
TIMEOUT_SECONDS = 180

GENERIC_QUERY_TEMPLATES = {
    "factual": [
        "What can you tell me about {subject}?",
        "Tell me about {subject}",
        "What is {subject}?",
        "Explain {subject}",
        "What do you know about {subject}?",
    ],
    "comparative": [
        "Compare {subject_a} and {subject_b}",
        "What is the difference between {subject_a} and {subject_b}?",
        "{subject_a} vs {subject_b}",
    ],
    "numeric": [
        "What are the numerical values for {subject}?",
        "Show me the data for {subject}",
        "What statistics are available for {subject}?",
    ],
    "relationship": [
        "How is {subject_a} related to {subject_b}?",
        "What is the relationship between {subject_a} and {subject_b}?",
    ],
    "count": [
        "How many {subject} are there?",
        "Count the {subject}",
        "What is the total number of {subject}?",
    ],
    "list": [
        "List all {subject}",
        "What are all the {subject}?",
        "Show me {subject}",
    ],
}


def get_api_health() -> Optional[dict]:
    """Check if API is healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def get_stats() -> dict:
    """Get database and cache stats."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        logger.warning(f"Could not get stats: {e}")
        return {}


def get_sample_documents(num_samples: int = 20) -> List[Dict]:
    """Get sample documents by querying for common terms."""
    sample_queries = ["data", "list", "show", "all", "info"]

    for term in sample_queries:
        try:
            response = requests.post(
                f"{API_BASE_URL}/query",
                json={"query": term, "top_k": num_samples, "use_cache": False},
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                context = data.get("retrieved_context", [])
                if context:
                    return context
        except Exception:
            continue

    return []


def get_table_info() -> List[Dict]:
    """Get information about indexed tables."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        if response.status_code == 200:
            data = response.json()
            tables = data.get("structured_tables", [])
            return [{"table": t, "columns": []} for t in tables]
        return []
    except Exception:
        return []


def extract_entities_from_content(documents: List[Dict]) -> List[str]:
    """Extract potential entity names from document content."""
    entities = set()

    for doc in documents:
        content = doc.get("content", "")

        words = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", content)
        entities.update(words[:5])

        lines = content.split("\n")[:3]
        for line in lines:
            if line.strip() and len(line) < 100:
                entities.add(line.strip())

    return list(entities)[:50]


def extract_table_names_from_content(documents: List[Dict]) -> List[str]:
    """Extract potential table/column names from content."""
    names = set()

    for doc in documents:
        content = doc.get("content", "")

        table_matches = re.findall(r'"([a-z_]+)"', content)
        names.update(table_matches)

        key_matches = re.findall(r"\b([a-z_]+)\b", content)
        names.update([m for m in key_matches if len(m) > 3])

    return list(names)[:30]


def generate_queries_from_content(documents: List[Dict], num_queries: int) -> List[str]:
    """Generate queries dynamically based on actual indexed content."""
    queries = []

    entities = extract_entities_from_content(documents)
    table_names = extract_table_names_from_content(documents)

    if not entities and not table_names:
        queries.extend(
            [
                "What data is available?",
                "Tell me about the indexed content",
                "What can you help me with?",
            ]
        )
        return queries

    for _ in range(num_queries):
        template_type = random.choice(list(GENERIC_QUERY_TEMPLATES.keys()))
        template = random.choice(GENERIC_QUERY_TEMPLATES[template_type])

        if "{subject}" in template:
            subject = random.choice(entities) if entities else "the data"
            query = template.format(subject=subject)
        elif "{subject_a}" in template and "{subject_b}" in template:
            subject_a = random.choice(entities) if entities else "data"
            subject_b = (
                random.choice(entities[1:]) if len(entities) > 1 else "information"
            )
            query = template.format(subject_a=subject_a, subject_b=subject_b)
        else:
            subject = random.choice(entities) if entities else "the data"
            query = template.format(subject=subject)

        if query not in queries:
            queries.append(query)

    return queries[:num_queries]


def run_query(query: str) -> Dict[str, Any]:
    """Run a single query and return the result."""
    start_time = time.time()
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"query": query, "top_k": 5},
            timeout=TIMEOUT_SECONDS,
        )
        total_time = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            return {
                "query": query,
                "answer": data.get("answer", "")[:200],
                "is_cached": data.get("cached", False),
                "total_time": total_time,
                "latency_breakdown": data.get("latency_breakdown", {}),
                "sources_count": len(data.get("sources", [])),
                "has_verification": "verification" in data,
                "is_hallucinated": data.get("is_hallucinated", False),
            }
        else:
            return {
                "query": query,
                "error": f"HTTP {response.status_code}",
                "total_time": total_time,
            }
    except Exception as e:
        return {
            "query": query,
            "error": str(e),
            "total_time": time.time() - start_time,
        }


def evaluate_pipeline(
    num_queries: Optional[int] = None,
    output_file: Optional[str] = None,
    api_url: Optional[str] = None,
):
    """Run evaluation on the RAG pipeline."""
    global API_BASE_URL
    if api_url:
        API_BASE_URL = api_url

    num_queries = num_queries or DEFAULT_NUM_QUERIES

    logger.info("=" * 60)
    logger.info("Scientific RAG Pipeline Evaluation")
    logger.info("=" * 60)

    health = get_api_health()
    if not health:
        logger.error("API is not healthy or not running")
        return
    logger.info(f"API Status: {health}")

    stats = get_stats()
    logger.info(f"Database Stats: {stats}")

    documents = get_sample_documents()
    if not documents:
        logger.warning("No documents indexed. Please ingest data first.")
        queries = [
            "What data is available?",
            "Tell me about the content",
        ]
    else:
        logger.info(f"Retrieved {len(documents)} sample documents for query generation")
        queries = generate_queries_from_content(documents, num_queries)

    logger.info("-" * 60)
    logger.info(f"Running {len(queries)} evaluation queries...")
    logger.info("-" * 60)

    results = []
    for i, query in enumerate(queries, 1):
        logger.info(f"[{i}/{len(queries)}] Query: {query[:60]}...")
        result = run_query(query)

        if "error" not in result:
            logger.info(
                f"  Time: {result['total_time']:.2f}s | "
                f"Cached: {result['is_cached']} | "
                f"Sources: {result['sources_count']} | "
                f"Hallucinated: {result['is_hallucinated']}"
            )
        else:
            logger.error(f"  Error: {result['error']}")

        results.append(result)

    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    logger.info(f"Total Queries: {len(results)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")

    if successful:
        avg_time = sum(r["total_time"] for r in successful) / len(successful)
        cached_count = sum(1 for r in successful if r.get("is_cached"))
        avg_sources = sum(r["sources_count"] for r in successful) / len(successful)
        hallucinated_count = sum(1 for r in successful if r.get("is_hallucinated"))

        logger.info(f"Average Latency: {avg_time:.2f}s")
        logger.info(
            f"Cache Hit Rate: {cached_count}/{len(successful)} ({cached_count / len(successful) * 100:.1f}%)"
        )
        logger.info(f"Average Sources: {avg_sources:.1f}")
        logger.info(
            f"Hallucination Rate: {hallucinated_count}/{len(successful)} ({hallucinated_count / len(successful) * 100:.1f}%)"
        )

        all_latencies = {}
        for r in successful:
            lb = r.get("latency_breakdown", {})
            for key, value in lb.items():
                if key not in all_latencies:
                    all_latencies[key] = []
                all_latencies[key].append(value)

        if all_latencies:
            logger.info("Latency Breakdown (avg):")
            for key, values in all_latencies.items():
                avg_val = sum(values) / len(values)
                logger.info(f"  {key}: {avg_val:.3f}s")

    if failed:
        logger.warning("Failed Queries:")
        for r in failed:
            logger.warning(f"  - {r['query'][:50]}... : {r.get('error', 'Unknown')}")

    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"eval_results_{timestamp}.json"

    eval_report = {
        "timestamp": datetime.now().isoformat(),
        "api_url": API_BASE_URL,
        "num_queries": len(queries),
        "total_queries": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "stats": stats,
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(eval_report, f, indent=2, default=str)

    logger.info(f"Results saved to: {output_file}")
    logger.info("=" * 60)

    return eval_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Scientific RAG Pipeline")
    parser.add_argument(
        "-n",
        "--num-queries",
        type=int,
        default=DEFAULT_NUM_QUERIES,
        help=f"Number of queries to run (default: {DEFAULT_NUM_QUERIES})",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Output file path for results"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=API_BASE_URL,
        help=f"API base URL (default: {API_BASE_URL})",
    )

    args = parser.parse_args()

    evaluate_pipeline(
        num_queries=args.num_queries, output_file=args.output, api_url=args.url
    )


if __name__ == "__main__":
    main()
