import os
import json
from openai import OpenAI

from core.config import REASONING_MODEL, logger
from core.database import get_engine
from sqlalchemy import text

client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"), api_key="ollama"
)

RETRIEVAL_PROMPT = """You are a Retrieval Planning Agent. Analyze the user query and determine the best strategy to find the answer.

You have access to:
1. document_chunks - Semantic vector search for PDFs, DOCX, TXT files
2. Structured tables - SQL tables from CSV/Excel data

Analyze the query and output a JSON plan:

{
    "search_strategy": "documents" | "tables" | "both",
    "reason": "why this strategy",
    "sql_hints": "any specific columns or tables to query (optional)",
    "keywords": ["important", "keywords", "from", "query"]
}

Guidelines:
- If the query asks for specific values, numbers, or data points → search tables
- If the query asks for explanations, summaries, or textual content → search documents
- If unclear → search both
- Look for keywords indicating table vs document search (e.g., "list", "show", "count" → tables; "explain", "describe", "what does it say" → documents)
"""


def analyze_query(user_query: str) -> dict:
    """Analyze query and determine best search strategy."""

    try:
        response = client.chat.completions.create(
            model=REASONING_MODEL,
            messages=[
                {"role": "system", "content": RETRIEVAL_PROMPT},
                {"role": "user", "content": user_query},
            ],
            temperature=0.0,
            max_tokens=500,
        )

        result = response.choices[0].message.content

        if result and result.strip():
            # Try to parse JSON
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                result = result.split("```")[1].split("```")[0]

            parsed = json.loads(result.strip())
            if parsed:
                return parsed

    except Exception as e:
        logger.error(f"Retrieval analysis error: {e}")

    # Default fallback
    return {
        "search_strategy": "both",
        "reason": "Default: search both documents and tables",
        "sql_hints": "",
        "keywords": [],
    }


def get_table_info() -> dict:
    """Get information about available tables."""
    engine = get_engine()
    tables_info = {}

    with engine.connect() as conn:
        # Get all tables
        result = conn.execute(
            text("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name != 'document_chunks'
        """)
        )

        for row in result.fetchall():
            table_name = row[0]

            # Get columns for each table
            cols_result = conn.execute(
                text(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
            """)
            )

            tables_info[table_name] = {
                "columns": [c[0] for c in cols_result.fetchall()],
                "row_count": 0,
            }

            # Get row count
            try:
                count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                tables_info[table_name]["row_count"] = count_result.fetchone()[0]
            except:
                pass

    return tables_info


def generate_sql_query(user_query: str, table_info: dict) -> str:
    """Generate SQL query based on user query and table structure."""

    tables_desc = "\n".join(
        [
            f"Table '{name}': columns {info['columns']}"
            for name, info in table_info.items()
        ]
    )

    prompt = f"""Based on the user question and available tables, generate a SQL query.

Available tables:
{tables_desc}

User question: {user_query}

Generate a SQL SELECT query to answer this question. Just output the SQL query, nothing else."""

    try:
        response = client.chat.completions.create(
            model=REASONING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )

        sql = response.choices[0].message.content.strip()

        # Clean up SQL
        if "```sql" in sql:
            sql = sql.split("```sql")[1].split("```")[0]
        elif "```" in sql:
            sql = sql.split("```")[1].split("```")[0]

        return sql.strip()

    except Exception as e:
        logger.error(f"SQL generation error: {e}")
        return None
