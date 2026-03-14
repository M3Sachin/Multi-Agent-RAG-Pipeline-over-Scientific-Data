# 🔬 Scientific RAG Pipeline
> **A robust AI pipeline for answering complex scientific questions by orchestrating semantic search over documents and SQL queries across structured experimental datasets.**

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql)](https://www.postgresql.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_AI-white?style=for-the-badge)](https://ollama.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)

---

## Quick Start

### 1. Setup Ollama (LLM + Embeddings)

This project uses open-source models via Ollama:

```bash
# Install Ollama from https://ollama.com

# Pull the LLM model (llama3.1)
ollama pull llama3.1

# Pull the embedding model
ollama pull nomic-embed-text

# Verify models are installed
ollama list
```

### 2. Start Database

```bash
docker compose up -d
```

### 3. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment

Create a `.env` file (or copy `.env_example`):

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/sci_rag

# Ollama (LLM + Embeddings)
OLLAMA_BASE_URL=http://localhost:11434/v1
REASONING_MODEL=llama3.1
VERIFIER_MODEL=llama3.1
EMBEDDING_MODEL=nomic-embed-text

# UI
API_BASE_URL=http://localhost:8000
```

### 6. Run the Pipeline

```bash
# Start both API and UI
python run.py

# Or start individually
python run.py api    # Starts FastAPI on port 8000
python run.py ui     # Starts Streamlit on port 8501
```

---

## Project Structure

```
sci-rag-agent/
├── api/
│   └── main.py              # FastAPI endpoints
├── core/
│   ├── config.py            # Configuration
│   └── database.py          # Database setup
├── services/
│   ├── cache.py             # Semantic cache
│   ├── entity_traversal.py  # Knowledge graph traversal
│   ├── ingestion.py         # ZIP ingestion + processing
│   ├── python_executor.py   # Safe code execution
│   ├── query.py             # Query + retrieval + verification
│   ├── retrieval_agent.py   # Query analysis + SQL generation
│   └── verifier.py          # Fact-checker agent
├── ui/
│   └── app.py               # Streamlit UI
├── utils/
│   ├── text_extraction.py   # PDF/DOCX/XLSX extraction
│   └── text_chunking.py     # Text chunking
├── scripts/
│   └── eval.py              # Evaluation script
├── run.py                   # Main runner script (API + UI)
├── start_api.bat            # Windows API starter (local use, not in git)
├── start_ui.bat             # Windows UI starter (local use, not in git)
├── .env                     # Environment variables
├── .env_example             # Example environment variables
├── requirements.txt         # Dependencies
└── docker-compose.yml       # PostgreSQL + pgvector
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Upload ZIP file with documents |
| `/query` | POST | Query indexed documents |
| `/execute` | POST | Execute Python code safely |
| `/stats` | GET | Get DB + cache stats |
| `/clear` | DELETE | Clear all indexed data |
| `/cache` | DELETE | Clear semantic cache |
| `/eval` | POST | Run evaluation on pipeline |
| `/health` | GET | Health check |

---

## Streamlit UI

The project includes a user-friendly Streamlit web interface for interacting with the pipeline.

### UI Features
- **Ingest Tab**: Upload ZIP files containing documents and structured data
- **Query Tab**: Ask questions and view answers with sources and verification
- **Evaluate Tab**: Run systematic evaluations on the pipeline
- **Execute Tab**: Run Python code in a safe sandbox environment
- **Delete Data Tab**: Manage and clear indexed data
- **Help Tab**: Step-by-step guide and usage instructions

### Accessing the UI
- **URL**: http://localhost:8501
- **API Status Indicator**: Shows connection status to the backend API (top-right corner)

### UI Configuration
The UI connects to the API using the `API_BASE_URL` environment variable (default: `http://localhost:8000`).

---

## Usage Examples

### Ingest Data

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@your_data.zip" \
  -F "clear_first=true"
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is thermal conductivity?",
    "top_k": 5,
    "use_cache": true
  }'
```

### Execute Python

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{"code": "result = (5+10)**2\nprint(result)"}'
```

### Run Evaluation

```bash
python scripts/eval.py
```

---

## Architecture Decisions

**Why PostgreSQL + pgvector?**
Unified infrastructure for relational data and vector similarity. Single datastore for documents, embeddings, and structured data.

**Why Ollama (open-source)?**
No API costs, runs locally, full control over models. Uses llama3.1 for reasoning and nomic-embed-text for embeddings.

**Why Semantic Cache?**
Caches similar queries using cosine similarity > 0.92. Dramatically reduces latency for repeated queries.

**Why Verifier Agent?**
Catches confident hallucinations by strictly comparing answer against retrieved context.

**Why Python Sandbox?**
AST-based validation blocks dangerous imports. Subprocess execution with 10s timeout prevents infinite loops.

**Why Multi-Agent Architecture?**
Distinct agents for retrieval planning and verification allow modular improvement and clearer reasoning traces.

---

## Limitations

- **Semantic Cache**: In-memory (use Redis for production)
- **Python Sandbox**: Runs locally with subprocess (use ephemeral Docker for production)
- **Ollama Dependency**: Requires Ollama running locally
- **Tool Calling**: The core reasoning agent uses standard RAG generation; dynamic tool calling (e.g., deciding to execute code during generation) is not yet implemented
- **Concurrency**: FastAPI is async, but DB connection pooling should be tuned for high concurrency

---

## Features Implemented

### Level 1 — Core
- Ingestion for 5 file types (PDF, TXT, CSV, DOCX, XLSX)
- Retrieval agent that decides where to search (documents vs tables)
- Reasoning agent with entity traversal + SQL generation
- Query API with sources + latency breakdown

### Level 2 — Scalability
- FastAPI async for concurrent queries
- Parallel processing for ingestion
- Semantic caching with 0.92 similarity threshold
- Cache invalidation on new data

### Level 3 — Robustness
- Verifier agent for hallucination detection
- Evaluation script for benchmarking
- Safe Python sandbox with AST validation

---

## Key Architectural Decisions

**1. Hybrid Retrieval (Vector + SQL)**
The system uses PostgreSQL with pgvector for unified storage. This allows semantic search over document chunks and structured SQL queries over experimental data in a single database, simplifying infrastructure and enabling cross-source queries.

**2. Modular Agent Design**
Distinct agents (Retrieval Planner, Verifier) allow independent development and testing. The retrieval planner uses an LLM to decide search strategy, while the verifier fact-checks answers against retrieved context.

**3. Semantic Caching**
Queries are embedded and cached based on cosine similarity (threshold 0.92). This reduces latency for repetitive queries while maintaining freshness through cache invalidation on data ingest.

**4. Safe Code Execution**
Python code execution uses AST validation to block dangerous imports and subprocess calls, with a 10-second timeout to prevent infinite loops. This balances utility with security for scientific calculations.

---

## One Thing I Would Do Differently

If I had more time, I would implement a **ReAct (Reasoning and Acting) agent loop** for the core reasoning agent. Currently, the system uses a standard RAG pattern where retrieval and tool use happen before generation. A ReAct agent would dynamically decide when to:
- Retrieve additional context
- Execute Python code for calculations
- Traverse entity relationships
- Verify intermediate results

This would make the system more flexible and capable of handling complex, multi-step scientific reasoning tasks.

---

## Known Limitations

1. **Tool Calling**: The reasoning agent does not dynamically call tools during generation; tools are used in a fixed pipeline
2. **Sandbox Security**: Python execution uses AST validation but runs in a subprocess; production systems should use containerized sandboxes
3. **Cache Storage**: In-memory cache is lost on restart; production should use Redis
4. **Entity Extraction**: Basic pattern matching for entity extraction; could be improved with NER models

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:password@localhost:5432/sci_rag` |
| `OLLAMA_BASE_URL` | Ollama API URL | `http://localhost:11434/v1` |
| `REASONING_MODEL` | LLM for reasoning | `llama3.1` |
| `VERIFIER_MODEL` | LLM for verification | `llama3.1` |
| `EMBEDDING_MODEL` | Embedding model | `nomic-embed-text` |
| `API_BASE_URL` | Backend API URL for UI | `http://localhost:8000` |
