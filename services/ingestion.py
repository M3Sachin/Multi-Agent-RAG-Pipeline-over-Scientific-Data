import os
import zipfile
import tempfile
import shutil
import pandas as pd
from openai import OpenAI
from sqlalchemy.orm import Session
from sqlalchemy import text
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.config import EMBEDDING_MODEL, logger
from core.database import get_engine
from utils.text_extraction import extract_text
from utils.text_chunking import chunk_text
from services.cache import clear_cache

client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"), api_key="ollama"
)

# Settings
EMBEDDING_BATCH_SIZE = 100  # Process embeddings in batches
MAX_WORKERS = 4  # Parallel threads for file processing


def get_file_chunks(file_path: str, rel_path: str, ext: str) -> list:
    """Extract chunks from a single file."""
    if ext == ".csv":
        return None  # Handle CSV separately

    content = extract_text(file_path)
    if content and not content.startswith("Error"):
        chunks = chunk_text(content)
        return [
            {"source": rel_path, "content": chunk, "type": ext}
            for chunk in chunks
            if chunk.strip()
        ]
    return []


def generate_embeddings_batch(chunks: list) -> list:
    """Generate embeddings for a batch of chunks."""
    if not chunks:
        return []

    # Extract content for embedding
    texts = [c["content"] for c in chunks]

    # Batch embedding request
    try:
        resp = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
        embeddings = resp.data

        # Attach embeddings to chunks
        for i, emb in enumerate(embeddings):
            if i < len(chunks):
                chunks[i]["embedding"] = emb.embedding
        return chunks
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return []


async def ingest_zip(
    file_content: bytes, filename: str, clear_first: bool = True
) -> dict:
    """
    Ingest a ZIP file with optimized parallel processing.
    """
    # Clear existing data and cache first
    if clear_first:
        clear_all_data()
        clear_cache()

    engine = get_engine()
    temp_dir = tempfile.mkdtemp()
    extract_dir = os.path.join(temp_dir, "extracted")
    os.makedirs(extract_dir)

    try:
        # Save and extract ZIP
        zip_path = os.path.join(temp_dir, filename)
        with open(zip_path, "wb") as f:
            f.write(file_content)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # Find valid files
        valid_ext = {".pdf", ".txt", ".md", ".csv", ".docx", ".xlsx", ".xls"}
        files_to_process = []
        for root, _, filenames in os.walk(extract_dir):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in valid_ext or ext in {".doc"}:
                    files_to_process.append(os.path.join(root, fname))

        total_chunks = 0
        all_chunks = []

        # Process files in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for file_path in files_to_process:
                ext = os.path.splitext(file_path)[1].lower()
                rel_path = os.path.relpath(file_path, extract_dir)

                if ext == ".csv":
                    # Handle CSV immediately - lowercase table name for consistency
                    table_name = (
                        os.path.splitext(os.path.basename(file_path))[0]
                        .replace("-", "_")
                        .replace(" ", "_")
                        .lower()
                    )
                    try:
                        df = pd.read_csv(file_path)
                        df.to_sql(table_name, engine, if_exists="replace", index=False)
                        total_chunks += len(df)
                    except Exception as e:
                        logger.error(f"Error processing CSV {table_name}: {e}")
                else:
                    # Submit for parallel processing
                    future = executor.submit(get_file_chunks, file_path, rel_path, ext)
                    futures[future] = file_path

            # Collect chunks from parallel processing
            for future in as_completed(futures):
                try:
                    chunks = future.result()
                    if chunks:
                        all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Error processing file: {e}")

        # Process embeddings in batches
        for i in range(0, len(all_chunks), EMBEDDING_BATCH_SIZE):
            batch = all_chunks[i : i + EMBEDDING_BATCH_SIZE]
            chunks_with_embeddings = generate_embeddings_batch(batch)

            if chunks_with_embeddings:
                # Batch insert into database
                with Session(engine) as session:
                    for chunk in chunks_with_embeddings:
                        try:
                            session.execute(
                                text("""
                                    INSERT INTO document_chunks (source_file, content, content_type, embedding)
                                    VALUES (:source, :content, :type, :embedding)
                                """),
                                chunk,
                            )
                        except Exception as e:
                            logger.error(f"Error inserting chunk: {e}")
                    session.commit()
                    total_chunks += len(chunks_with_embeddings)

        return {
            "status": "success",
            "files_processed": len(files_to_process),
            "chunks_created": total_chunks,
            "message": "Processed successfully",
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def clear_all_data() -> dict:
    """Clear all indexed data."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM document_chunks"))
        tables_result = conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name != 'document_chunks'"
            )
        )
        for r in tables_result.fetchall():
            try:
                conn.execute(text(f"DROP TABLE IF EXISTS {r[0]}"))
            except:
                pass
        conn.commit()
    return {"status": "success", "message": "All data cleared"}
