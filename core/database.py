import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from core.config import DATABASE_URL

engine = create_engine(DATABASE_URL)


def init_database():
    """Initialize database with required extensions and tables."""
    with Session(engine) as session:
        session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        session.commit()

    with engine.connect() as conn:
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                source_file VARCHAR(500),
                content TEXT,
                content_type VARCHAR(50),
                embedding vector(768)
            )
        """)
        )
        conn.commit()


def get_engine():
    return engine
