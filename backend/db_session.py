"""
Database session management for PostgreSQL.
Provides engine, session factory, and connection handling.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load backend/.env then repo-root .env (so local uvicorn sees DATABASE_URL)
_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / "backend" / ".env")
load_dotenv(_root / ".env")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/aistockgpt")

# Railway provides DATABASE_URL with "postgres://" prefix.
# SQLAlchemy 2.x requires "postgresql://" prefix.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Yield a database session for FastAPI dependency injection."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
