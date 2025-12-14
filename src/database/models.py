from datetime import datetime
import os
from typing import Optional

from sqlalchemy import (
    DateTime,
    Integer,
    String,
    Text,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import (
    Mapped,
    declarative_base,
    mapped_column,
)

Base = declarative_base()

_async_engine = None
_AsyncSessionLocal: Optional[async_sessionmaker] = None


def get_async_engine():
    global _async_engine
    if _async_engine is None:
        database_url = os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/postgres")
        if database_url.startswith("postgresql+psycopg://"):
            async_url = database_url
        elif database_url.startswith("postgresql://"):
            async_url = database_url.replace("postgresql://", "postgresql+psycopg://")
        else:
            async_url = database_url
        _async_engine = create_async_engine(async_url, echo=False)
    return _async_engine


def get_async_sessionmaker() -> async_sessionmaker:
    global _AsyncSessionLocal
    if _AsyncSessionLocal is None:
        _AsyncSessionLocal = async_sessionmaker(bind=get_async_engine(), class_=AsyncSession, expire_on_commit=False)
    return _AsyncSessionLocal


def get_async_session() -> AsyncSession:
    return get_async_sessionmaker()()


class FileSummary(Base):
    __tablename__ = "file_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    filepath: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


async def init_db():
    async_engine = get_async_engine()
    async with async_engine.begin() as conn:
        await conn.run_sync(lambda sync_conn: Base.metadata.create_all(bind=sync_conn))

