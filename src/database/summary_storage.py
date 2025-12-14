import hashlib
from typing import Optional, List, Dict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import get_async_session, FileSummary


def get_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


async def fetch_summary(content: str, filepath: Optional[str] = None) -> Optional[str]:
    content_hash = get_content_hash(content)
    
    async with get_async_session() as session:
        result = await session.execute(
            select(FileSummary).filter(FileSummary.content_hash == content_hash)
        )
        file_summary = result.scalar_one_or_none()
        
        if file_summary:
            return file_summary.summary
    
    return None


async def upload_summary(content: str, summary: str, filepath: Optional[str] = None) -> None:
    content_hash = get_content_hash(content)
    
    async with get_async_session() as session:
        result = await session.execute(
            select(FileSummary).filter(FileSummary.content_hash == content_hash)
        )
        existing_summary = result.scalar_one_or_none()
        
        if existing_summary:
            existing_summary.summary = summary
            if filepath:
                existing_summary.filepath = filepath
        else:
            new_summary = FileSummary(
                content_hash=content_hash,
                filepath=filepath,
                summary=summary
            )
            session.add(new_summary)
        
        await session.commit()


async def fetch_summary_by_filepath(filepath: str) -> Optional[str]:
    async with get_async_session() as session:
        result = await session.execute(
            select(FileSummary)
            .filter(FileSummary.filepath == filepath)
            .order_by(FileSummary.updated_at.desc())
        )
        file_summary = result.scalar_one_or_none()
        
        if file_summary:
            return file_summary.summary
    
    return None


async def list_all_summaries(limit: Optional[int] = None) -> List[Dict]:
    async with get_async_session() as session:
        query = select(FileSummary).order_by(FileSummary.updated_at.desc())
        
        if limit:
            query = query.limit(limit)
        
        result = await session.execute(query)
        summaries = result.scalars().all()
        
        return [
            {
                "id": s.id,
                "content_hash": s.content_hash,
                "filepath": s.filepath,
                "summary": s.summary,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "updated_at": s.updated_at.isoformat() if s.updated_at else None,
            }
            for s in summaries
        ]
