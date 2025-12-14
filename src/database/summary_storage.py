import hashlib
from typing import Optional, List, Dict

from .models import get_session, FileSummary


def get_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def fetch_summary(content: str, filepath: Optional[str] = None) -> Optional[str]:
    content_hash = get_content_hash(content)
    
    with get_session() as session:
        file_summary = session.query(FileSummary).filter(
            FileSummary.content_hash == content_hash
        ).first()
        
        if file_summary:
            return file_summary.summary
    
    return None


def upload_summary(content: str, summary: str, filepath: Optional[str] = None) -> None:
    content_hash = get_content_hash(content)
    
    with get_session() as session:
        existing_summary = session.query(FileSummary).filter(
            FileSummary.content_hash == content_hash
        ).first()
        
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
        
        session.commit()


def fetch_summary_by_filepath(filepath: str) -> Optional[str]:
    with get_session() as session:
        file_summary = session.query(FileSummary).filter(
            FileSummary.filepath == filepath
        ).order_by(FileSummary.updated_at.desc()).first()
        
        if file_summary:
            return file_summary.summary
    
    return None


def list_all_summaries(limit: Optional[int] = None) -> List[Dict]:
    with get_session() as session:
        query = session.query(FileSummary).order_by(FileSummary.updated_at.desc())
        
        if limit:
            query = query.limit(limit)
        
        summaries = query.all()
        
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
