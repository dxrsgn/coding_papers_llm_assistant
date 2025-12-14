import os
import hashlib
from pathlib import Path
from typing import Optional

from src.database import fetch_summary, upload_summary


LONG_TERM_MEMORY_DIR = Path(os.getenv("LONG_TERM_MEMORY_DIR", ".cache/agent_memory"))

def _should_use_db(use_db: Optional[bool] = None) -> bool:
    if use_db is not None:
        return use_db
    return bool(os.getenv("DATABASE_URL"))

async def recall_file_summary(content: str, use_db: Optional[bool] = None) -> Optional[str]:
    if _should_use_db(use_db):
        return await fetch_summary(content)
    
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    memory_file = LONG_TERM_MEMORY_DIR / f"{content_hash}.txt"
    if memory_file.exists():
        return memory_file.read_text()
    return None


async def memorize_file_summary(content: str, summary: str, use_db: Optional[bool] = None) -> None:
    if _should_use_db(use_db):
        await upload_summary(content, summary)
        return
    
    LONG_TERM_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    memory_file = LONG_TERM_MEMORY_DIR / f"{content_hash}.txt"
    memory_file.write_text(summary)
