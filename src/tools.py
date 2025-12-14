import subprocess
import asyncio
from pathlib import Path

import arxiv
from langchain.tools import tool


@tool
async def get_git_history(limit: int = 5) -> str:
    """Get git commit history with the specified limit."""
    if limit <= 0:
        limit = 5
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                ["git", "log", f"-n{limit}", "--name-status"],
                check=True,
                capture_output=True,
                text=True,
            )
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as exc:
        return exc.stderr.strip() or "Unable to read git history"
    except FileNotFoundError:
        return "git is not available"


@tool
async def search_arxiv(query: str) -> str:
    """Search arXiv for papers matching the query."""
    loop = asyncio.get_event_loop()
    search = arxiv.Search(
        query=query,
        max_results=2,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    entries = []
    def get_results():
        return list(search.results())
    results = await loop.run_in_executor(None, get_results)
    for result in results:
        summary = result.summary.strip().replace("\n", " ")
        entries.append(f"Title: {result.title}\nSummary: {summary}")
    if not entries:
        return "No results found."
    return "\n\n".join(entries)


@tool
async def get_file_history(filepath: str, limit: int = 3) -> str:
    """Get git commit history and diffs for a specific file."""
    if limit <= 0:
        limit = 3
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                ["git", "log", f"-n{limit}", "-p", "--", filepath],
                check=True,
                capture_output=True,
                text=True,
            )
        )
        return result.stdout.strip() or "No history found for this file."
    except subprocess.CalledProcessError as exc:
        return exc.stderr.strip() or "Unable to get file history"
    except FileNotFoundError:
        return "git is not available"


@tool
async def read_file_content(filepath: str) -> str:
    """Read the content of a file from the project directory."""
    root = Path.cwd().resolve()
    path = Path(filepath).expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return "Access denied."
    if not path.exists():
        return "File not found."
    if not path.is_file():
        return "Target is not a file."
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, path.read_text)
    except UnicodeDecodeError:
        return "Unable to read file as text."


@tool
async def call_code_reader(filepath: str) -> str:
    """Call the code reader node to read the file content."""
    return "devlead_node must call code_reader_node"


@tool
async def list_directory(directory: str = ".") -> str:
    """List the structure of the current directory or specified directory."""
    root = Path.cwd().resolve()
    path = Path(directory).expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return "Access denied."
    if not path.exists():
        return "Directory not found."
    if not path.is_dir():
        return "Target is not a directory."
    try:
        loop = asyncio.get_event_loop()
        def get_structure():
            items = []
            for item in sorted(path.iterdir()):
                if item.is_dir():
                    items.append(f"{item.name}/")
                else:
                    items.append(item.name)
            return "\n".join(items) if items else "Directory is empty."
        return await loop.run_in_executor(None, get_structure)
    except PermissionError:
        return "Permission denied."
    except Exception as exc:
        return f"Error listing directory: {str(exc)}"

