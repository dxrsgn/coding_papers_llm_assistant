import subprocess
import asyncio
import fnmatch
from pathlib import Path

from langchain.tools import tool


def load_ignore_patterns(directory: Path) -> list[str]:
    patterns = []
    for ignore_file in [".gitignore", ".dockerignore"]:
        ignore_path = directory / ignore_file
        if ignore_path.exists() and ignore_path.is_file():
            try:
                with open(ignore_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            patterns.append(line)
            except Exception:
                pass
    return patterns


def should_ignore(path: Path, ignore_dir: Path, root: Path, patterns: list[str]) -> bool:
    try:
        rel_to_ignore = path.relative_to(ignore_dir)
    except ValueError:
        return False
    
    path_str = str(rel_to_ignore).replace("\\", "/")
    path_parts = path_str.split("/")
    
    for pattern in patterns:
        pattern = pattern.rstrip("/")
        if not pattern:
            continue
        
        if pattern.startswith("/"):
            pattern = pattern[1:]
            if fnmatch.fnmatch(path_parts[0], pattern) or fnmatch.fnmatch(path_str, pattern):
                return True
        elif "/" in pattern:
            if fnmatch.fnmatch(path_str, pattern) or any(fnmatch.fnmatch("/".join(path_parts[i:]), pattern) for i in range(len(path_parts))):
                return True
        else:
            if any(fnmatch.fnmatch(part, pattern) for part in path_parts) or fnmatch.fnmatch(path.name, pattern):
                return True
    
    return False


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
    """Read the content of the file"""
    return "devlead_node must call code_reader_node"


@tool
async def list_directory(directory: str = ".") -> str:
    """List the structure of the current directory or specified directory.
    
    Args:
        directory: The directory path to list. Defaults to current directory ".".
    """
    root = Path.cwd().resolve()
    if not directory or directory == "":
        directory = "."
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
        
        ignore_patterns = load_ignore_patterns(path)
        
        def build_tree(dir_path: Path, prefix: str = "", is_last: bool = True) -> str:
            lines = []
            try:
                items = sorted(dir_path.iterdir(), key=lambda x: (x.is_file(), x.name))
                filtered_items = [item for item in items if not should_ignore(item, path, root, ignore_patterns)]
                
                for i, item in enumerate(filtered_items):
                    is_last_item = i == len(filtered_items) - 1
                    current_prefix = "└── " if is_last_item else "├── "
                    lines.append(f"{prefix}{current_prefix}{item.name}{'/' if item.is_dir() else ''}")
                    
                    if item.is_dir():
                        extension = "    " if is_last_item else "│   "
                        subtree = build_tree(item, prefix + extension, is_last_item)
                        if subtree.strip():
                            lines.append(subtree)
            except PermissionError:
                pass
            return "\n".join(lines)
        
        def get_structure():
            tree = path.name + ("/" if path.is_dir() else "")
            try:
                items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
                filtered_items = [item for item in items if not should_ignore(item, path, root, ignore_patterns)]
                
                if not filtered_items:
                    return tree + "\n└── (empty)"
                tree_lines = [tree]
                for i, item in enumerate(filtered_items):
                    is_last = i == len(filtered_items) - 1
                    current_prefix = "└── " if is_last else "├── "
                    tree_lines.append(f"{current_prefix}{item.name}{'/' if item.is_dir() else ''}")
                    if item.is_dir():
                        extension = "    " if is_last else "│   "
                        subtree = build_tree(item, extension, is_last)
                        if subtree.strip():
                            tree_lines.append(subtree)
                return "\n".join(tree_lines)
            except PermissionError:
                return tree + "\n└── (permission denied)"
        
        return await loop.run_in_executor(None, get_structure)
    except PermissionError:
        return "Permission denied."
    except Exception as exc:
        return f"Error listing directory: {str(exc)}"
