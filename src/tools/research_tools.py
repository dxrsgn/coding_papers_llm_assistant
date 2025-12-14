import asyncio

import arxiv
from langchain.tools import tool


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
