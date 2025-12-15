import asyncio

import arxiv
from langchain.tools import tool


# TODO: check whether arxiv lib has async client
@tool
async def search_arxiv(query: str) -> str:
    """Search arXiv for papers matching the query."""
    loop = asyncio.get_event_loop()
    client = arxiv.Client(
        page_size = 3,
        delay_seconds = 3,
        num_retries = 5
    )
    search = arxiv.Search(
        query=query,
        max_results=2,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    entries = []
    def get_results():
        return list(client.results(search))
    results = await loop.run_in_executor(None, get_results)
    for result in results:
        summary = result.summary.strip().replace("\n", " ")
        entries.append(f"Title: {result.title}\nSummary: {summary}")
    if not entries:
        return "No results found."
    return "\n\n".join(entries)
