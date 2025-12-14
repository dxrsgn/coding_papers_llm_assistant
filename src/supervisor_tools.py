from langchain_core.tools import tool

from .state import ResearcherState, CoderState


def build_supervisor_tools(researcher_subgraph, coding_subgraph):
    # functions to separate contexts of subgraph from main graph
    # and adapt inputs to subgraph's state
    # complies with official langgraph documentation https://docs.langchain.com/oss/python/langgraph/use-subgraphs
    @tool
    async def call_researcher(task: str) -> str:
        """Delegate a research task to the RESEARCH agent."""
        input_state = {
            "messages": [],
            "user_query": task,
            "research_context": "",
            "code_context": "",
        }
        result = await researcher_subgraph.ainvoke(ResearcherState(**input_state))
        return result.get("research_context", "No research context found")

    @tool
    async def call_coder(task: str) -> str:
        """Delegate a coding task to the DEV agent."""
        input_state = {
            "messages": [],
            "user_query": task,
            "research_context": "",
            "code_context": "",
        }
        result = await coding_subgraph.ainvoke(CoderState(**input_state))
        return result.get("code_context", "No code context found")

    return [call_researcher, call_coder]
