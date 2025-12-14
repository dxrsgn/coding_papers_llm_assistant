import os
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from .state import ResearcherState, CoderState


def build_subagent_wrappers(researcher_subgraph, coding_subgraph):
    use_db = bool(os.getenv("DATABASE_URL"))
    
    @tool
    async def call_researcher(task: str, config: RunnableConfig) -> str:
        """Delegate a research task to the RESEARCH agent."""
        input_state = {
            "messages": [],
            "user_query": task,
            "research_context": "",
            "code_context": "",
        }
        configurable = config.get("configurable", {}) if config else {}
        configurable |= {"use_db": use_db}
        config = RunnableConfig(configurable=configurable)
        result = await researcher_subgraph.ainvoke(ResearcherState(**input_state), config=config)
        return result.get("research_context", "No research context found")

    @tool
    async def call_coder(task: str, config: RunnableConfig) -> str:
        """Delegate a coding task to the DEV agent."""
        input_state = {
            "messages": [],
            "user_query": task,
            "research_context": "",
            "code_context": "",
        }
        configurable = config.get("configurable", {}) if config else {}
        configurable |= {"use_db": use_db}
        config = RunnableConfig(configurable=configurable)
        result = await coding_subgraph.ainvoke(CoderState(**input_state), config=config)
        return result.get("code_context", "No code context found")

    return [call_researcher, call_coder]
