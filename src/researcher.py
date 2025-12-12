from typing import Dict, Optional
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.prompts.researcher import researcher_prompt
from .state import AgentState
from .tools import search_arxiv


def should_continue_research(state: AgentState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return "researcher"
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


async def researcher_agent_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict:
    configurable = (config or {}).get("configurable", {})
    api_base = configurable.get("llm_api_base")
    api_key = configurable.get("llm_api_key")
    model_name = configurable.get("model", "qwen")
    model = ChatOpenAI(model=model_name, temperature=0, base_url=api_base, api_key=api_key)
    model_with_tools = model.bind_tools([search_arxiv])
    
    messages = state.get("messages", [])
    
    if not messages:
        prompt_messages = researcher_prompt.format_messages(
            user_query=state.get("user_query", ""),
        )
        all_messages = prompt_messages
    else:
        system_content = researcher_prompt.format_messages(
            user_query=state.get("user_query", ""),
        )[0].content
        all_messages = [SystemMessage(content=system_content)] + messages
    
    response = await model_with_tools.ainvoke(all_messages)
    
    result: Dict = {
        "messages": [response],
    }
    
    if isinstance(response, AIMessage) and not response.tool_calls:
        active_agents = state.get("active_agents", [])
        if "Researcher" not in active_agents:
            result["active_agents"] = active_agents + ["Researcher"]
    
    return result


def build_researcher_subgraph():
    subgraph = StateGraph(AgentState)
    subgraph.add_node("researcher", researcher_agent_node)
    subgraph.add_node("tools", ToolNode([search_arxiv]))
    subgraph.set_entry_point("researcher")
    subgraph.add_conditional_edges("researcher", should_continue_research)
    subgraph.add_edge("tools", "researcher")
    return subgraph.compile()