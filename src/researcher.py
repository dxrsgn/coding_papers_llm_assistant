from typing import Dict, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.prompts.researcher import researcher_system_prompt, researcher_user_prompt
from .state import AgentState
from .tools import search_arxiv


def should_continue_research(state: AgentState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return "researcher"
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "summarize"


async def researcher_agent_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict:
    configurable = (config or {}).get("configurable", {})
    api_base = configurable.get("llm_api_base")
    api_key = configurable.get("llm_api_key")
    model_name = configurable.get("model", "qwen")
    model = ChatOpenAI(model=model_name, temperature=0, base_url=api_base, api_key=api_key)
    model_with_tools = model.bind_tools([search_arxiv])
    
    messages = state.get("messages", [])
    user_query = state.get("user_query") or ""
    research_context = state.get("research_context") or ""
    code_context = state.get("code_context") or ""
    
    user_content = researcher_user_prompt(research_context, code_context, user_query)
    
    if not messages:
        all_messages = [
            SystemMessage(content=researcher_system_prompt),
            HumanMessage(content=user_content),
        ]
    else:
        all_messages = [SystemMessage(content=researcher_system_prompt)] + messages
    
    response = await model_with_tools.ainvoke(all_messages)
    
    result: Dict = {
        "messages": [response],
    }
    
    return result


async def summarize_research_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict:
    configurable = (config or {}).get("configurable", {})
    api_base = configurable.get("llm_api_base")
    api_key = configurable.get("llm_api_key")
    model_name = configurable.get("model", "qwen")
    model = ChatOpenAI(model=model_name, temperature=0, base_url=api_base, api_key=api_key)
    
    messages = state.get("messages", [])
    conversation = get_buffer_string(messages)
    
    summary_prompt = [
        SystemMessage(content="Summarize the following research conversation concisely, focusing on key findings, papers found, and conclusions."),
        HumanMessage(content=conversation)
    ]
    
    response = await model.ainvoke(summary_prompt)
    
    return {"research_context": response.content}


def build_researcher_subgraph():
    subgraph = StateGraph(AgentState)
    subgraph.add_node("researcher", researcher_agent_node)
    subgraph.add_node("tools", ToolNode([search_arxiv]))
    subgraph.add_node("summarize", summarize_research_node)
    subgraph.set_entry_point("researcher")
    subgraph.add_conditional_edges("researcher", should_continue_research)
    subgraph.add_edge("tools", "researcher")
    subgraph.add_edge("summarize", END)
    return subgraph.compile()
