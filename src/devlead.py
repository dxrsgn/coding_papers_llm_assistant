import re
from typing import Dict, Optional
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.prompts.devlead import devlead_prompt
from src.prompts.code_reader import code_reader_prompt
from .state import AgentState
from .tools import get_git_history, read_file_content, call_code_reader


async def devlead_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict:
    configurable = (config or {}).get("configurable", {})
    api_base = configurable.get("llm_api_base")
    api_key = configurable.get("llm_api_key")
    model_name = configurable.get("model", "qwen")
    model = ChatOpenAI(model=model_name, temperature=0, base_url=api_base, api_key=api_key)
    model_with_tools = model.bind_tools([get_git_history, read_file_content, call_code_reader])
    
    messages = state.get("messages", [])
    query = state.get("user_query", "")
    
    if not messages:
        prompt_messages = devlead_prompt.format_messages(
            user_query=query,
        )
        all_messages = prompt_messages
    else:
        system_content = devlead_prompt.format_messages(
            user_query=query,
        )[0].content
        all_messages = [SystemMessage(content=system_content)] + messages
    
    response = await model_with_tools.ainvoke(all_messages)
    
    result: Dict = {
        "messages": [response],
    }
    
    if isinstance(response, AIMessage) and not response.tool_calls:
        active_agents = state.get("active_agents", [])
        if "DevLead" not in active_agents:
            result["active_agents"] = active_agents + ["DevLead"]
    
    return result


async def code_reader_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict:
    configurable = (config or {}).get("configurable", {})
    api_base = configurable.get("llm_api_base")
    api_key = configurable.get("llm_api_key")
    model_name = configurable.get("model", "qwen")
    model = ChatOpenAI(model=model_name, temperature=0, base_url=api_base, api_key=api_key)
    chain = code_reader_prompt | model
    
    target = None
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                if tool_call.get("name") == "call_code_reader":
                    args = tool_call.get("args", {})
                    target = args.get("filepath")
                    break
    
    if not target:
        query = state.get("user_query", "")
        match = re.search(r"[\w./-]+\.[A-Za-z0-9]+", query)
        target = match.group(0) if match else None
    
    content = await read_file_content.ainvoke({"filepath": target}) if target else "No file provided."
    response = await chain.ainvoke(
        {
            "filepath": target or "",
            "file_content": content,
        }
    )
    return {
        "messages": [AIMessage(content=response.content)],
        "active_agents": state.get("active_agents", []) + ["CodeReader"],
    }


def should_continue_devlead(state: AgentState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return "devlead"
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call.get("name") == "call_code_reader":
                return "code_reader"
        return "tools"
    return END


def build_coding_agent_subgraph():
    subgraph = StateGraph(AgentState)
    subgraph.add_node("devlead", devlead_node)
    subgraph.add_node("tools", ToolNode([get_git_history, call_code_reader]))
    subgraph.add_node("code_reader", code_reader_node)
    subgraph.set_entry_point("devlead")
    subgraph.add_conditional_edges("devlead", should_continue_devlead)
    subgraph.add_edge("tools", "devlead")
    subgraph.add_edge("code_reader", "devlead")
    return subgraph.compile()

