import re
from typing import Dict, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.prompts.devlead import devlead_system_prompt, devlead_user_prompt
from src.prompts.code_reader import code_reader_system_prompt, code_reader_user_prompt
from .state import CoderState
from .tools import get_git_history, get_file_history, read_file_content, call_code_reader


async def devlead_node(state: CoderState, config: Optional[RunnableConfig] = None) -> Dict:
    configurable = (config or {}).get("configurable", {})
    api_base = configurable.get("llm_api_base")
    api_key = configurable.get("llm_api_key")
    model_name = configurable.get("model", "qwen")
    model = ChatOpenAI(model=model_name, temperature=0, base_url=api_base, api_key=api_key)
    model_with_tools = model.bind_tools([get_git_history, get_file_history, read_file_content, call_code_reader])
    
    messages = state.get("messages", [])
    user_query = state.get("user_query") or ""
    research_context = state.get("research_context") or ""
    code_context = state.get("code_context") or ""
    
    user_content = devlead_user_prompt(research_context, code_context, user_query)
    
    if not messages:
        all_messages = [
            SystemMessage(content=devlead_system_prompt),
            HumanMessage(content=user_content),
        ]
    else:
        all_messages = [SystemMessage(content=devlead_system_prompt)] + messages
    
    response = await model_with_tools.ainvoke(all_messages)
    
    result: Dict = {
        "messages": [response],
    }
    
    return result


async def code_reader_node(state: CoderState, config: Optional[RunnableConfig] = None) -> Dict:
    configurable = (config or {}).get("configurable", {})
    api_base = configurable.get("llm_api_base")
    api_key = configurable.get("llm_api_key")
    model_name = configurable.get("model", "qwen")
    model = ChatOpenAI(model=model_name, temperature=0, base_url=api_base, api_key=api_key)
    
    target = None
    tool_call_id = None
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                if tool_call.get("name") == "call_code_reader":
                    args = tool_call.get("args", {})
                    target = args.get("filepath")
                    tool_call_id = tool_call.get("id")
                    break
    
    if not target:
        query = state.get("user_query", "")
        match = re.search(r"[\w./-]+\.[A-Za-z0-9]+", query)
        target = match.group(0) if match else None
    
    content = await read_file_content.ainvoke({"filepath": target}) if target else "No file provided."
    
    all_messages = [
        SystemMessage(content=code_reader_system_prompt),
        HumanMessage(content=code_reader_user_prompt(target or "", content)),
    ]
    
    response = await model.ainvoke(all_messages)
    return {
        "messages": [ToolMessage(content=response.content, tool_call_id=tool_call_id or "")],
    }


async def summarize_code_node(state: CoderState, config: Optional[RunnableConfig] = None) -> Dict:
    configurable = (config or {}).get("configurable", {})
    api_base = configurable.get("llm_api_base")
    api_key = configurable.get("llm_api_key")
    model_name = configurable.get("model", "qwen")
    model = ChatOpenAI(model=model_name, temperature=0, base_url=api_base, api_key=api_key)
    
    messages = state.get("messages", [])
    conversation = get_buffer_string(messages)
    
    summary_prompt = [
        SystemMessage(content="Summarize the following code analysis conversation concisely, focusing on files examined, code structure, and key findings."),
        HumanMessage(content=conversation)
    ]
    
    response = await model.ainvoke(summary_prompt)
    
    return {"code_context": response.content}


def should_continue_devlead(state: CoderState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return "devlead"
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call.get("name") == "call_code_reader":
                return "code_reader"
        return "tools"
    return "summarize"


def build_coding_agent_subgraph():
    subgraph = StateGraph(CoderState)
    subgraph.add_node("devlead", devlead_node)
    subgraph.add_node("tools", ToolNode([get_git_history, get_file_history, call_code_reader]))
    subgraph.add_node("code_reader", code_reader_node)
    subgraph.add_node("summarize", summarize_code_node)
    subgraph.set_entry_point("devlead")
    subgraph.add_conditional_edges("devlead", should_continue_devlead)
    subgraph.add_edge("tools", "devlead")
    subgraph.add_edge("code_reader", "devlead")
    subgraph.add_edge("summarize", END)
    return subgraph.compile()
