from typing import Dict, Optional, List, Callable, Coroutine, Any
from functools import partial
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from .prompts.supervisor import supervisor_system_prompt, supervisor_user_prompt
from .utils import create_llm
from .state import AgentState


async def supervisor_node(state: AgentState, subagents: List[BaseTool], config: Optional[RunnableConfig] = None) -> Dict:
    configurable = (config or {}).get("configurable", {})
    api_base = configurable.get("llm_api_base", None)
    api_key = configurable.get("llm_api_key", None)
    model_name = configurable.get("model", "qwen")
    llm = create_llm(model=model_name, temperature=0, base_url=api_base, api_key=api_key)
    llm = llm.bind_tools(subagents)
    
    query = state.get("user_query") or ""
    state_messages = state.get("messages") or []
    research_context = state.get("research_context") or ""
    code_context = state.get("code_context") or ""
    
    user_content = supervisor_user_prompt(query, research_context, code_context)
    
    if not state_messages:
        all_messages = [
            SystemMessage(content=supervisor_system_prompt),
            HumanMessage(content=user_content),
        ]
    else:
        all_messages = [SystemMessage(content=supervisor_system_prompt)] + state_messages
    
    response = await llm.ainvoke(all_messages)
    
    return {"messages": [response], "num_iterations": state.get("num_iterations", 0) + 1}


def build_supervisor(subagents: List[BaseTool]):
    # inject tools into supervisor node
    return partial(supervisor_node, subagents=subagents)
