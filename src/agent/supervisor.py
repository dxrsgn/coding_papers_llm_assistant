from typing import Dict, Optional, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from .prompts.supervisor import supervisor_system_prompt, supervisor_user_prompt
from .utils import create_llm, normalize_message_content
from .state import AgentState


# purely programmatic node
# to preprocess user input and add it to state
# removing this logic from supervisor node itself
# as other way, we can move this logic into the main.py/app.py, i.e. to client itself
async def prepare_user_input(state: AgentState, config: Optional[RunnableConfig] = None) -> dict:
    query = state.get("user_query") or ""
    research_context = state.get("research_context") or ""
    code_context = state.get("code_context") or ""
    user_content = supervisor_user_prompt(query, research_context, code_context)
    return {"messages": [HumanMessage(content=user_content)]}
    


async def supervisor_node(state: AgentState, subagents: List[BaseTool], config: Optional[RunnableConfig] = None) -> Dict:
    configurable = (config or {}).get("configurable", {})
    api_base = configurable.get("llm_api_base", None)
    api_key = configurable.get("llm_api_key", None)
    model_name = configurable.get("model", "qwen")
    llm = create_llm(model=model_name, temperature=0, base_url=api_base, api_key=api_key)
    llm = llm.bind_tools(subagents)
    
    state_messages = state.get("messages") or []
    
    # it seems like various providers (e.g. at openrouter and at itmo's cluster may have different response formats)
    normalized_messages = [normalize_message_content(msg) for msg in state_messages]
    
    response = await llm.ainvoke(
        [SystemMessage(content=supervisor_system_prompt)] + normalized_messages
    )
    
    return {
        "messages": [response],
        "num_iterations": state.get("num_iterations", 0) + 1
    }


def build_supervisor(subagents: List[BaseTool]):
    async def supervisor(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict:
        return await supervisor_node(state, subagents, config)
    return supervisor
