from typing import Dict, Literal, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from .prompts.supervisor import supervisor_system_prompt, supervisor_user_prompt
from .utils import create_llm, StructuredRetryRunnable
from .state import AgentState


class TaskDelegation(BaseModel):
    agent: Literal["DEV", "RESEARCH"] = Field(
        description="The agent to delegate to: DEV for code tasks, RESEARCH for research tasks"
    )
    task: str = Field(
        description="Specific task description for the agent"
    )


class SupervisorDecision(BaseModel):
    reasoning: str = Field(
        description="Brief reasoning about what needs to be done"
    )
    next_action: Literal["DELEGATE", "FINISH"] = Field(
        description="DELEGATE to assign task to an agent, FINISH when the query is fully answered"
    )
    delegation: Optional[TaskDelegation] = Field(
        default=None,
        description="Task delegation details when next_action is DELEGATE"
    )


async def supervisor_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict:
    configurable = (config or {}).get("configurable", {})
    api_base = configurable.get("llm_api_base", None)
    api_key = configurable.get("llm_api_key", None)
    model_name = configurable.get("model", "qwen")
    llm = create_llm(model=model_name, temperature=0, base_url=api_base, api_key=api_key)
    
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
        all_messages = [
            SystemMessage(content=supervisor_system_prompt),
            HumanMessage(content=user_content),
        ] + state_messages
    
    structured_chain = StructuredRetryRunnable(llm=llm, model_class=SupervisorDecision)
    decision = await structured_chain.ainvoke(all_messages)
    
    result = {
        "next_action": decision.next_action,
    }
    
    if decision.next_action == "DELEGATE" and decision.delegation:
        result["route"] = decision.delegation.agent
        result["current_task"] = decision.delegation.task
    
    return result

