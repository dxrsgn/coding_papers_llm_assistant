from typing import Dict, Literal, Optional
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from src.prompts.router import router_prompt
from .utils import create_llm, StructuredRetryRunnable
from .state import AgentState


class RouterDecision(BaseModel):
    """Routing decision for directing user queries to appropriate agents."""
    route: Literal["DEV", "RESEARCH"] = Field(
        description="The route to take: DEV for code-related queries, RESEARCH for research/theory-related queries"
    )


async def router_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict:
    configurable = (config or {}).get("configurable", {})
    api_base = configurable.get("llm_api_base", None)
    api_key = configurable.get("llm_api_key", None)
    model_name = configurable.get("model", "qwen")
    llm = create_llm(model=model_name, temperature=0, base_url=api_base, api_key=api_key)
    
    query = state.get("user_query", "")
    messages = router_prompt.format_messages(user_query=query)
    
    structured_chain = StructuredRetryRunnable(llm=llm, model_class=RouterDecision)
    decision = await structured_chain.ainvoke(messages)
    
    message = AIMessage(
        content=f"Route decision: {decision.route}",
        additional_kwargs={"route": decision.route}
    )
    
    return {
        "messages": [message],
        "active_agents": state.get("active_agents", []) + ["Router"],
    }

