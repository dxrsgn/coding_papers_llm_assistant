from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .devlead import build_coding_agent_subgraph
from .researcher import build_researcher_subgraph
from .router import router_node
from .state import AgentState, ResearcherState, CoderState


def route_from_router(state: AgentState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return "RESEARCH"
    
    last_message = messages[-1]
    route = last_message.additional_kwargs.get("route", "").upper()
    if route == "DEV":
        return "DEV"
    return "RESEARCH"


def build_graph():
    graph = StateGraph(AgentState)

    researcher_subgraph = build_researcher_subgraph()
    coding_subgraph = build_coding_agent_subgraph()

    # functions to separate contexts of subgraph from main graph
    # and adapt inputs to subgraph's state
    # complies with official langgraph documentation https://docs.langchain.com/oss/python/langgraph/use-subgraphs
    async def call_researcher(state: AgentState) -> dict:
        input_state = {
            "user_query": state.get("user_query", ""),
            "research_context": state.get("research_context", ""),
            "code_context": state.get("code_context", ""),
        }
        result = await researcher_subgraph.ainvoke(ResearcherState(**input_state))
        # return last message of subgraph into shared message history
        return {"messages": result.get("messages", [])[-1], "research_context": result.get("research_context", "")}

    async def call_coder(state: AgentState) -> dict:
        input_state = {
            "user_query": state.get("user_query", ""),
            "research_context": state.get("research_context", ""),
            "code_context": state.get("code_context", ""),
        }
        result = await coding_subgraph.ainvoke(CoderState(**input_state))
        return {"messages": result.get("messages", [])[-1], "code_context": result.get("code_context", "")}

    graph.add_node("Router", router_node)
    graph.add_node("Researcher", call_researcher)
    graph.add_node("DevLead", call_coder)
    graph.set_entry_point("Router")
    graph.add_conditional_edges(
        "Router",
        route_from_router,
        {
            "DEV": "DevLead",
            "RESEARCH": "Researcher",
            "__else__": "Researcher",
        },
    )
    graph.add_edge("Researcher", END)
    graph.add_edge("DevLead", END)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)

