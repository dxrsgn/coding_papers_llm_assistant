from langgraph.graph import END, StateGraph

from .devlead import build_coding_agent_subgraph
from .researcher import build_researcher_subgraph
from .router import router_node
from .state import AgentState


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
    graph.add_node("Router", router_node)
    graph.add_node("Researcher", build_researcher_subgraph())
    graph.add_node("DevLead", build_coding_agent_subgraph())
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
    return graph.compile()

