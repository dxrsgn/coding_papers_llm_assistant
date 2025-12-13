from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .devlead import build_coding_agent_subgraph
from .researcher import build_researcher_subgraph
from .supervisor import supervisor_node
from .state import AgentState, ResearcherState, CoderState


def route_from_supervisor(state: AgentState) -> str:
    next_action = (state.get("next_action") or "").upper()
    if next_action == "FINISH":
        return "END"
    route = (state.get("route") or "").upper()
    if route == "DEV":
        return "DEV"
    return "RESEARCH"


def build_graph():
    graph = StateGraph(AgentState)

    researcher_subgraph = build_researcher_subgraph()
    coding_subgraph = build_coding_agent_subgraph()

    async def call_researcher(state: AgentState) -> dict:
        input_state = {
            "user_query": state.get("current_task") or state.get("user_query", ""),
            "research_context": state.get("research_context", ""),
            "code_context": state.get("code_context", ""),
        }
        result = await researcher_subgraph.ainvoke(ResearcherState(**input_state))
        return {"messages": result.get("messages", [])[-1], "research_context": result.get("research_context", "")}

    async def call_coder(state: AgentState) -> dict:
        input_state = {
            "user_query": state.get("current_task") or state.get("user_query", ""),
            "research_context": state.get("research_context", ""),
            "code_context": state.get("code_context", ""),
        }
        result = await coding_subgraph.ainvoke(CoderState(**input_state))
        return {"messages": result.get("messages", [])[-1], "code_context": result.get("code_context", "")}

    graph.add_node("Supervisor", supervisor_node)
    graph.add_node("Researcher", call_researcher)
    graph.add_node("DevLead", call_coder)
    graph.set_entry_point("Supervisor")
    graph.add_conditional_edges(
        "Supervisor",
        route_from_supervisor,
        {
            "DEV": "DevLead",
            "RESEARCH": "Researcher",
            "END": END,
        },
    )
    graph.add_edge("Researcher", "Supervisor")
    graph.add_edge("DevLead", "Supervisor")
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
