from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage

from .devlead import build_coding_agent_subgraph
from .researcher import build_researcher_subgraph
from .supervisor import build_supervisor
from .subagent_wrappers import build_subagent_wrappers
from .state import AgentState


def route_from_supervisor(state: AgentState) -> str:
    num_iterations = state.get("num_iterations", 0)
    if num_iterations >= 3:
        return END
    last_message = state.get("messages", [])[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "supervisor_routing"
    return END


def build_graph():
    graph = StateGraph(AgentState)

    researcher_subgraph = build_researcher_subgraph()
    coding_subgraph = build_coding_agent_subgraph()

    # we dont use there conditional edges to route subagents, instead
    # to propose layered structure we make them as tools for supervisor to call
    # this complies with official langgraph doc https://docs.langchain.com/oss/python/langchain/supervisor#3-wrap-sub-agents-as-tools
    supervisor_tools = build_subagent_wrappers(researcher_subgraph, coding_subgraph)
    supervisor = build_supervisor(supervisor_tools)

    graph.add_node("Supervisor", supervisor)
    graph.add_node("supervisor_routing", ToolNode(supervisor_tools))
    graph.set_entry_point("Supervisor")
    graph.add_conditional_edges("Supervisor", route_from_supervisor)
    graph.add_edge("supervisor_routing", "Supervisor")
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
