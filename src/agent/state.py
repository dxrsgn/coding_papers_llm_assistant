from typing import Annotated, List, Optional, TypedDict
from operator import add

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    # previous queries are also stored there as human messages
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    research_context: Optional[str]
    code_context: Optional[str]
    num_iterations: int

# creating separate states in order to handle them
# as sepearate subgraphs where messages field is a history of only specific agent
class CoderState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    code_context: Optional[str]
    research_context: Optional[str]

class ResearcherState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    research_context: Optional[str]
    code_context: Optional[str]

