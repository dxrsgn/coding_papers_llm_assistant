from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    route: Optional[str]
    research_context: Optional[str]
    code_context: Optional[str]
    # NOTE: though assingment requirements suggest to store in main state
    # final answer and intermediate states (like classification & partial results & etc)
    # i decide to make main state lean, because all this information is only needed in subgraphs
    # and final answer seems redundant, because it is already stored in messages field

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

