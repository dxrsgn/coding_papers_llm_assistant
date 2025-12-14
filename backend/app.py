from dotenv import load_dotenv
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from src.agent.graph import build_graph
from src.agent.state import AgentState

from phoenix.otel import register
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor

load_dotenv()

tracer_provider = register(
    project_name="aboba",
    auto_instrument=False,
)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
HTTPXClientInstrumentor().instrument()
tracer = tracer_provider.get_tracer("aboba")

app = FastAPI()
graph = build_graph()


class MessageRequest(BaseModel):
    message: str
    thread_id: str = "default_thread"


class MessageResponse(BaseModel):
    response: str


@app.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest):
    try:
        config = RunnableConfig(
            configurable={
                "thread_id": request.thread_id,
                "llm_api_base": os.getenv("OPENAI_API_BASE"),
                "llm_api_key": os.getenv("OPENAI_API_KEY"),
                "model": os.getenv("MODEL"),
            }
        )
        
        existing_state = graph.get_state(config=config)
        messages = existing_state.values.get("messages", [])
        input_state = {
            **existing_state.values,
            "messages": messages,
            "user_query": request.message,
        }
        
        result = await graph.ainvoke(AgentState(**input_state), config=config)
        messages = result.get("messages", [])
        last_message = messages[-1]
        
        if not isinstance(last_message, AIMessage):
            raise ValueError("Last message is not an AIMessage")

        if isinstance(last_message.content, list):
            final_response = ""
            for part in last_message.content:
                if isinstance(part, str):
                    final_response += part
                elif isinstance(part, dict) and "text" in part:
                    final_response += part["text"]
        elif isinstance(last_message.content, str):
            final_response = last_message.content
        else:
            final_response = str(last_message.content)

        return MessageResponse(response=final_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
